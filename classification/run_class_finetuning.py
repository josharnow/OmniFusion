# Add these imports at the top of the file
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt # Added for plotting

# --- ADD THIS DEBUGGING BLOCK AT THE VERY TOP ---
import torch
import sys
try:
    print("--- Checking PyTorch and CUDA environment ---")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    print("--- Environment check complete ---")
except Exception as e:
    print(f"!!! An error occurred during environment check: {e} !!!")
# --- END DEBUGGING BLOCK ---

import argparse
import datetime
import numpy as np
import time
# import torch # Already imported above
import torch.backends.cudnn as cudnn
import json
import os
import wandb
import pandas as pd
import torchvision.transforms as transforms
from datasets.derm_data import Uni_Dataset
from pathlib import Path
from models.modeling_finetune import *

from timm.data.mixup import Mixup
# from timm.models import create_model # Not used directly, model created based on name
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from furnace.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
from torch.utils.data import WeightedRandomSampler
from furnace.engine_for_finetuning import train_one_epoch, evaluate, evaluate_tta
from furnace.utils import NativeScalerWithGradNormCount as NativeScaler
import torch.nn.functional as F
from torch import nn
import furnace.utils as utils
from scipy import interpolate
from typing import Any, Dict


def load_checkpoint(path: str, map_location: str = 'cpu', allow_fallback: bool = True):
    """Load a checkpoint with PyTorch 2.6 safety defaults while allowing trusted fallbacks."""
    load_kwargs: Dict[str, Any] = {'map_location': map_location}
    used_weights_only = False

    serialization = getattr(torch, 'serialization', None)
    if serialization is not None and hasattr(serialization, 'add_safe_globals'):
        try:
            allowlist = []
            np_multiarray = getattr(np.core, 'multiarray', None)
            scalar_cls = getattr(np_multiarray, 'scalar', None) if np_multiarray is not None else None
            if scalar_cls is not None:
                allowlist.append(scalar_cls)
            if allowlist:
                serialization.add_safe_globals(allowlist)
            load_kwargs['weights_only'] = True
            used_weights_only = True
        except Exception as safe_err:
            print(f"Warning: Failed to register safe globals for numpy scalar when loading {path}: {safe_err}")

    try:
        return torch.load(path, **load_kwargs)
    except Exception as load_err:
        if used_weights_only and allow_fallback:
            print(f"Safe load failed for {path} (weights_only=True). Retrying with weights_only=False. Error: {load_err}")
            load_kwargs.pop('weights_only', None)
            return torch.load(path, **load_kwargs)
        raise

# --- NEW, CORRECTED LOSS CLASS ---
class WeightedLabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing and class weights.
    """
    def __init__(self, smoothing=0.1, class_weights=None):
        super(WeightedLabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.class_weights = class_weights

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # --- START FIX: Cast logits to float32 for stable log_softmax ---
        # This prevents overflow to Inf when x is in float16
        logprobs = F.log_softmax(x.float(), dim=-1)
        # --- END FIX ---

        # Handle soft targets (from mixup)
        if target.ndim == 2:
            # Soft targets are already a distribution, just apply cross-entropy
            loss = - (target * logprobs).sum(dim=-1)
            # Find the original class index to apply weights
            target_indices = target.argmax(dim=-1)
        # Handle hard targets
        else:
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            target_indices = target

        # Apply class weights
        if self.class_weights is not None:
            weights = self.class_weights[target_indices]
            loss = loss * weights

        return loss.mean()


def get_args():
    parser = argparse.ArgumentParser('fine-tuning and evaluation script for image classification', add_help=False)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # Model parameters
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--sin_pos_emb', action='store_true')
    parser.set_defaults(sin_pos_emb=True)
    parser.add_argument('--disable_sin_pos_emb', action='store_false', dest='sin_pos_emb')

    parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")
    parser.add_argument('--ood_eval', action='store_true', default=False,
                        help='whether conduct zero-shot evaluation')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--weights', action='store_true', default=False, help='Use weighted sampling')
    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')
    parser.add_argument('--percent_data', default=1.0, type=float)

    # TTA
    parser.add_argument('--TTA', action='store_true', default=False)
    # train monitor
    parser.add_argument('--monitor', default='acc', type=str, help='monitor used in training')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--pretrained_checkpoint', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module|state_dict', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--test_csv_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--image_key', default='image', type=str,
                        help='image columns used in dataframe')
    parser.add_argument('--nb_classes', default=0, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')

    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'IMNET100', 'image_folder'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--csv_path', default=None, type=str,
                        help='csv file path')
    parser.add_argument('--root_path', default=None, type=str,
                        help='image root path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--wandb_name', default='demo', type=str,
                        help='wandb name')
    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--enable_linear_eval', action='store_true', default=False)
    parser.add_argument('--enable_multi_print', action='store_true',default=False, help='allow each gpu prints something')
    parser.add_argument('--disable_amp', type=int, default=0, help='Disable automatic mixed precision training (train in float32).')


    parser.add_argument('--exp_name', default='', type=str,
                        help='name of exp. it is helpful when save the checkpoint')

    # --- ADD EARLY STOPPING ARGUMENTS ---
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Number of epochs to wait for improvement before stopping (0 to disable).')
    parser.add_argument('--early_stopping_delta', type=float, default=0.001,
                        help='Minimum change in the monitored metric to qualify as an improvement.')
    # --- END EARLY STOPPING ARGUMENTS ---


    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except ImportError:
            print("Please 'pip install deepspeed'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init

# --- NEW PLOTTING FUNCTION ---
def plot_training_curves(history, output_dir, epochs_completed):
    """
    Plots and saves training and validation metrics.

    Args:
        history (dict): A dictionary containing lists of metrics per epoch.
                        Expected keys: 'train_loss', 'val_loss', 'val_acc',
                                       'val_bacc', 'val_auc'.
        output_dir (str): The directory to save the plot images.
        epochs_completed (int): The number of epochs actually completed.
    """
    epochs_range = range(epochs_completed) # Use actual completed epochs

    plt.figure(figsize=(18, 5))

    # Plot Loss
    plt.subplot(1, 3, 1)
    if history['train_loss']: # Check if list is not empty
        plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    if history['val_loss']:
        plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # Plot Accuracy (Overall and Balanced)
    plt.subplot(1, 3, 2)
    if history['val_acc']:
        plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    if history['val_bacc']:
        plt.plot(epochs_range, history['val_bacc'], label='Validation Balanced Accuracy')
    plt.legend(loc='lower right')
    plt.title('Validation Accuracy Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1]) # Accuracy typically between 0 and 1
    plt.grid(True)

    # Plot AUC
    plt.subplot(1, 3, 3)
    if history['val_auc']:
        plt.plot(epochs_range, history['val_auc'], label='Validation AUC-ROC')
    plt.legend(loc='lower right')
    plt.title('Validation AUC-ROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.ylim([0.5, 1]) # AUC typically between 0.5 and 1
    plt.grid(True)

    plt.tight_layout() # Adjust layout

    # Save the combined plot
    plot_path = os.path.join(output_dir, "training_validation_curves.png")
    try:
        plt.savefig(plot_path)
        print(f"Training curves saved to {plot_path}")
    except Exception as e:
        print(f"Error saving training curves plot: {e}")
    plt.close() # Close the figure to free memory

def main(args, ds_init):

    if not args.enable_linear_eval:
        args.aa = 'rand-m9-mstd0.5-inc1'

    print("--- Arguments ---")
    for arg, value in sorted(vars(args).items()):
         print(f"{arg}: {value}")
    print("-----------------")


    print("Before entering init_distributed_mode function")
    utils.init_distributed_mode(args)
    print("After exiting init_distributed_mode function")

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # import random # Already imported if needed elsewhere
    # random.seed(seed)

    cudnn.benchmark = True

    # mean and std for imagenet
    mean = [0.485, 0.456, 0.406]
    std = [0.228, 0.224, 0.225]

    normalize = transforms.Normalize(mean=mean, std=std)
    # --- Define Transforms ---
    # Simplified train_trans based on previous discussions (adjust if needed)
    train_trans = [
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3), # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]
    if args.aa: # Add RandAugment if specified
        print(f"Using RandAugment: {args.aa}")
        train_trans.insert(1, transforms.RandAugment(num_ops=9, magnitude=9)) # Example RandAugment

    val_trans = [
        transforms.Resize(args.input_size + 32, interpolation=3), # Resize slightly larger
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        normalize
    ]
    test_trans = transforms.Compose(val_trans) # Use validation transform for non-TTA test
    if args.TTA:
        # Define TTA transforms if needed (e.g., FiveCrop + flips)
        # For simplicity, using ToTensor() for now as in the original code snippet provided
        test_trans = transforms.Compose([transforms.ToTensor(), normalize])
        print("Using TTA transforms (currently basic ToTensor+Normalize, adjust if needed)")


    data_transforms = {
        'train': transforms.Compose(train_trans),
        'val': transforms.Compose(val_trans),
        'test': test_trans
    }
    # --- End Define Transforms ---


    if args.nb_classes == 2:
        binary = True
    else:
        binary = False

    df = pd.read_csv(args.csv_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True) # Reset index after shuffle

    # --- Create Datasets ---
    try:
        dataset_train = Uni_Dataset(df=df,
                                    root=args.root_path,
                                    train=True,
                                    transforms=data_transforms['train'],
                                    binary=binary,
                                    data_percent=args.percent_data,
                                    image_key=args.image_key)
        dataset_val = Uni_Dataset(df=df,
                                  root=args.root_path,
                                  val=True,
                                  transforms=data_transforms['val'],
                                  binary=binary,
                                  image_key=args.image_key)
        dataset_test = Uni_Dataset(df=df,
                    root=args.root_path,
                    test=True,
                    transforms=data_transforms['test'],
                    binary=binary,
                    image_key=args.image_key)
        print("Datasets created successfully.")
    except Exception as e:
        print(f"Error creating datasets: {e}")
        print(f"CSV Path: {args.csv_path}, Root Path: {args.root_path}")
        # Potentially print df.head() or check file existence
        exit(1)
    # --- End Create Datasets ---

    global_rank = utils.get_rank()
    num_tasks = utils.get_world_size() # Get num_tasks here

    # --- Define Samplers ---
    sampler_train = None # Initialize
    if args.distributed:
        if args.weights:
            try:
                # Calculate weights for sampler (only needs to be done once)
                label_column = "binary_label" if binary else "label"
                label_counts = dataset_train.count_label(label_column)
                total_samples = sum(label_counts)
                class_weights_sampler = [total_samples / (len(label_counts) * count) if count > 0 else 0 for count in label_counts] # Avoid division by zero
                weight_dict_sampler = dict(zip(label_counts.index, class_weights_sampler))

                print('Label distribution for Sampler:')
                for label, count in label_counts.items():
                    print(f'Label {label}: {count}, Weight: {weight_dict_sampler.get(label, 0):.4f}')

                # Get weights for each sample in the training portion of the *original* dataframe
                train_df = df[df['split'] == 'train'].iloc[:len(dataset_train)] # Ensure correct length if percent_data < 1
                train_y = train_df[label_column].values.tolist()
                sample_weights = torch.tensor([weight_dict_sampler.get(label, 0) for label in train_y]) # Use .get with default

                if len(sample_weights) != len(dataset_train):
                     print(f"Warning: Length mismatch! sample_weights ({len(sample_weights)}) vs dataset_train ({len(dataset_train)}). This might happen with data_percent < 1.0.")
                     # Adjust if necessary, though WeightedRandomSampler might handle it if num_samples is correct

                # Create sampler, ensure num_samples matches dataset size
                sampler_train = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset_train), replacement=True)
                print("Using WeightedRandomSampler for distributed training.")
                # NOTE: WeightedRandomSampler does NOT need DistributedSampler wrapper
            except Exception as e:
                print(f"Error setting up WeightedRandomSampler: {e}")
                print("Falling back to DistributedSampler.")
                sampler_train = torch.utils.data.DistributedSampler(
                    dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
        else:
            # Standard distributed sampler
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print("Using standard DistributedSampler.")

        # Validation and Test Samplers for Distributed Eval
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False) # Typically False for consistent testing
        else:
            # If not distributed eval, use sequential on rank 0, None on others
            sampler_val = torch.utils.data.SequentialSampler(dataset_val) if global_rank == 0 else None
            sampler_test = torch.utils.data.SequentialSampler(dataset_test) if global_rank == 0 else None

    else: # Not distributed training
        if args.weights:
             # Calculate weights and create sampler (similar to distributed case but no DDP involved)
            label_column = "binary_label" if binary else "label"
            label_counts = dataset_train.count_label(label_column)
            total_samples = sum(label_counts)
            class_weights_sampler = [total_samples / (len(label_counts) * count) if count > 0 else 0 for count in label_counts]
            weight_dict_sampler = dict(zip(label_counts.index, class_weights_sampler))
            train_df = df[df['split'] == 'train'].iloc[:len(dataset_train)]
            train_y = train_df[label_column].values.tolist()
            sample_weights = torch.tensor([weight_dict_sampler.get(label, 0) for label in train_y])
            sampler_train = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset_train), replacement=True)
            print("Using WeightedRandomSampler for single GPU training.")
        else:
            # Standard random sampler for single GPU
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            print("Using standard RandomSampler.")

        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    # --- End Define Samplers ---

    print('train size:', len(dataset_train), ',val size:', len(dataset_val), ',test size:', len(dataset_test))


    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    # --- Create DataLoaders ---
    # Need to handle shuffle=False when using WeightedRandomSampler
    shuffle_train = (sampler_train is None) or isinstance(sampler_train, torch.utils.data.DistributedSampler)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        shuffle=shuffle_train # Only shuffle if sampler allows (i.e., not WeightedRandomSampler)
    )

    # Only create val/test loaders if the dataset/sampler exists (relevant for non-rank 0 in non-dist-eval)
    data_loader_val = None
    if dataset_val is not None and sampler_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size), # Use larger batch for eval
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

    data_loader_test = None
    if dataset_test is not None and sampler_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size, # Use train batch size or specific eval batch size
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    # --- End Create DataLoaders ---


    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # --- Create Model ---
    print(f"Creating model: {args.model}")
    if args.model=="PanDerm_Large_FT":
        model = panderm_large_patch16_224_finetune(
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            use_rel_pos_bias=args.rel_pos_bias,
            init_values=args.layer_scale_init_value,
            lin_probe=args.enable_linear_eval)
        patch_size = model.patch_embed.patch_size
    elif args.model=='PanDerm_Base_FT':
        model = panderm_base_patch16_224_finetune(
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            use_rel_pos_bias=args.rel_pos_bias,
            init_values=args.layer_scale_init_value,
            lin_probe=args.enable_linear_eval)
        patch_size = model.patch_embed.patch_size
    else:
        # Fallback or error for unknown models
        raise ValueError(f"Model {args.model} not recognized.")
    # print(model) # Optional: print full model structure
    print(f"Model {args.model} created.")
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size
    # --- End Create Model ---


    # --- Load Pretrained Checkpoint ---
    if args.pretrained_checkpoint:
        print(f"Attempting to load pretrained checkpoint: {args.pretrained_checkpoint}")
        if args.pretrained_checkpoint.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.pretrained_checkpoint, map_location='cpu', check_hash=True)
            checkpoint_model_key = 'model' # Assume default key for URL checkpoints
        else:
            if not os.path.exists(args.pretrained_checkpoint):
                 print(f"Error: Pretrained checkpoint file not found at {args.pretrained_checkpoint}")
                 exit(1)
            checkpoint = load_checkpoint(args.pretrained_checkpoint, map_location='cpu')
            # Find the actual key used for the model's state_dict
            checkpoint_model_key = None
            possible_keys = args.model_key.split('|') + ['state_dict', 'model_state'] # Add more common keys
            for key in possible_keys:
                if key in checkpoint:
                    checkpoint_model_key = key
                    break
            if checkpoint_model_key is None:
                 # If no standard key is found, assume the whole dict is the state_dict
                 checkpoint_model_key = None # Will assign checkpoint directly later

        if checkpoint_model_key:
            print(f"Using key '{checkpoint_model_key}' for model state_dict.")
            checkpoint_model = checkpoint[checkpoint_model_key]
        else:
            print("No standard model key found in checkpoint, attempting to load entire dictionary as state_dict.")
            checkpoint_model = checkpoint

        # --- Key Cleaning and Adaptation ---
        new_state_dict = {}
        # Determine if prefixes exist consistently
        has_encoder_prefix = all(k.startswith('encoder.') for k in checkpoint_model.keys() if '.' in k)
        has_module_prefix = all(k.startswith('module.') for k in checkpoint_model.keys() if '.' in k)

        print(f"Checkpoint keys analysis: Consistent 'encoder.' prefix? {has_encoder_prefix}, Consistent 'module.' prefix? {has_module_prefix}")

        num_skipped = 0
        for k, v in checkpoint_model.items():
            new_k = k
            # Strip prefixes only if they were consistently found
            if has_module_prefix:
                 new_k = new_k.replace('module.', '', 1)
            if has_encoder_prefix:
                 new_k = new_k.replace('encoder.', '', 1)

            # Rename norm layer
            if new_k.startswith('norm.'):
                 new_k = new_k.replace('norm.', 'fc_norm.', 1)

            # Skip irrelevant layers
            if new_k.startswith('decoder.') or new_k.startswith('teacher.') or "relative_position_index" in new_k or new_k.startswith('mask_token'):
                 num_skipped += 1
                 continue

            new_state_dict[new_k] = v
        checkpoint_model = new_state_dict
        print(f"Skipped {num_skipped} keys (decoder/teacher/pos_index/mask_token).")
        # --- End Key Cleaning ---


        # --- Shape Mismatch Handling (Head) ---
        model_state_dict = model.state_dict() # Get current model state dict
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and k in model_state_dict: # Check if key exists in both
                 if checkpoint_model[k].shape != model_state_dict[k].shape:
                     print(f"Removing key {k} from pretrained checkpoint due to shape mismatch: "
                           f"Checkpoint shape {checkpoint_model[k].shape}, Model shape {model_state_dict[k].shape}")
                     del checkpoint_model[k]
            elif k in checkpoint_model and k not in model_state_dict:
                 print(f"Removing key {k} from pretrained checkpoint as it doesn't exist in the current model.")
                 del checkpoint_model[k]
        # --- End Shape Mismatch Handling ---


        # --- Relative Position Bias Handling ---
        if model.use_rel_pos_bias:
             # Check for old shared format first
             shared_bias_key = "rel_pos_bias.relative_position_bias_table"
             if shared_bias_key in checkpoint_model:
                 print("Expanding shared relative position embedding to each transformer block.")
                 num_layers = model.get_num_layers()
                 rel_pos_bias_shared = checkpoint_model.pop(shared_bias_key)
                 for i in range(num_layers):
                      block_key = f"blocks.{i}.attn.relative_position_bias_table"
                      if block_key in model_state_dict: # Only add if the model expects it
                           checkpoint_model[block_key] = rel_pos_bias_shared.clone()
                 print(f"Expanded shared bias to {num_layers} blocks.")

             # Now interpolate per-block tables if necessary
             for k in list(checkpoint_model.keys()):
                  if "relative_position_bias_table" in k:
                      if k not in model_state_dict:
                           print(f"Skipping interpolation for {k}: Key not found in current model.")
                           continue

                      rel_pos_bias = checkpoint_model[k]
                      src_num_pos, num_attn_heads = rel_pos_bias.size()
                      dst_num_pos, _ = model_state_dict[k].size()

                      if src_num_pos == dst_num_pos:
                           continue # No interpolation needed if sizes match

                      dst_patch_shape = model.patch_embed.patch_shape
                      num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)

                      # Ensure src_num_pos is large enough
                      if src_num_pos <= num_extra_tokens:
                          print(f"Warning: Source table size {src_num_pos} too small for extra tokens {num_extra_tokens} in key {k}. Skipping interpolation.")
                          # Remove the key from checkpoint if it can't be interpolated? Or keep as is?
                          # For now, remove to avoid loading errors, but this indicates a potential issue.
                          del checkpoint_model[k]
                          continue

                      src_size_sq = src_num_pos - num_extra_tokens
                      # Ensure src_size_sq is non-negative and a perfect square
                      if src_size_sq < 0 or int(src_size_sq**0.5)**2 != src_size_sq:
                           print(f"Warning: Calculated source size squared ({src_size_sq}) is invalid for key {k}. Skipping interpolation.")
                           del checkpoint_model[k]
                           continue
                      src_size = int(src_size_sq**0.5)

                      dst_size_sq = dst_num_pos - num_extra_tokens
                      if dst_size_sq < 0 or int(dst_size_sq**0.5)**2 != dst_size_sq:
                           print(f"Warning: Calculated destination size squared ({dst_size_sq}) is invalid for key {k}. Skipping interpolation.")
                           # This case is less likely if the model definition is correct, but good to check.
                           del checkpoint_model[k]
                           continue
                      dst_size = int(dst_size_sq**0.5)


                      if src_size == 0 or dst_size == 0:
                           print(f"Warning: Calculated src ({src_size}) or dst ({dst_size}) size is zero for key {k}. Skipping interpolation.")
                           del checkpoint_model[k]
                           continue

                      print(f"Interpolating position embedding for {k} from {src_size}x{src_size} to {dst_size}x{dst_size}")

                      extra_tokens = rel_pos_bias[-num_extra_tokens:, :] if num_extra_tokens > 0 else None
                      rel_pos_bias_core = rel_pos_bias[:-num_extra_tokens, :] if num_extra_tokens > 0 else rel_pos_bias

                      # --- Geometric progression calculation ---
                      left, right = 1.01, 1.5
                      while right - left > 1e-6:
                           q = (left + right) / 2.0
                           gp = lambda a, r, n: a * (1.0 - r ** n) / (1.0 - r) if r != 1.0 else a * n
                           calculated_gp = gp(1, q, src_size // 2) if src_size // 2 > 0 else 0
                           if calculated_gp > dst_size // 2:
                                right = q
                           else:
                                left = q
                      q = (left + right) / 2.0
                      dis = []
                      cur = 1
                      for i in range(src_size // 2):
                           dis.append(cur)
                           cur += q ** (i + 1)
                      r_ids = [-x for x in reversed(dis)]
                      x = r_ids + [0] + dis
                      y = x # Assuming square patches/tables
                      # --- Interpolation ---
                      t = dst_size / 2.0 # Use float division
                      dx = np.arange(-t + 0.5, t - 0.4, 1.0) # Adjust range slightly for better centering?
                      dy = dx
                      # Check if grid sizes are valid for interp2d
                      if len(x) < 2 or len(y) < 2:
                           print(f"Warning: Source grid size too small ({len(x)}x{len(y)}) for interpolation in key {k}. Skipping.")
                           del checkpoint_model[k]
                           continue

                      all_rel_pos_bias = []
                      for head_i in range(num_attn_heads):
                           z = rel_pos_bias_core[:, head_i].view(src_size, src_size).float().numpy()
                           try:
                                f = interpolate.interp2d(x, y, z, kind='cubic')
                                interpolated_bias = torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device)
                                all_rel_pos_bias.append(interpolated_bias)
                           except Exception as interp_e:
                                print(f"Error during interpolation for key {k}, head {head_i}: {interp_e}. Skipping key.")
                                # Remove the problematic key and break inner loop
                                if k in checkpoint_model: del checkpoint_model[k]
                                all_rel_pos_bias = [] # Reset list
                                break # Stop processing heads for this key

                      if not all_rel_pos_bias: # If interpolation failed for any head
                           continue # Skip to next key

                      rel_pos_bias_interpolated = torch.cat(all_rel_pos_bias, dim=-1)
                      # --- Combine with extra tokens ---
                      if extra_tokens is not None:
                           # Ensure shapes match for concatenation
                           if rel_pos_bias_interpolated.shape[0] + extra_tokens.shape[0] == dst_num_pos:
                                new_rel_pos_bias = torch.cat((rel_pos_bias_interpolated, extra_tokens), dim=0)
                           else:
                               print(f"Warning: Shape mismatch after interpolation for {k}. Expected {dst_num_pos}, got {rel_pos_bias_interpolated.shape[0] + extra_tokens.shape[0]}. Skipping key.")
                               del checkpoint_model[k]
                               continue
                      else:
                           if rel_pos_bias_interpolated.shape[0] == dst_num_pos:
                                new_rel_pos_bias = rel_pos_bias_interpolated
                           else:
                               print(f"Warning: Shape mismatch after interpolation for {k} (no extra tokens). Expected {dst_num_pos}, got {rel_pos_bias_interpolated.shape[0]}. Skipping key.")
                               del checkpoint_model[k]
                               continue
                      checkpoint_model[k] = new_rel_pos_bias
        # --- End Relative Position Bias Handling ---

        print("Attempting to load state dict...")
        load_result = utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
        print(load_result) # Print missing/unexpected keys message
    # --- End Load Pretrained Checkpoint ---

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model # Assign before potential DDP wrapping
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print(f'Number of params (M): {n_parameters / 1.e6:.2f}') # Use f-string

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size if len(dataset_train) > 0 else 0


    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)


    # --- Layer Decay Setup ---
    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        print("Using layer decay:", args.layer_decay)
    else:
        assigner = None
        print("Not using layer decay (layer_decay >= 1.0)")


    if assigner is not None:
        print("Assigned layer decay values = %s" % str(assigner.values))
    # --- End Layer Decay Setup ---

    skip_weight_decay_list = model.no_weight_decay()
    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)
    print("Skip weight decay list: ", skip_weight_decay_list)


    # --- Optimizer and Scaler Setup ---
    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )
        print("Using DeepSpeed optimizer.")
        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            print("Wrapping model with DistributedDataParallel.")
            # find_unused_parameters can be True if parts of model aren't used (e.g., lin probe)
            # Set to False if confident all params are used, might be slightly faster
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
            model_without_ddp = model.module # Update model_without_ddp reference

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()
        print(f"Using optimizer: {args.opt}")
    # --- End Optimizer and Scaler Setup ---


    # --- LR Scheduler and WD Scheduler Setup ---
    if num_training_steps_per_epoch <= 0:
         print("Warning: num_training_steps_per_epoch is zero or negative. Skipping scheduler setup. Check dataset/batch size.")
         lr_schedule_values = None
         wd_schedule_values = None
    else:
        print("LR Scheduler = cosine")
        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )
        # Setup weight decay scheduler
        if args.weight_decay_end is None:
            args.weight_decay_end = args.weight_decay # Use constant WD if end not specified
            print("Using constant weight decay:", args.weight_decay)
            wd_schedule_values = None # Indicate constant WD
        else:
            print(f"Using cosine weight decay schedule: {args.weight_decay} -> {args.weight_decay_end}")
            wd_schedule_values = utils.cosine_scheduler(
                args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
            print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))
    # --- End LR Scheduler and WD Scheduler Setup ---


    # --- Loss Function Setup ---
    # Calculate class weights for the loss function (needs to be done only once)
    try:
        label_column_loss = "binary_label" if binary else "label"
        label_counts_loss = dataset_train.count_label(label_column_loss)
        total_samples_loss = sum(label_counts_loss)
        # Ensure counts are not zero before division
        # class_weights_list = [total_samples_loss / (len(label_counts_loss) * count) if count > 0 else 0 for count in label_counts_loss.values] # Use .values
        # Ensure counts are not zero before division
        # --- FIX: Use square root of inverse frequency to soften extreme weights ---
        class_weights_list = [(total_samples_loss / (len(label_counts_loss) * count))**0.5 if count > 0 else 0 for count in label_counts_loss.values] # Use .values
        class_weights = torch.tensor(class_weights_list, device=device, dtype=torch.float) # Ensure float
        print("Calculated Class Weights for Loss:", class_weights)
        if torch.any(class_weights <= 0):
             print("Warning: Some class weights are zero or negative. Check label counts.")
    except Exception as e:
        print(f"Error calculating class weights for loss: {e}. Using uniform weights.")
        class_weights = None # Fallback to no weights

    # Choose criterion based on mixup/smoothing and apply weights
    if mixup_fn is not None:
        # Mixup handles smoothing, use custom weighted loss for soft targets
        print("Using WeightedLabelSmoothingCrossEntropy with MixUp")
        criterion = WeightedLabelSmoothingCrossEntropy(smoothing=args.smoothing, class_weights=class_weights)
    elif args.smoothing > 0.:
        # Use custom weighted loss with label smoothing
        print("Using WeightedLabelSmoothingCrossEntropy")
        criterion = WeightedLabelSmoothingCrossEntropy(smoothing=args.smoothing, class_weights=class_weights)
    else:
        # Standard CrossEntropyLoss already accepts weights
        print("Using standard nn.CrossEntropyLoss with weights")
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights) # Pass tensor directly

    print("criterion = %s" % str(criterion))
    # --- End Loss Function Setup ---


    # --- Load Checkpoint ---
    # Load checkpoint AFTER defining optimizer, criterion, model EMA etc.
    # Ensures scheduler state, etc., are loaded if present in checkpoint
    print("Attempting to load model/optimizer/scaler state...")
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
    # --- End Load Checkpoint ---


    # --- Evaluation Mode ---
    if args.eval:
        epoch=args.start_epoch # Use start epoch from loaded checkpoint if available
        print(f"--- Starting Evaluation Only Mode (Epoch {epoch}) ---")
        if data_loader_test is None:
             print("Error: Test data loader not available for evaluation. Check sampler setup.")
             exit(1)

        # Ensure model is in eval mode
        model.eval()

        if args.TTA:
            print(f"Starting evaluation with TTA")
            # Assuming evaluate_tta handles distributed eval internally if needed
            test_res, _ = evaluate_tta(data_loader_test, model, device, args.output_dir, epoch, mode='test',
                                                num_class=args.nb_classes)
        else:
            print(f"Starting evaluation without TTA")
            # Assuming evaluate handles distributed eval internally if needed
            test_res, wandb_res = evaluate(data_loader_test, model, device,args.output_dir, epoch, mode='test',
                                        num_class=args.nb_classes)
            print('Test Results:', test_res)
            # Log results if needed
            if utils.is_main_process():
                 print("Logging evaluation results to WandB...")
                 wandb.log({f"eval_{k}": v for k,v in wandb_res.items()})

        exit(0) # Exit after evaluation
    # --- End Evaluation Mode ---


    # --- Training Loop ---
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    # --- EARLY STOPPING INITIALIZATION ---
    best_val_score = None
    epochs_no_improve = 0
    # Determine if higher score is better based on the monitor metric
    higher_is_better = args.monitor not in ['loss', 'Val Loss'] # Check against 'loss' and potential wandb key
    print(f"Early Stopping Monitor: {args.monitor}, Higher is better: {higher_is_better}, Patience: {args.early_stopping_patience}, Delta: {args.early_stopping_delta}")
    # --- END EARLY STOPPING INITIALIZATION ---

    # --- HISTORY LISTS INITIALIZATION ---
    train_loss_hist = []
    val_loss_hist = []
    val_acc_hist = []
    val_bacc_hist = []
    val_auc_hist = []
    # --- END HISTORY LISTS INITIALIZATION ---

    final_epoch = args.start_epoch # Initialize final_epoch to start_epoch


    for epoch in range(args.start_epoch, args.epochs):
        final_epoch = epoch # Update last completed epoch number
        if args.distributed:
             # Need to set epoch for DistributedSampler, but not WeightedRandomSampler
             if isinstance(data_loader_train.sampler, torch.utils.data.DistributedSampler):
                 print(f"Setting epoch {epoch} for DistributedSampler.")
                 data_loader_train.sampler.set_epoch(epoch)
             # else: print("Sampler is not DistributedSampler, not setting epoch.") # Optional debug print

        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        print(f"\n--- Starting Epoch {epoch} ---")
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            args=args
        )

        # --- Validation Step ---
        # Only run validation if data loader exists (handles non-rank 0 in non-dist-eval)
        # And respect disable_eval_during_finetuning flag
        current_val_score = None # Initialize
        wandb_res = {} # Initialize empty dict for logging
        if data_loader_val is not None and not args.disable_eval_during_finetuning:
            print(f"--- Starting Validation Epoch {epoch} ---")
            val_stats, wandb_res = evaluate(data_loader_val, model, device, args.output_dir, epoch, mode='val',
                                            num_class=args.nb_classes)
            print('-------------------------- Validation Results:', wandb_res)

            # --- APPEND METRICS TO HISTORY LISTS ---
            try:
                # Use .get() for safety in case a key is missing from train_stats or wandb_res
                train_loss_hist.append(train_stats.get('loss', float('nan')))
                val_loss_hist.append(wandb_res.get('Val Loss', float('nan')))
                val_acc_hist.append(wandb_res.get('Val Acc', float('nan')))
                val_bacc_hist.append(wandb_res.get('Val BAcc', float('nan')))
                val_auc_hist.append(wandb_res.get('Val ROC', float('nan')))
            except Exception as hist_e:
                print(f"Warning: Error appending metrics to history lists: {hist_e}")
            # --- END APPEND METRICS ---


            # --- EARLY STOPPING & BEST CHECKPOINT LOGIC ---
            # Map common monitor args to potential wandb keys
            monitor_key_map = {
                'acc': 'Val Acc', 'accuracy': 'Val Acc',
                'bacc': 'Val BAcc', 'balanced_accuracy': 'Val BAcc',
                'recall': 'Val Recall_macro', 'recall_macro': 'Val Recall_macro',
                'auc': 'Val ROC', 'roc': 'Val ROC', 'auc-roc':'Val ROC',
                'loss': 'Val Loss',
                'f1': 'Val W_F1', 'f1_weighted': 'Val W_F1'
            }
            monitor_key = monitor_key_map.get(args.monitor.lower(), args.monitor)
            current_val_score = wandb_res.get(monitor_key)

            if current_val_score is None:
                 print(f"Warning: Monitored metric key '{monitor_key}' (derived from '{args.monitor}') not found in validation results: {wandb_res.keys()}. Skipping improvement check for epoch {epoch}.")
            else:
                print(f"Epoch {epoch} - Monitored Metric ({args.monitor} -> {monitor_key}): {current_val_score:.4f}")
                improved = False
                if best_val_score is None: # First epoch with a valid score
                    best_val_score = current_val_score
                    improved = True
                    print("Initialized best validation score.")
                elif higher_is_better:
                    if current_val_score > best_val_score + args.early_stopping_delta:
                        print(f"Improvement detected: {current_val_score:.4f} > {best_val_score:.4f} + {args.early_stopping_delta}")
                        best_val_score = current_val_score
                        improved = True
                    # else: print(f"No improvement: {current_val_score:.4f} <= {best_val_score:.4f} + {args.early_stopping_delta}") # Redundant print
                else: # Lower is better (e.g., loss)
                    if current_val_score < best_val_score - args.early_stopping_delta:
                        print(f"Improvement detected: {current_val_score:.4f} < {best_val_score:.4f} - {args.early_stopping_delta}")
                        best_val_score = current_val_score
                        improved = True
                    # else: print(f"No improvement: {current_val_score:.4f} >= {best_val_score:.4f} - {args.early_stopping_delta}") # Redundant print

                if improved:
                    print(f"Validation score improved to {best_val_score:.4f}. Saving best model.")
                    epochs_no_improve = 0
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch='best', model_ema=model_ema)
                else:
                    epochs_no_improve += 1
                    if best_val_score is not None:
                         print(f"Validation score did not improve for {epochs_no_improve} epoch(s). Best score: {best_val_score:.4f}")
                    # else: print(f"Validation score did not improve for {epochs_no_improve} epoch(s). (Best score not yet established).") # Redundant

        elif args.disable_eval_during_finetuning:
             print(f"Skipping validation for epoch {epoch} as per --disable_eval_during_finetuning.")
             # If eval is disabled, we might want to save based on epoch number or just save the last one
             if args.output_dir and args.save_ckpt: # Save last checkpoint if eval disabled
                  print("Saving model checkpoint (last epoch since validation is disabled).")
                  utils.save_model(
                         args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                         loss_scaler=loss_scaler, epoch='last_no_eval', model_ema=model_ema)

        # --- SAVE REGULAR CHECKPOINT (even if validation skipped) ---
        if args.output_dir and args.save_ckpt and (epoch + 1) % args.save_ckpt_freq == 0:
             print(f"Saving checkpoint for epoch {epoch}...")
             utils.save_model(
                 args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                 loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)

        # --- LOGGING ---
        # Combine train stats and validation results (wandb_res) for logging
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                      **wandb_res, # Log all validation metrics directly (will be empty if eval skipped)
                      'epoch': epoch,
                      'n_parameters': n_parameters}

        if utils.is_main_process(): # Ensure wandb logs only on main process
             try:
                wandb.log(log_stats)
             except Exception as wandb_e:
                 print(f"Warning: Failed to log metrics to WandB for epoch {epoch}: {wandb_e}")


        # Log to local file
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            try:
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
            except Exception as log_e:
                 print(f"Warning: Failed to write to local log file for epoch {epoch}: {log_e}")


        # --- EARLY STOPPING CHECK ---
        # Only check if validation was performed and score was found
        if current_val_score is not None and args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs due to no improvement in {args.monitor} for {args.early_stopping_patience} consecutive epochs.")
            # final_epoch already updated at start of loop
            break # Exit the training loop
        # --- END EARLY STOPPING CHECK ---

    # --- End Training Loop ---

    # --- FINAL TESTING ---
    # Run final test only if not in eval mode and if a best checkpoint was likely saved (or eval wasn't disabled)
    if not args.eval and data_loader_test is not None:
        print("\n--- Starting Final Testing Phase ---")
        best_ckpt_path = os.path.join(args.output_dir, 'checkpoint-best.pth')
        last_ckpt_path = os.path.join(args.output_dir, f'checkpoint-{final_epoch}.pth') # Use actual final epoch
        last_no_eval_ckpt_path = os.path.join(args.output_dir, 'checkpoint-last_no_eval.pth')

        model_weight_to_load = None
        if args.disable_eval_during_finetuning and os.path.exists(last_no_eval_ckpt_path):
             model_weight_to_load = last_no_eval_ckpt_path
             print(f"Loading last checkpoint '{model_weight_to_load}' for final test (eval was disabled).")
        elif os.path.exists(best_ckpt_path):
            model_weight_to_load = best_ckpt_path
            print(f"Loading best checkpoint '{model_weight_to_load}' for final test.")
        elif os.path.exists(last_ckpt_path):
             model_weight_to_load = last_ckpt_path
             print(f"Warning: Best checkpoint not found. Loading last epoch checkpoint '{model_weight_to_load}' for final test.")
        else:
            print(f"Warning: No best or last checkpoint found in {args.output_dir}. Skipping final test.")

        if model_weight_to_load:
            try:
                # Load checkpoint onto CPU first
                model_dict = load_checkpoint(model_weight_to_load, map_location='cpu')

                # Determine the model to load into
                load_model = model_without_ddp # Load into the base model

                # Get the state dict
                state_dict_to_load = model_dict.get('model', model_dict) # Handle different save formats

                # Check and adjust keys for DDP vs non-DDP mismatch
                is_ddp_checkpoint = all(k.startswith('module.') for k in state_dict_to_load.keys())

                adjusted_state_dict = {}
                if is_ddp_checkpoint:
                     print("Adjusting keys: Removing 'module.' prefix from checkpoint.")
                     adjusted_state_dict = {k.replace('module.', '', 1): v for k, v in state_dict_to_load.items()}
                else:
                     adjusted_state_dict = state_dict_to_load

                # Load the adjusted state dict
                missing_keys, unexpected_keys = load_model.load_state_dict(adjusted_state_dict, strict=False)
                if missing_keys: print("Warning: Missing keys during final test model load:", missing_keys)
                if unexpected_keys: print("Warning: Unexpected keys during final test model load:", unexpected_keys)

                # Ensure the model is back on the correct device for evaluation
                load_model.to(device)

                # Use the potentially DDP-wrapped model if evaluation needs to be distributed
                # For final testing, usually done on a single GPU (rank 0) or all GPUs via dist_eval in evaluate func
                eval_model = model # Pass the main model (potentially DDP wrapped) to evaluate functions

                # --- Perform Final Evaluation ---
                if args.TTA:
                    print(f"Starting final test with TTA using loaded model.")
                    test_stats, _ = evaluate_tta(data_loader_test, eval_model, device, args.output_dir, epoch='best_or_last',
                                                    mode='test',
                                                    num_class=args.nb_classes)
                else:
                    print(f"Starting final test without TTA using loaded model.")
                    test_stats, wandb_test = evaluate(data_loader_test, eval_model, device, args.output_dir, epoch='best_or_last',
                                                      mode='test',
                                                      num_class=args.nb_classes)
                    if utils.is_main_process(): # Log final test metrics
                         final_test_metrics = {f"final_test_{k}": v for k, v in wandb_test.items()}
                         print("Final Test Results:", final_test_metrics)
                         try:
                              wandb.log(final_test_metrics)
                         except Exception as wandb_e:
                              print(f"Warning: Failed to log final test metrics to WandB: {wandb_e}")
            except Exception as load_e:
                 print(f"Error loading model checkpoint {model_weight_to_load} for final testing: {load_e}")

    # --- End Final Testing ---


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('\nTotal Training time {}'.format(total_time_str))

    # --- PLOT CURVES AFTER TRAINING ---
    # Ensure history lists are populated before plotting
    if utils.is_main_process() and args.output_dir and train_loss_hist: # Check if training actually ran
        history = {
            'train_loss': train_loss_hist,
            'val_loss': val_loss_hist,
            'val_acc': val_acc_hist,
            'val_bacc': val_bacc_hist,
            'val_auc': val_auc_hist
        }
        # Use final_epoch + 1 if loop completed, or final_epoch if it broke early
        # Correction: The number of completed epochs is final_epoch + 1
        epochs_completed = final_epoch + 1 
        
        # --- FIX: Check list lengths against epochs_completed ---
        # Ensure all history lists match the *shortest* valid metric list length
        # This handles the case where the error happened mid-epoch or lists are inconsistent
        valid_lengths = [len(h) for h in history.values() if h] # Get lengths of non-empty lists
        if not valid_lengths:
             print("Skipping curve plotting: No metric history was recorded.")
             epochs_completed = 0
        else:
             epochs_completed = min(valid_lengths) # Use the shortest recorded history
             if not all(len(h) == epochs_completed for h in history.values() if h):
                  print(f"Warning: History list lengths are inconsistent. Truncating all plots to {epochs_completed} epochs.")
                  # Truncate lists to the shortest valid length
                  for key in history:
                       if history[key]: # Only truncate if list is not empty
                           history[key] = history[key][:epochs_completed]
        
        if epochs_completed > 0:
             print(f"Plotting curves for {epochs_completed} completed epochs.")
             plot_training_curves(history, args.output_dir, epochs_completed)
        # --- END FIX ---
            
    elif utils.is_main_process():
         print("Skipping curve plotting (no training epochs completed or no output directory).")
    # --- END PLOT CURVES ---


if __name__ == '__main__':
    # This will load the WANDB_API_KEY from your .env file into the environment
    load_dotenv()

    print("Getting args", flush=True)
    opts, ds_init = get_args()
    print("Args received")

    # Initialize wandb only in the main process
    # Check if WANDB_API_KEY is set, otherwise disable W&B
    wandb_enabled = os.environ.get("WANDB_API_KEY") is not None and utils.is_main_process()

    if wandb_enabled:
        try:
            project_name = 'FM_FT_screening' if not opts.eval else 'panderm-finetune-eval'
            wandb.init(
                project=project_name,
                name=opts.wandb_name,
                notes="baselines",
                config=opts)
            print("WandB initialized successfully.")
        except Exception as e:
            print(f"Error initializing WandB: {e}. Disabling WandB logging.")
            wandb_enabled = False # Disable if init fails
            wandb = None # Set wandb to None to avoid errors later
    else:
        print("WandB disabled (API key not found or not main process).")
        wandb = None # Set wandb to None

    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)

    # Finish wandb run only in the main process if it was enabled
    if wandb_enabled and wandb is not None:
        try:
            wandb.finish()
            print("WandB finished successfully.")
        except Exception as e:
            print(f"Error finishing WandB run: {e}")