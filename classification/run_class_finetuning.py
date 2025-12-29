# Add these imports at the top of the file
import os
from dotenv import load_dotenv

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
# import torch
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
from timm.models import create_model
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
from sklearn.metrics import precision_recall_curve # <--- +++ ADDED THIS IMPORT

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
    parser.add_argument('--disable_amp', action='store_true', default=False, help='Disable automatic mixed precision training (train in float32).')

    parser.add_argument('--exp_name', default='', type=str,
                        help='name of exp. it is helpful when save the checkpoint')
    
    parser.add_argument('--tune_threshold', action='store_true',
                        help='Conditionally tune threshold for evaluation')
    
    parser.add_argument('--sq_root_loss', action='store_true',
                        help='Applies square root to the loss function to stabilize training.')
    
    parser.add_argument('--focal_loss', action='store_true',
                        help='Use Focal Loss instead of Cross Entropy Loss.')
    

    parser.add_argument('--no_class_weights', action='store_true',
                        help='Disable class alpha weighting.')
    
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone weights for linear probe.')


    parser.add_argument('--eval_threshold', default=0.5, type=float, help='Threshold for binary classification evaluation.')

    parser.add_argument('--focal_loss_gamma', default=2.0, type=float, help='Gamma parameter for Focal Loss.')

    parser.add_argument('--weight_cap', default=float('inf'), type=float, help='Cap for class alpha weights to avoid extreme values.')

    parser.add_argument('--custom_majority_alpha', default=None, type=float, help='Custom alpha weight for majority class in Focal Loss.')

    parser.add_argument('--custom_minority_alpha', default=None, type=float, help='Custom alpha weight for minority class in Focal Loss.')

    # For thesis stage 2
    parser.add_argument('--stratify_source', action='store_true',
                        help='Enable stratified sampling by Source (ISIC vs External) AND Label')


    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # Optional: You can pass your class_weights here
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate standard Cross Entropy (element-wise)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss) # Get the probability of the true class
        
        # Apply the Focal term: (1 - pt)^gamma
        # If pt is high (easy), (1-pt) is near 0, so loss is crushed.
        # If pt is low (hard), (1-pt) is near 1, so loss is preserved.
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def main(args, ds_init):

    if not args.enable_linear_eval:
        args.aa = 'rand-m9-mstd0.5-inc1'

    print(args)

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
    # random.seed(seed)

    cudnn.benchmark = True

    # mean and std for imagenet
    mean = [0.485, 0.456, 0.406]
    std = [0.228, 0.224, 0.225]

    normalize = transforms.Normalize(mean=mean, std=std)
    train_trans = [
        transforms.Resize(256),
        transforms.RandomResizedCrop(args.input_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(hue=0.2),
        transforms.ToTensor(),
        normalize]

    val_trans = [transforms.Resize(256),
                 transforms.CenterCrop(args.input_size),
                 transforms.ToTensor(),
                 normalize]
    
    # --- ADD THIS ---
    # Define TTA test transforms: Resize, Crop, ToTensor, but NO normalization
    # The TTAHandler will apply its own normalization.
    test_trans_tta = [transforms.Resize(256),
                      transforms.CenterCrop(args.input_size),
                      transforms.ToTensor()]
    # --- END ADD ---

    data_transforms = {
        'train': transforms.Compose(train_trans),
        'val': transforms.Compose(val_trans),
        # 'test': transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]) if args.TTA else transforms.Compose(val_trans) # uncomment it when you are testing on F17K or Daffodil
        # 'test': transforms.Compose([transforms.ToTensor()]) if args.TTA else transforms.Compose(val_trans) # For HAM TTA
        'test': transforms.Compose(test_trans_tta) if args.TTA else transforms.Compose(val_trans) # For HAM TTA
    }

    if args.nb_classes == 2:
        binary = True
    else:
        binary = False

    df = pd.read_csv(args.csv_path)
    df = df.sample(frac=1, random_state=42)
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

    # dataset_test = dataset_val 
    dataset_test = Uni_Dataset(df=df,
                root=args.root_path,
                test=True,
                transforms=data_transforms['test'],
                binary=binary,
                image_key=args.image_key)

    global_rank = utils.get_rank()
    if args.weights:
        if args.stratify_source:
            print(">>> Using Source-Stratified Sampling (4 Groups: ISIC/Ext * Benign/Mal) <<<")
            
            # 1. Get the training dataframe
            # We assume dataset_train corresponds to df[df['split'] == 'train']
            train_df = df[df['split'] == 'train'].copy()
            
            # 2. Identify Source (ISIC vs External)
            # Logic: ISIC images start with 'ISIC_2024', everything else is External
            train_df['is_isic'] = train_df['image'].astype(str).str.startswith('ISIC_2024')
            
            # 3. Identify Label
            label_col = "binary_label" if binary else "label"
            
            # 4. Create 4 Distinct Groups
            # Group 0: External Benign
            # Group 1: External Malignant
            # Group 2: ISIC Benign
            # Group 3: ISIC Malignant
            train_df['group'] = 0 # Default (External Benign)
            train_df.loc[ (~train_df['is_isic']) & (train_df[label_col] == 1), 'group'] = 1
            train_df.loc[ (train_df['is_isic']) & (train_df[label_col] == 0), 'group'] = 2
            train_df.loc[ (train_df['is_isic']) & (train_df[label_col] == 1), 'group'] = 3
            
            # 5. Calculate Weights so each GROUP has equal probability
            # (i.e., each of the 4 groups contributes 25% of the batch)
            group_counts = train_df['group'].value_counts().sort_index()
            print("Stratified Group Counts:", group_counts.to_dict())
            
            # Weight = 1 / count
            group_weights = 1.0 / group_counts
            
            # Map group weights back to each sample
            sample_weights = torch.tensor(train_df['group'].map(group_weights).values, dtype=torch.double)
            
            sampler_train = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset_train), replacement=True)
        else:
            label_counts = dataset_train.count_label("binary_label" if binary else "label")
            total_samples = sum(label_counts)
            weights = [total_samples / (len(label_counts) * count) for count in label_counts]
            weight_dict = dict(zip(label_counts.index, weights))

            print('Label distribution:')
            for label, count in label_counts.items():
                print(f'Label {label}: {count}')

            train_y = df[(df['split'] == 'train')]["binary_label" if binary else "label"].values.tolist()
            sample_weights = torch.tensor([weight_dict[label] for label in train_y])
            sampler_train = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset_train), replacement=True)
        print("Using WeightedRandomSampler")
    else:
        num_tasks = utils.get_world_size()

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train =", str(sampler_train))

    print('train size:', len(dataset_train), ',val size:', len(dataset_val), ',test size:', len(dataset_test))

    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
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
        print(model)
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
        print(model)
        patch_size = model.patch_embed.patch_size
    else:
        raise ValueError(f"Invalid model name '{args.model}', not supported.")
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size


    # if args.pretrained_checkpoint:
    #     if args.pretrained_checkpoint.startswith('https'):
    #         checkpoint = torch.hub.load_state_dict_from_url(
    #             args.pretrained_checkpoint, map_location='cpu', check_hash=True)
    #     else:
    #         checkpoint = torch.load(args.pretrained_checkpoint, map_location='cpu')
    #
    #         if args.pretrained_checkpoint.split('/')[-1].startswith('open_clip'):
    #             state_dict = checkpoint['state_dict']
    #             state_dict = {k:v for k,v in state_dict.items() if 'visual' in k}
    #             checkpoint = {}
    #             checkpoint['model'] = {k.replace('module.visual.', 'encoder.'): v for k,v in state_dict.items()}
    #
    #     print("Load ckpt from %s" % args.pretrained_checkpoint)
    if args.pretrained_checkpoint:
        if args.pretrained_checkpoint.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.pretrained_checkpoint, map_location='cpu', check_hash=True)
        else:
            checkpoint_model = torch.load(args.pretrained_checkpoint, map_location='cpu', weights_only=False) # Weights_only changed to fix unpickling error
            checkpoint = {'model': checkpoint_model}
            if args.pretrained_checkpoint.split('/')[-1].startswith('open_clip'):
                state_dict = checkpoint['state_dict']
                state_dict = {k: v for k, v in state_dict.items() if 'visual' in k}
                checkpoint = {}
                checkpoint['model'] = {k.replace('module.visual.', 'encoder.'): v for k, v in state_dict.items()}
        print('===========================')
        print("CHECKPOINT MODEL:")
        print(checkpoint_model.keys())
        print(checkpoint_model)
        print('===========================')
        print("Load ckpt from %s" % args.pretrained_checkpoint)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        all_keys = list(checkpoint_model.keys())
        # print("##########origin keys:", len(all_keys), all_keys)
        # NOTE: remove all decoder keys
        all_keys = [key for key in all_keys if key.startswith('encoder.')]
        print("all keys:", all_keys)
        for key in all_keys:
            new_key = key.replace('encoder.','')
            checkpoint_model[new_key] = checkpoint_model[key]
            checkpoint_model.pop(key)

        for key in list(checkpoint_model.keys()):
            if key.startswith('decoder.'):
                checkpoint_model.pop(key)
            if key.startswith('teacher.'):
                checkpoint_model.pop(key)

        # NOTE: replace norm with fc_norm
        for key in list(checkpoint_model.keys()):
            if key.startswith('norm.'):
                new_key = key.replace('norm.','fc_norm.')
                checkpoint_model[new_key] = checkpoint_model[key]
                checkpoint_model.pop(key)

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        if model.use_rel_pos_bias and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
            print("Expand the shared relative position embedding to each transformer block. ")
            num_layers = model.get_num_layers()
            rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
            for i in range(num_layers):
                checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()

            checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")

        all_keys = list(checkpoint_model.keys())

        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

            if "relative_position_bias_table" in key and args.rel_pos_bias:
                rel_pos_bias = checkpoint_model[key]
                src_num_pos, num_attn_heads = rel_pos_bias.size()
                dst_num_pos, _ = model.state_dict()[key].size()
                dst_patch_shape = model.patch_embed.patch_shape
                if dst_patch_shape[0] != dst_patch_shape[1]:
                    raise NotImplementedError()
                num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
                src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
                dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
                if src_size != dst_size:
                    print("Position interpolate for %s from %dx%d to %dx%d" % (
                        key, src_size, src_size, dst_size, dst_size))
                    extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                    rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    print("Original positions = %s" % str(x))
                    print("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(num_attn_heads):
                        z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                        f = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(
                            torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                    rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                    new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                    checkpoint_model[key] = new_rel_pos_bias

        print("##############new keys:", len(checkpoint_model), checkpoint_model.keys())

        # TODO interpolate position embedding if need

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    # --- FIX: EXPLICITLY FREEZE BACKBONE FOR LINEAR PROBE ---
    if args.freeze_backbone:
        print("Info: Enabling Linear Eval. Freezing backbone weights...")
        for name, param in model.named_parameters():
            if 'head' not in name: # Keep the classifier head unfrozen
                param.requires_grad = False
            else:
                print(f"Keeping {name} unfrozen.")
    # --------------------------------------------------------

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module
        
        # --- MODIFICATION: Optimizer and Loss Scaler are only created if NOT in --eval mode ---
        if not args.eval:
            optimizer = create_optimizer(
                args, model_without_ddp, skip_list=skip_weight_decay_list,
                get_num_layer=assigner.get_layer_id if assigner is not None else None,
                get_layer_scale=assigner.get_scale if assigner is not None else None)
            loss_scaler = NativeScaler()
        else:
            optimizer = None
            loss_scaler = None
    
    # --- This block only runs if we are in TRAINING mode ---
    if not args.eval:
        print("Use step level LR scheduler!")

        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )
        if args.weight_decay_end is None:
            args.weight_decay_end = args.weight_decay
            wd_schedule_values = utils.cosine_scheduler(
                args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
            print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

        # MODIFICATION: Calculate class weights and apply them to the loss function
        label_counts = dataset_train.count_label("binary_label" if binary else "label")
        total_samples = sum(label_counts)

        # Applies a weight cap to avoid extremely high weights, which could destabilize training
        weight_cap = args.weight_cap # This can be adjusted based on experimentation (e.g., 100.0, 200.0, etc.)

        class_weights_list = [min((total_samples / (len(label_counts) * count)), weight_cap) for count in label_counts] if not args.sq_root_loss else [min((total_samples / (len(label_counts) * count))**0.5, weight_cap) for count in label_counts]
        
        # --- FIX: Use square root of inverse frequency to soften extreme weights ---
        # class_weights_list = [(total_samples / (len(label_counts) * count))**0.5 for count in label_counts]

        class_weights = torch.tensor(class_weights_list if (not args.custom_majority_alpha or not args.custom_minority_alpha) else [args.custom_majority_alpha, args.custom_minority_alpha], device=device)
        # print("Using Class Weights for Loss:", class_weights)

        if mixup_fn is not None:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        elif args.focal_loss:
            if not args.no_class_weights:
                print("Using Focal Loss with Class Weights:", class_weights)
            criterion = FocalLoss(gamma=args.focal_loss_gamma, alpha=class_weights if not args.no_class_weights else None, reduction='mean')
        else:
            # Apply the calculated class weights here
            if not args.no_class_weights:
                print("Using Class Weights for Loss:", class_weights)
            # NOTE - This might be the best way to address imbalanced datasets
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights if not args.no_class_weights else None)


        print("criterion = %s" % str(criterion))

        utils.auto_load_model(
            args=args, model=model, model_without_ddp=model_without_ddp,
            optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    # --- +++ CORRECTED AND UNIFIED args.eval BLOCK +++ ---
    elif args.eval:
        epoch=0
        model_weight = args.resume
        model_dict = torch.load(model_weight, map_location=device, weights_only=False)
        model.load_state_dict(model_dict['model'])
        
        best_threshold = args.eval_threshold  # Default threshold if not tuning
        if args.tune_threshold:
            print("\n--- EVALUATION-ONLY MODE WITH THRESHOLD TUNING ---")
            
            # --- (UNIFIED) STEP 1: Get probabilities from VAL set to find threshold ---
            print("Running on validation set to find best threshold (using standard eval)...")
            # We *always* use the standard (non-TTA) validation loader to find the threshold.
            # The 'evaluate' function now returns metrics, wandb_res, prediction_array, true_label_decode_array
            _, _, val_probs, val_labels = evaluate(
                data_loader_val, model, device, args.output_dir, epoch, 
                mode='val_tune', num_class=args.nb_classes, decision_threshold=None
            )

            # --- (UNIFIED) STEP 2: Calculate best threshold (Targeting a Specific Recall) ---
            TARGET_RECALL = 0.90  # <<< SET YOUR GOAL: e.g., 90% or 95% sensitivity
            print(f"Finding threshold for at least {TARGET_RECALL*100}% recall...")
            
            val_probs_positive = val_probs[:, 1]
            precisions, recalls, thresholds = precision_recall_curve(val_labels, val_probs_positive)

            try:
                # Find the last index where recall is >= your target
                # (The arrays are sorted by recall, descending, so [0] is highest recall)
                # We find the *last* index ([0][-1]) which has the highest precision 
                # *while still meeting* our recall target.
                target_recall_index = np.where(recalls >= TARGET_RECALL)[0][-1]
                best_threshold = thresholds[target_recall_index]
                
                print(f"Found threshold for >= {TARGET_RECALL*100}% recall: {best_threshold:.4f}")
                print(f"  > Precision at this threshold: {precisions[target_recall_index]:.4f}")
                print(f"  > Recall at this threshold: {recalls[target_recall_index]:.4f}")

            except IndexError:
                # Fallback if target recall is never achieved
                print(f"Could not achieve {TARGET_RECALL*100}% recall. Defaulting to max F1.")
                f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
                best_threshold = thresholds[np.argmax(f1_scores)]

            print(f"Using threshold: {best_threshold:.4f}")
            
            # --- +++ ADDED THIS MEMORY CLEANUP +++ ---
            # --- This is the key fix for the args.eval path ---
            print("Clearing validation probability arrays...")
            try:
                del val_probs
                del val_labels
                del precisions
                del recalls
                del thresholds
            except Exception as e:
                print(f"Could not delete val arrays: {e}")
                
            torch.cuda.empty_cache() # Clear cache before final test run
            # --- +++ END MEMORY CLEANUP +++ ---

        # --- (UNIFIED) STEP 3: Run on TEST set using the tuned threshold ---
        if args.TTA:
            print(f"Starting TEST evaluation with TTA (using threshold={best_threshold:.4f})")
            test_stats, _, _, _ = evaluate_tta(
                data_loader_test, model, device, args.output_dir, epoch,
                mode='test', num_class=args.nb_classes,
                decision_threshold=best_threshold # <-- Pass tuned threshold
            )
        else:
            print(f"Starting TEST evaluation without TTA (using threshold={best_threshold:.4f})")
            test_stats, wandb_test, _, _ = evaluate(
                data_loader_test, model, device, args.output_dir, epoch, 
                mode='test', num_class=args.nb_classes,
                decision_threshold=best_threshold # <-- Pass tuned threshold
            )
            print("Logging final test metrics (with tuned threshold) to wandb...")
            wandb.log(wandb_test)
        
        print("Final Tuned Test Metrics:", test_stats)
        exit(0)
    # --- +++ END CORRECTED BLOCK +++ ---
    
    else:
        # This else should not be reachable if logic is correct
        print("Error: --eval flag was not processed correctly.")
        exit(1)

    # --- This block only runs if we are in TRAINING mode ---
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_auc = 0.0
    max_performance = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        # The .set_epoch() method is specific to the DistributedSampler and is needed
        # to ensure proper shuffling in a multi-GPU environment. The WeightedRandomSampler
        # does not have this method. This check prevents an AttributeError.
        if args.distributed and isinstance(data_loader_train.sampler, torch.utils.data.DistributedSampler):
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            args=args
        )
        # Note: evaluate() now returns: metrics, wandb_res, prediction_array, true_label_decode_array
        val_stats, wandb_res, _, _ = evaluate(
            data_loader_val, model, device, args.output_dir, epoch, mode='val',
            num_class=args.nb_classes, decision_threshold=None # Use default threshold for epoch-to-epoch saving
        )
        print('--------------------------',wandb_res)
        val_bacc, val_acc, val_auc_roc, valwf1, val_mean_recall = wandb_res['Val BAcc'], wandb_res['Val Acc'] , wandb_res['Val ROC'], \
            wandb_res['Val W_F1'], wandb_res['Val Recall_macro']
        wandb.log(wandb_res)
        if args.nb_classes == 2:
            if max_performance < val_auc_roc:
                max_performance = val_auc_roc
                if args.output_dir:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch='best', model_ema=model_ema)
            print(f'Max AUCROC: {max_performance:.2f}%')
        elif args.monitor == 'acc':

            if max_performance < val_acc:
                max_performance = val_acc
                if args.output_dir:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch='best', model_ema=model_ema)
            print(f'Max val mean accuracy: {max_performance:.2f}%') 

        elif args.monitor == 'recall':
            if max_performance < val_mean_recall:
                max_performance = val_mean_recall
                if args.output_dir:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch='best', model_ema=model_ema)
            print(f'Max val mean recall: {max_performance:.2f}%')

        # --- +++ ADD THIS BLOCK +++ ---
        # Always save the latest epoch as 'checkpoint-last.pth'
        if args.output_dir:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch='last', model_ema=model_ema)
        # --- +++ END BLOCK +++ ---

        # --- +++ MODIFIED FINAL EVALUATION BLOCK (WITH MEMORY CLEANUP) +++ ---
        if epoch == (args.epochs - 1):
            print("\n--- FINAL EVALUATION ---")
            
            # --- STEP 1: Free all unnecessary training memory ---
            # The optimizer and loss_scaler are the biggest memory hogs
            print("Clearing training tensors (optimizer, loss_scaler) from GPU memory...")
            try:
                del optimizer
                del loss_scaler
            except Exception as e:
                print(f"Could not delete optimizer/loss_scaler: {e}")

            # Clear the CUDA cache to free up fragmented memory
            torch.cuda.empty_cache()

            print("Loading best checkpoint (checkpoint-best.pth)...")
            model_weight = args.output_dir + '/' + 'checkpoint-best.pth'
            model_dict = torch.load(model_weight, map_location=device, weights_only=False)
            model.load_state_dict(model_dict['model'])

            best_threshold = 0.5  # Default threshold if not tuning
            if args.tune_threshold:
                # --- STEP 2: Get probabilities from VAL set to find threshold ---
                print("Running on validation set to find best threshold...")
                _, _, val_probs, val_labels = evaluate(
                    data_loader_val, model, device, args.output_dir, epoch, 
                    mode='val_tune', num_class=args.nb_classes, decision_threshold=None
                )

                # --- STEP 3: Calculate best threshold (Targeting a Specific Recall) ---
                TARGET_RECALL = 0.90  # <<< SET YOUR GOAL: e.g., 90% or 95% sensitivity
                print(f"Finding threshold for at least {TARGET_RECALL*100}% recall...")
                
                val_probs_positive = val_probs[:, 1]
                precisions, recalls, thresholds = precision_recall_curve(val_labels, val_probs_positive)

                try:
                    # Find the last index where recall is >= your target
                    target_recall_index = np.where(recalls >= TARGET_RECALL)[0][-1]
                    best_threshold = thresholds[target_recall_index]
                    
                    print(f"Found threshold for >= {TARGET_RECALL*100}% recall: {best_threshold:.4f}")
                    print(f"  > Precision at this threshold: {precisions[target_recall_index]:.4f}")
                    print(f"  > Recall at this threshold: {recalls[target_recall_index]:.4f}")

                except IndexError:
                    # Fallback if target recall is never achieved
                    print(f"Could not achieve {TARGET_RECALL*100}% recall. Defaulting to max F1.")
                    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
                    best_threshold = thresholds[np.argmax(f1_scores)]

                print(f"Using threshold: {best_threshold:.4f}")

                # --- STEP 4: Free the validation arrays from memory ---
                print("Clearing validation probability arrays...")
                try:
                    del val_probs
                    del val_labels
                    del precisions
                    del recalls
                    del thresholds
                except Exception as e:
                    print(f"Could not delete val arrays: {e}")
                
                torch.cuda.empty_cache() # Clear cache again before final test run

            # --- STEP 5: Run on TEST set using the tuned threshold ---
            if args.TTA:
                print(f"Starting test with TTA (using threshold={best_threshold:.4f})")
                test_stats, _, _, _ = evaluate_tta(
                    data_loader_test, model, device, args.output_dir, epoch,
                    mode='test', num_class=args.nb_classes,
                    decision_threshold=best_threshold # <-- Pass tuned threshold
                )
            else:
                print(f"Starting test without TTA (using threshold={best_threshold:.4f})")
                test_stats, wandb_test, _, _ = evaluate(
                    data_loader_test, model, device, args.output_dir, epoch, 
                    mode='test', num_class=args.nb_classes,
                    decision_threshold=best_threshold # <-- Pass tuned threshold
                )
                print("Logging final test metrics (with tuned threshold) to wandb...")
                wandb.log(wandb_test)
            
            print("Final Tuned Test Metrics:", test_stats)
        # --- +++ END MODIFIED BLOCK +++ ---

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    # This will load the WANDB_API_KEY from your .env file into the environment
    load_dotenv()

    print("Getting args", flush=True)
    opts, ds_init = get_args()
    print("Args received")

    project_name = 'FM_FT_screening' if not opts.eval else 'panderm-finetune'
    wandb.init(
        project=project_name,
        name=opts.wandb_name,
        notes="baselines", \
        config=opts)
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
    wandb.finish()