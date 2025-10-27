# evaluate_only.py
# This script is designed to run ONLY the evaluation/testing phase from run_class_finetuning.py
# It loads a pre-trained model and evaluates it on the test dataset.

import argparse
import os
import sys
import json
import torch
import torch.backends.cudnn as cudnn
import numpy as np

# --- Imports from your project ---
from furnace import utils
from furnace.datasets import build_dataset
from furnace.engine_for_finetuning import evaluate
from timm.models import create_model

def get_args():
    """
    Parses command-line arguments.
    Copied from run_class_finetuning.py and added --checkpoint_path
    """
    parser = argparse.ArgumentParser('PanDerm evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # --- NEW: Path to your trained model ---
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='Absolute path to the pre-trained model checkpoint (.pth file) to evaluate.')

    # Model parameters
    parser.add_argument('--model', default='PanDerm_Large_FT', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--rel_pos_bias', action='store_true',
                        help='Use relative position bias')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias',
                        help='Disable relative position bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--sin_pos_emb', action='store_true',
                        help='Use sine-cosine positional embedding')
    parser.add_argument('--disable_sin_pos_emb', action='store_false', dest='sin_pos_emb',
                        help='Disable sine-cosine positional embedding')
    parser.set_defaults(sin_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--ood_eval', action='store_true', default=False,
                        help='Use OOD evaluation set')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.2, metavar='PCT',
                        help='Drop path rate (default: 0.2)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--weights', action='store_true', default=False,
                        help='Use class weights')

    # Model EMA
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

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
        weight decay. We use a cosine schedule for WD. (Set the same value as WD to disable)""")

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--TTA', action='store_true', default=False,
                        help='Use Test Time Augmentation')
    parser.add_argument('--percent_data', type=float, default=1.0,
                        help='Percent of data to use (default: 1.0)')
    parser.add_argument('--monitor', type=str, default='recall',
                        help='Metric to monitor for saving best model')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--pretrain_model_key', default='model', type=str)
    parser.add_argument('--nb_classes', default=7, type=int,
                        help='number of the classification types')
    parser.add_argument('--unscale_lr', action='store_true')
    parser.set_defaults(unscale_lr=True)

    # Dataset parameters
    parser.add_argument('--data_path', default='/path/to/data/', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--data_set', default='Derm',
                        help='Dataset type')
    parser.add_argument('--test_set', default='test', type=str,
                        help='Test set folder name')
    parser.add_argument('--train_set', default='train', type=str,
                        help='Train set folder name')
    parser.add_argument('--validation_set', default='validation', type=str,
                        help='Validation set folder name')
    
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

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # AMP parameters
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Use PyTorch AMP (Automatic Mixed Precision)')

    return parser.parse_args()


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # --- Set random seeds ---
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # --- Build Test Dataset ---
    dataset_test = build_dataset(is_train=False, args=args, test_set=True)
    
    print(f"Built test dataset with {len(dataset_test)} images.")

    if True:  # args.dist_eval:
        if len(dataset_test) % args.world_size != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will pad the dataset and may cause slightly different results.')
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=args.world_size, rank=utils.get_rank(), shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # --- Build Model ---
    model = create_model(
        args.model,
        pretrained=False, # We load weights manually
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path,
        layer_scale_init_value=args.layer_scale_init_value,
        rel_pos_bias=args.rel_pos_bias,
        sin_pos_emb=args.sin_pos_emb,
    )
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # --- Load Checkpoint ---
    if not args.checkpoint_path:
        print("Error: --checkpoint_path is required for evaluation.")
        sys.exit(1)
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        sys.exit(1)
        
    print(f"Loading model from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')

    if 'model' in checkpoint:
        model_state_dict = checkpoint['model']
    else:
        print("Error: 'model' key not found in checkpoint.")
        sys.exit(1)
        
    # Handle potential DDP prefix 'module.'
    if all(k.startswith('module.') for k in model_state_dict.keys()):
        print("Stripping 'module.' prefix from checkpoint keys.")
        model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}

    msg = model_without_ddp.load_state_dict(model_state_dict, strict=False)
    print("Model loaded with message:", msg)
    
    # --- Setup Logger ---
    logger = utils.get_logger(
        logging_file=os.path.join(args.output_dir, f'eval_log_{utils.get_rank()}.txt')
    )
    
    # --- Run Evaluation (FIXED CALL) ---
    print("Starting evaluation phase...")
    test_stats = evaluate(
        data_loader=data_loader_test,
        model=model,
        device=device,
        # args=args,
        # amp=args.use_amp,
        # model_ema=None,
        # logger=logger,
        num_class=args.nb_classes,  # <-- RENAMED from n_classes
        out_dir=args.output_dir,    # <-- ADDED
        epoch=0,                    # <-- ADDED (0 for eval-only)
        mode='Test'                 # <-- ADDED
    )

    print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")
    print(f"Balanced Accuracy (BAcc): {test_stats['BAcc']:.4f}")
    
    # Save results
    if args.output_dir and utils.is_main_process():
        json_path = os.path.join(args.output_dir, "test_stats.json")
        print(f"Saving test stats to {json_path}")
        with open(json_path, 'w') as f:
            json.dump(test_stats, f, indent=4)

    print("Evaluation complete.")


if __name__ == '__main__':
    main(get_args())