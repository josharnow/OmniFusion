"""Model factory and preprocessing utilities for PanDerm evaluation.

This module wires the encoders used in the PanDerm paper (s41591-025-03747-y.pdf)
and provides the evaluation transforms. It supports:
- PanDerm Large/Base encoders (factory functions in modeling_finetune)
- Baseline backbones (e.g., dinov2, SwAVDerm)

Only comments/docstrings added—no functional changes.
"""

import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from torch import nn
from torchvision import transforms
from models.modeling_finetune import *
import open_clip

from models.modeling_finetune import VisionTransformer
from utils.utils import cae_kwargs
from functools import partial

from .skin_ehdlf import skin_ehdlf_hybrid

def get_norm_constants(which_img_norm: str = 'imagenet'):
    """Return mean/std tuples used for normalization in eval transforms.

    Options include ImageNet stats and OpenAI CLIP stats as commonly used in ViT setups.
    """
    print('normalization method: ',which_img_norm)
    constants_zoo = {
        'imagenet': {'mean': (0.485, 0.456, 0.406), 'std': (0.228, 0.224, 0.225)},
        'openai_clip':{'mean': (0.48145466, 0.4578275, 0.40821073), 'std': (0.26862954, 0.26130258, 0.27577711)},
        'uniform': {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)}
    }

    constants = constants_zoo[which_img_norm]
    return constants.get('mean'), constants.get('std')

def get_eval_transforms(
        which_img_norm: str = 'imagenet',
        img_resize: int = 224,
        center_crop: bool = False
):
    r"""
    Build the image transformation used for PanDerm feature extraction.

    Args:
        - which_img_norm (str): key selecting normalization stats

    Return:
        - eval_transform (torchvision.Transform): composed torchvision transform.
    """
    mean, std = get_norm_constants(which_img_norm)
    eval_trans = [transforms.Resize(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=mean, std=std)]
    eval_transform = transforms.Compose(eval_trans)
    return eval_transform


def get_encoder(args, model_name,which_img_norm='imagenet'):
    """Create the requested encoder and its eval transform, loading checkpoints as needed.

    Returns: (model, eval_transform)
    """
    # which_img_norm='imagenet'
    print('loading model checkpoint')

    if model_name == 'imgnet_large21k':
        model = timm.create_model(args.pretrained_checkpoint,
                                  num_classes=0,
                                  dynamic_img_size=True,
                                  pretrained=True)
    elif model_name == 'SwAVDerm':
        model = TimmCNNEncoder(kwargs = {'features_only': True, 'out_indices': (4,), 'pretrained': True, 'num_classes': 0})
        checkpoint = torch.load(args.pretrained_checkpoint, map_location='cpu')
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('module', 'model'): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    elif model_name == 'dinov2':
        model = timm.create_model("vit_large_patch14_dinov2.lvd142m",
                                  num_classes=0,
                                  dynamic_img_size=True,
                                  pretrained=True)
    elif model_name == 'PanDerm_Large_LP':
        model = panderm_large_patch16_224()
        state_dict = torch.load(args.pretrained_checkpoint, map_location='cpu',weights_only=True)
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    elif model_name == 'PanDerm_Base_LP':
        model = panderm_base_patch16_224()

        model.load_state_dict(torch.load(args.pretrained_checkpoint, map_location='cpu'), strict=False) 
        model.head = torch.nn.Identity()
    elif model_name == 'SkinEHDLF_Hybrid':
        # The hybrid model includes its own classifier, so we might need to handle args.num_classes if your pipeline expects the encoder to exclude the head.
        # However, since this is a custom hybrid, it's often easier to return the full model.
        # If your pipeline strictly separates encoder/head, you might need to adjust logic in run_class_finetuning.py
        
        print(f"Initializing SkinEHDLF Hybrid Model with {args.nb_classes} classes...")
        
        # Load the model directly. 
        # Note: We pass num_classes here because SkinEHDLF includes its own classifier head.
        model = skin_ehdlf_hybrid(num_classes=args.nb_classes)

        # SkinEHDLF is a hybrid, not a pure ViT, so it has no single 'patch_size'.
        # We set these to None or defaults to avoid crashes later if args are accessed.
        args.patch_size = None
        args.window_size = None
        
        # Note: SkinEHDLF is composed of pretrained parts, so we typically don't load a single "SkinEHDLF" checkpoint unless you have saved one from Phase 3 training.
        if hasattr(args, 'pretrained_checkpoint') and args.pretrained_checkpoint:
             print(f"Loading custom hybrid checkpoint from {args.pretrained_checkpoint}")
             checkpoint = torch.load(args.pretrained_checkpoint, map_location='cpu')
             model.load_state_dict(checkpoint, strict=False)
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    
    # Print model summary for reproducibility/debugging.
    print(model)
    # constants = MODEL2CONSTANTS[model_name]
    eval_transform = get_eval_transforms(
        which_img_norm=which_img_norm,
        img_resize=256,
        center_crop=True
    )
    # img_transforms = get_eval_transforms(mean=constants['mean'],
    #                                      std=constants['std'],
    #                                      target_img_size = target_img_size)

    return model, eval_transform