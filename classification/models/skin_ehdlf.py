import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive Attention-based Feature Fusion Mechanism as described in SkinEHDLF.
    It projects features to a common dimension, computes attention weights, 
    and performs a weighted combination.
    """
    def __init__(self, dim_1, dim_2, dim_3, common_dim=512):
        super().__init__()
        
        # Project all features to a common dimension
        self.proj_1 = nn.Linear(dim_1, common_dim)
        self.proj_2 = nn.Linear(dim_2, common_dim)
        self.proj_3 = nn.Linear(dim_3, common_dim)
        
        # Attention mechanism
        # A simple MLP to learn weights for each branch from the concatenated features
        self.attn_fc = nn.Sequential(
            nn.Linear(common_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 3), # Output 3 weights
            nn.Softmax(dim=1)
        )
        
        self.layer_norm = nn.LayerNorm(common_dim)
    
    # --- Accept 3 inputs, Fuse them, Return 1 output ---
    def forward(self, x1, x2, x3):
        # 1. Projection
        f1 = self.proj_1(x1)
        f2 = self.proj_2(x2)
        f3 = self.proj_3(x3)
        
        # 2. Stack
        concat_features = torch.cat([f1, f2, f3], dim=1)
        
        # 3. Attention
        weights = self.attn_fc(concat_features)
        
        # 4. Fusion
        w1 = weights[:, 0].unsqueeze(1)
        w2 = weights[:, 1].unsqueeze(1)
        w3 = weights[:, 2].unsqueeze(1)
        
        fused = (w1 * f1) + (w2 * f2) + (w3 * f3)
        
        return self.layer_norm(fused)

class SkinEHDLF(nn.Module):
    """
    Hybrid model combining ConvNeXt, EfficientNetV2, and Swin Transformer
    with Adaptive Feature Fusion.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        # 1. Define Backbones (Feature Extractors) using timm
        # We set num_classes=0 to get the feature vector (pooling is usually included by default in timm forward_features + pooling, 
        # or we can grab the representation before the head)
        
        # 1. ConvNeXt Base (Facebook 22k pretrain -> 1k finetune)
        self.convnext = timm.create_model(
            'convnext_base.fb_in22k_ft_in1k', 
            pretrained=pretrained, 
            num_classes=0
        )
        
        # 2. EfficientNetV2-M (ImageNet-21k pretrain -> 1k finetune)
        self.efficientnet = timm.create_model(
            'tf_efficientnetv2_m.in21k_ft_in1k', 
            pretrained=pretrained, 
            num_classes=0
        )
        
        # 3. Swin Base (Microsoft 22k pretrain -> 1k finetune)
        self.swin = timm.create_model(
            'swin_base_patch4_window7_224.ms_in22k_ft_in1k', 
            pretrained=pretrained, 
            num_classes=0
        )
        
        # Determine feature dimensions dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dim1 = self.convnext(dummy_input).shape[1]
            dim2 = self.efficientnet(dummy_input).shape[1]
            dim3 = self.swin(dummy_input).shape[1]
            
        print(f"Feature Dims - ConvNeXt: {dim1}, EfficientNet: {dim2}, Swin: {dim3}")

        # 2. Fusion Layer
        self.fusion_dim = 1024 # Size of the fused vector
        self.fusion_layer = AdaptiveFeatureFusion(dim1, dim2, dim3, common_dim=self.fusion_dim)
        
        # 3. Classification Head (Deep Dense Stack)
        # MODIFIED: Implemented the 6-layer Dense Head described in Table 7 of the SkinEHDLF paper.
        # This consists of 5 hidden layers (1024 units) and 1 output layer.
        
        layers = []
        input_dim = self.fusion_dim # Starts at 1024
        
        # Create 5 hidden layers
        for i in range(5):
            layers.append(nn.Linear(input_dim, 1024))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3)) # Dropout rate from paper table 7
            input_dim = 1024 # Next layer input is 1024
            
        # Final Output Layer (6th layer)
        layers.append(nn.Linear(input_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)

        # --- FIX: Add this attribute to prevent the crash in main() ---
        self.use_rel_pos_bias = False

    # --- FIX: Add this method for the optimizer (weight decay) ---
    def no_weight_decay(self):
        # This model relies on backbones that handle their own decay, 
        # or we can return an empty set to apply decay everywhere by default.
        return {}

    # --- FIX: Add this method for the scheduler (layer decay) ---
    def get_num_layers(self):
        # The training script uses this for layer-wise learning rate decay.
        # Since this is a hybrid model, we return 1 and will disable 
        # layer decay in the arguments.
        return 1
    
    def forward_features(self, x, **kwargs):
        # We accept 'is_train' via kwargs to satisfy the caller, but we don't use it. The model's internal mode (self.training) handles the behavior.
        # This works because 'self.convnext' exists in this class!
        f_convnext = self.convnext(x)
        f_efficient = self.efficientnet(x)
        f_swin = self.swin(x)
        
        # Pass the 3 features to the fusion layer
        return self.fusion_layer(f_convnext, f_efficient, f_swin)

    def forward(self, x):
        # This standard pipeline pattern now works perfectly
        x = self.forward_features(x)
        x = self.classifier(x)
        return x

# Factory function for builder
def skin_ehdlf_hybrid(num_classes=2, **kwargs):
    return SkinEHDLF(num_classes=num_classes, **kwargs)