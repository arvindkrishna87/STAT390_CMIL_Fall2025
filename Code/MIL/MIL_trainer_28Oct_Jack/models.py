"""
Model architectures for MIL training
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Any, Optional, Tuple

from config import MODEL_CONFIG


class AttentionPool(nn.Module):
    """
    Attention pooling mechanism for MIL
    Pools patch-level features into single bag-level representation
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Small neural network to compute attention scores for each patch
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor, return_weights: bool = False):
        """
        Args:
            x: (B, M, D) where B=batch size, M=patches per bag, D=embedding dim
            return_weights: whether to return attention weights
        
        Returns:
            weighted_x: (B, D) weighted sum of patch embeddings
            weights: (B, M) attention weights (if return_weights=True)
        """
        weights = self.attention(x)  # (B, M, 1)
        weights = torch.softmax(weights, dim=1)  # Normalize attention scores
        
        weighted_x = (weights * x).sum(dim=1)  # (B, D)
        
        if return_weights:
            return weighted_x, weights.squeeze(-1)  # (B, D), (B, M)
        return weighted_x


class HierarchicalAttnMIL(nn.Module):
    """
    Hierarchical Attention MIL model for multi-stain pathology images
    
    Three levels of attention:
    1. Patch-level: within each stain-slice
    2. Stain-level: across slices within each stain  
    3. Case-level: across different stains
    """
    def __init__(self, base_model=None, num_classes: int = 2, embed_dim: int = 512):
        super().__init__()
        
        if base_model is None:
            base_model = models.densenet121(pretrained=True)
        
        # Shared feature extractor (pretrained CNN)
        self.features = base_model.features
        
        # Adaptive pooling to get richer features than just 1x1
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Patch projector: maps CNN features to patch embeddings
        self.patch_projector = nn.Linear(base_model.classifier.in_features * 4, embed_dim)
        
        # Three levels of attention
        self.patch_attention = AttentionPool(embed_dim, MODEL_CONFIG['attention_hidden_dim'])
        self.stain_attention = AttentionPool(embed_dim, MODEL_CONFIG['attention_hidden_dim'])
        self.case_attention = AttentionPool(embed_dim, MODEL_CONFIG['attention_hidden_dim'])
        
        # Final classifier
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, stain_slices_dict: Dict[str, List[torch.Tensor]], 
                return_attn_weights: bool = False):
        """
        Args:
            stain_slices_dict: {
                "h&e": [slice1_tensor, slice2_tensor, ...],   # each slice: (P, C, H, W)
                "melan": [slice1_tensor, slice2_tensor, ...],
                "sox10": [slice1_tensor, slice2_tensor, ...]
            }
            return_attn_weights: whether to return attention weights for visualization
        
        Returns:
            logits: (num_classes,) classification logits
            all_weights: attention weights at all levels (if return_attn_weights=True)
        """
        stain_embeddings = {}
        stain_attention_weights = {}
        
        # Process each stain type separately
        for stain_name, slice_list in stain_slices_dict.items():
            if not slice_list:  # Skip if no slices for this stain
                continue
            
            slice_embeddings = []
            slice_attention_weights = []
            
            # Process each slice within this stain
            for slice_tensor in slice_list:
                # slice_tensor shape: (P, C, H, W) where P = number of patches
                P, C, H, W = slice_tensor.shape
                
                # Extract features for all patches in this slice
                patch_features = self.features(slice_tensor)  # (P, F, h, w)
                pooled = self.pool(patch_features).view(P, -1)  # (P, 4*F)
                patch_embeddings = self.patch_projector(pooled)  # (P, D)
                
                # Apply patch-level attention to get slice embedding
                if return_attn_weights:
                    slice_emb, patch_weights = self.patch_attention(
                        patch_embeddings.unsqueeze(0), return_weights=True
                    )
                    slice_attention_weights.append(patch_weights.squeeze(0))
                else:
                    slice_emb = self.patch_attention(patch_embeddings.unsqueeze(0))
                
                slice_embeddings.append(slice_emb.squeeze(0))  # (D,)
            
            # Stack slice embeddings for this stain
            if slice_embeddings:
                stain_slice_embeddings = torch.stack(slice_embeddings)  # (num_slices, D)
                
                # Apply stain-level attention across slices
                if return_attn_weights:
                    stain_emb, stain_weights = self.stain_attention(
                        stain_slice_embeddings.unsqueeze(0), return_weights=True
                    )
                    stain_attention_weights[stain_name] = {
                        'slice_weights': stain_weights.squeeze(0),
                        'patch_weights': slice_attention_weights
                    }
                else:
                    stain_emb = self.stain_attention(stain_slice_embeddings.unsqueeze(0))
                
                stain_embeddings[stain_name] = stain_emb.squeeze(0)  # (D,)
        
        # If no stains have data, return zero logits
        if not stain_embeddings:
            logits = torch.zeros(self.classifier.out_features).to(next(self.parameters()).device)
            if return_attn_weights:
                return logits, {}
            return logits
        
        # Stack stain embeddings for case-level attention (fusion point)
        stain_emb_list = list(stain_embeddings.values())
        case_stain_embeddings = torch.stack(stain_emb_list)  # (num_stains, D)
        
        # Apply case-level attention across stains
        if return_attn_weights:
            case_emb, case_weights = self.case_attention(
                case_stain_embeddings.unsqueeze(0), return_weights=True
            )
            # Package all attention weights for visualization
            all_weights = {
                'case_weights': case_weights.squeeze(0),
                'stain_weights': stain_attention_weights,
                'stain_order': list(stain_embeddings.keys())
            }
        else:
            case_emb = self.case_attention(case_stain_embeddings.unsqueeze(0))
        
        # Final classification
        logits = self.classifier(case_emb.squeeze(0))  # (num_classes,)
        
        if return_attn_weights:
            return logits, all_weights
        
        return logits


def create_model(num_classes: int = None, embed_dim: int = None, pretrained: bool = True) -> HierarchicalAttnMIL:
    """
    Factory function to create the MIL model
    """
    if num_classes is None:
        num_classes = MODEL_CONFIG['num_classes']
    if embed_dim is None:
        embed_dim = MODEL_CONFIG['embed_dim']
    
    # Create base model
    base_model = models.densenet121(pretrained=pretrained)
    
    # Create and return MIL model
    model = HierarchicalAttnMIL(
        base_model=base_model,
        num_classes=num_classes,
        embed_dim=embed_dim
    )
    
    return model