"""
Attention analysis and visualization utilities
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from PIL import Image
import torch


def analyze_attention_weights(model, test_loader, output_dir: str, top_n: int = 5):
    """
    Analyze and visualize attention weights from the model
    
    Args:
        model: Trained MIL model
        test_loader: Test data loader
        output_dir: Directory to save visualizations
        top_n: Number of top/bottom patches to visualize
    """
    print("\n" + "=" * 60)
    print("ATTENTION ANALYSIS")
    print("=" * 60)
    
    attention_dir = os.path.join(output_dir, "attention_analysis")
    os.makedirs(attention_dir, exist_ok=True)
    
    model.eval()
    attention_summary = []
    
    with torch.no_grad():
        for batch in test_loader:
            case_data = batch[0]
            case_id = case_data["case_id"]
            stain_slices = case_data["stain_slices"]
            
            # Get predictions with attention weights
            logits, attention_weights = model(stain_slices, return_attn_weights=True)
            
            # Analyze this case
            case_summary = analyze_case_attention(
                case_id, stain_slices, attention_weights, 
                attention_dir, top_n
            )
            attention_summary.append(case_summary)
    
    # Save overall summary
    save_attention_summary(attention_summary, attention_dir)
    
    print(f"Attention analysis saved to: {attention_dir}")


def analyze_case_attention(case_id: Any, stain_slices: Dict, attention_weights: Dict,
                           output_dir: str, top_n: int = 5) -> Dict:
    """
    Analyze attention for a single case
    """
    # Create patch_attention subdirectory
    patch_attention_dir = os.path.join(output_dir, "patch_attention")
    os.makedirs(patch_attention_dir, exist_ok=True)
    
    case_summary = {
        'case_id': case_id,
        'stain_attention': {},
        'most_attended_stain': None,
        'stain_order': attention_weights.get('stain_order', [])
    }
    
    # Case-level attention (across stains)
    if 'case_weights' in attention_weights:
        case_weights = attention_weights['case_weights'].cpu().numpy()
        stain_order = attention_weights['stain_order']
        
        # Find most attended stain
        max_idx = np.argmax(case_weights)
        case_summary['most_attended_stain'] = stain_order[max_idx]
        case_summary['stain_attention'] = {
            stain: float(weight) 
            for stain, weight in zip(stain_order, case_weights)
        }
    
    # Stain-level and patch-level attention
    if 'stain_weights' in attention_weights:
        for stain, weights_dict in attention_weights['stain_weights'].items():
            slice_weights = weights_dict.get('slice_weights', [])
            patch_weights_list = weights_dict.get('patch_weights', [])
            
            if len(slice_weights) > 0:
                slice_weights_np = slice_weights.cpu().numpy()
                
                # Find most and least attended slices
                most_attended_slice_idx = np.argmax(slice_weights_np)
                least_attended_slice_idx = np.argmin(slice_weights_np)
                
                # Visualize patches for most attended slice
                if len(patch_weights_list) > most_attended_slice_idx:
                    patch_weights = patch_weights_list[most_attended_slice_idx].cpu().numpy()
                    slice_tensor = stain_slices[stain][most_attended_slice_idx]
                    
                    visualize_patch_attention(
                        case_id, stain, most_attended_slice_idx,
                        slice_tensor, patch_weights, patch_attention_dir, top_n, 
                        prefix="top"
                    )
                
                # Visualize patches for least attended slice
                if len(patch_weights_list) > least_attended_slice_idx:
                    patch_weights = patch_weights_list[least_attended_slice_idx].cpu().numpy()
                    slice_tensor = stain_slices[stain][least_attended_slice_idx]
                    
                    visualize_patch_attention(
                        case_id, stain, least_attended_slice_idx,
                        slice_tensor, patch_weights, patch_attention_dir, top_n,
                        prefix="bottom"
                    )
    
    return case_summary


def visualize_patch_attention(case_id: Any, stain: str, slice_idx: int,
                              slice_tensor: torch.Tensor, patch_weights: np.ndarray,
                              output_dir: str, top_n: int = 5, prefix: str = "top"):
    """
    Visualize top/bottom attended patches for a slice
    """
    num_patches = len(patch_weights)
    
    # Get top/bottom N patches
    if prefix == "top":
        indices = np.argsort(patch_weights)[-top_n:][::-1]  # Descending
        title_prefix = "Most"
    else:
        indices = np.argsort(patch_weights)[:top_n]  # Ascending
        title_prefix = "Least"
    
    # Limit to available patches
    indices = indices[:min(top_n, num_patches)]
    
    if len(indices) == 0:
        return
    
    # Create figure
    n_cols = min(5, len(indices))
    n_rows = (len(indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
            
        # Get patch image (C, H, W) -> (H, W, C)
        patch_img = slice_tensor[idx].cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        patch_img = patch_img * std + mean
        patch_img = np.clip(patch_img, 0, 1)
        
        # Plot
        axes[i].imshow(patch_img)
        axes[i].set_title(f"Patch {idx}\nWeight: {patch_weights[idx]:.4f}", fontsize=10)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Case {case_id} - {stain} - Slice {slice_idx}\n{title_prefix} Attended Patches", 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save
    filename = f"case_{case_id}_{stain}_slice{slice_idx}_{prefix}_patches.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def save_attention_summary(attention_summary: List[Dict], output_dir: str):
    """
    Save text summary of attention analysis
    """
    summary_path = os.path.join(output_dir, "attention_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("ATTENTION ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        for case_info in attention_summary:
            case_id = case_info['case_id']
            f.write(f"Case {case_id}:\n")
            f.write("-" * 40 + "\n")
            
            # Most attended stain
            if case_info['most_attended_stain']:
                f.write(f"Most attended stain: {case_info['most_attended_stain']}\n")
            
            # Stain-level attention weights
            if case_info['stain_attention']:
                f.write("\nStain-level attention:\n")
                for stain, weight in case_info['stain_attention'].items():
                    f.write(f"  {stain}: {weight:.4f}\n")
            
            f.write("\n")
    
    print(f"Attention summary saved to: {summary_path}")


def plot_attention_distribution(attention_summary: List[Dict], output_dir: str):
    """
    Plot distribution of attention across stains
    """
    import matplotlib.pyplot as plt
    from collections import defaultdict
    
    # Aggregate stain attention across all cases
    stain_attention_agg = defaultdict(list)
    
    for case_info in attention_summary:
        for stain, weight in case_info.get('stain_attention', {}).items():
            stain_attention_agg[stain].append(weight)
    
    if not stain_attention_agg:
        return
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stains = list(stain_attention_agg.keys())
    positions = range(len(stains))
    
    # Box plot
    data = [stain_attention_agg[stain] for stain in stains]
    bp = ax.boxplot(data, positions=positions, labels=stains, patch_artist=True)
    
    # Styling
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.set_ylabel('Attention Weight')
    ax.set_xlabel('Stain Type')
    ax.set_title('Distribution of Stain-Level Attention Weights')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    filepath = os.path.join(output_dir, "stain_attention_distribution.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Attention distribution plot saved to: {filepath}")