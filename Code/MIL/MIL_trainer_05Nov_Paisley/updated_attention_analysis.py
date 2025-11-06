"""
Updated attention analysis and visualization utilities, now shows all combinations of most extreme low/high grade and low/high attentions
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from PIL import Image
import torch
from collections import defaultdict

from config import IMAGE_CONFIG


# # NEW FUNCTION: Collect patch data for categorization
# def collect_patch_data(case_id: str, case_grade: str, stain_slices: Dict, 
#                       attention_weights: Dict) -> List[Dict]:
#     """
#     Collect patch data with attention weights and grades for categorization
#     """
#     patches_data = []
    
#     if 'stain_weights' not in attention_weights:
#         return patches_data
    
#     for stain, weights_dict in attention_weights['stain_weights'].items():
#         patch_weights_list = weights_dict.get('patch_weights', [])
        
#         for slice_idx, patch_weights in enumerate(patch_weights_list):
#             patch_weights_np = patch_weights.cpu().numpy()
#             slice_tensor = stain_slices[stain][slice_idx]
            
#             for patch_idx, patch_weight in enumerate(patch_weights_np):
#                 patches_data.append({
#                     'case_id': case_id,
#                     'stain': stain,
#                     'slice_idx': slice_idx,
#                     'patch_idx': patch_idx,
#                     'attention_weight': patch_weight,
#                     'grade': case_grade,
#                     'patch_tensor': slice_tensor[patch_idx].clone()
#                 })
    
#     return patches_data

# def analyze_attention_weights(model, test_loader, output_dir: str, top_n: int = 5):
#     """
#     Analyze and visualize attention weights from the model,
#     separated by high/benign grade and top/bottom attention.

#     Args:
#         model: Trained MIL model
#         test_loader: Test data loader
#         output_dir: Directory to save visualizations
#         top_n: Number of top/bottom patches to visualize
#     """
#     print("\n" + "=" * 60)
#     print("ATTENTION ANALYSIS (Separated by Grade & Attention Rank)")
#     print("=" * 60)

#     attention_dir = os.path.join(output_dir, "attention_analysis")
#     os.makedirs(attention_dir, exist_ok=True)

#     model.eval()
#     attention_summary = []
#     all_patches_data = []  # collect all patch-level data

#     with torch.no_grad():
#         for batch in test_loader:
#             case_data = batch[0]
#             case_id = case_data["case_id"]
#             stain_slices = case_data["stain_slices"]
#             label = case_data["label"]

#             # Forward pass with attention weights
#             logits, attention_weights = model(stain_slices, return_attn_weights=True)

#             # # Individual case summary (optional)
#             # case_summary = analyze_case_attention(
#             #     case_id, stain_slices, attention_weights, attention_dir, top_n
#             # )
#             # attention_summary.append(case_summary)

#             # Label → grade
#             case_grade = 'high' if label.item() == 1 else 'benign'

#             # Collect per-patch attention data
#             case_patches_data = collect_patch_data(
#                 case_id, case_grade, stain_slices, attention_weights
#             )
#             all_patches_data.extend(case_patches_data)
# 
    # # --- Organize by grade & attention rank ---
    
    # # sort patches by high grade and benign
    # high_patches = [p for p in all_patches_data if p['grade'] == 'high']
    # benign_patches = [p for p in all_patches_data if p['grade'] == 'benign']

    # # visualize high grade and benign 
    # visualize_ranked_attention(high_patches, "high", attention_dir, top_n)
    # visualize_ranked_attention(benign_patches, "benign", attention_dir, top_n)

    # # --- Save overall summary ---
    # save_attention_summary(attention_summary, attention_dir)

    # print(f"Attention analysis saved to: {attention_dir}")


# #------ Helper function -----------
# def visualize_ranked_attention(patches, grade_name: str, attention_dir: str, top_n: int):
#     """
#     Helper: Visualize top and bottom attention patches for a given grade.

#     Args:
#         patches: List of patch data dicts with 'attention' and 'image'
#         grade_name: 'high' or 'benign'
#         attention_dir: Output directory for saving
#         top_n: Number of patches to visualize for each group
#     """
#     # Sort patches by attention (descending)
#     patches_sorted = sorted(patches, key=lambda x: x['attention_weight'], reverse=True)

#     # Split into top and bottom
#     top_patches = patches_sorted[:top_n]
#     bottom_patches = patches_sorted[-top_n:]

#     # Save or visualize results
#     visualize_attention_subset(
#         top_patches,
#         os.path.join(attention_dir, f"{grade_name}_grade_top_attention"),
#         title=f"{grade_name.capitalize()} Grade - Top {top_n} Attention"
#     )

#     visualize_attention_subset(
#         bottom_patches,
#         os.path.join(attention_dir, f"{grade_name}_grade_bottom_attention"),
#         title=f"{grade_name.capitalize()} Grade - Bottom {top_n} Attention"
#     )


# # NEW, JUST ADDED THIS MISSING FUNCTION:
# def visualize_attention_subset(patches, output_path: str, title: str):
#     """
#     Visualize a subset of patches
#     """
#     if not patches:
#         return
        
#     n_patches = len(patches)
#     fig, axes = plt.subplots(1, n_patches, figsize=(3*n_patches, 3))
#     if n_patches == 1:
#         axes = [axes]
    
#     for i, patch_data in enumerate(patches):
#         # Get and process patch image
#         patch_img = patch_data['patch_tensor'].cpu().numpy().transpose(1, 2, 0)
#         mean = np.array(IMAGE_CONFIG['normalize_mean'])
#         std = np.array(IMAGE_CONFIG['normalize_std'])
#         patch_img = patch_img * std + mean
#         patch_img = np.clip(patch_img, 0, 1)
        
#         axes[i].imshow(patch_img)
#         axes[i].set_title(f"Attn: {patch_data['attention_weight']:.4f}", fontsize=9)
#         axes[i].axis('off')
    
#     plt.suptitle(title, fontsize=12)
#     plt.tight_layout()
    
#     # Save
#     plt.savefig(f"{output_path}.png", dpi=150, bbox_inches='tight')
#     plt.close()


# new version

def analyze_attention_weights(model, test_loader, output_dir: str, top_n: int = 5):
    """
    For each case, find and visualize:
    - Highest attention patch from highest attention stain→slice
    - Lowest attention patch from lowest attention stain→slice  
    Separated by grade
    """
    print("\n" + "=" * 60)
    print("ATTENTION ANALYSIS (Hierarchical Extreme Patches by Case & Grade)")
    print("=" * 60)

    attention_dir = os.path.join(output_dir, "attention_analysis")
    os.makedirs(attention_dir, exist_ok=True)

    model.eval()
    
    # Store patches for each category
    high_grade_highest_patches = []
    high_grade_lowest_patches = []
    benign_grade_highest_patches = [] 
    benign_grade_lowest_patches = []

    with torch.no_grad():
        for batch in test_loader:
            case_data = batch[0]
            case_id = case_data["case_id"]
            stain_slices = case_data["stain_slices"]
            label = case_data["label"]

            # Get predictions with attention weights
            logits, attention_weights = model(stain_slices, return_attn_weights=True)
            
            # Get grade
            case_grade = 'high' if label.item() == 1 else 'benign'
            
            # Find extreme patches following hierarchy: stain → slice → patch
            highest_patch = find_highest_attention_patch(case_id, stain_slices, attention_weights)
            lowest_patch = find_lowest_attention_patch(case_id, stain_slices, attention_weights)
            
            if highest_patch:
                if case_grade == 'high':
                    high_grade_highest_patches.append(highest_patch)
                else:
                    benign_grade_highest_patches.append(highest_patch)
            
            if lowest_patch:
                if case_grade == 'high':
                    high_grade_lowest_patches.append(lowest_patch)
                else:
                    benign_grade_lowest_patches.append(lowest_patch)

    # Visualize each category
    visualize_patches_category(high_grade_highest_patches, "high_grade_highest_attention", attention_dir)
    visualize_patches_category(high_grade_lowest_patches, "high_grade_lowest_attention", attention_dir)
    visualize_patches_category(benign_grade_highest_patches, "benign_grade_highest_attention", attention_dir) 
    visualize_patches_category(benign_grade_lowest_patches, "benign_grade_lowest_attention", attention_dir)

    print(f"Attention analysis saved to: {attention_dir}")



# old below

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
        
        # Denormalize using image normalization constants from config
        mean = np.array(IMAGE_CONFIG['normalize_mean'])
        std = np.array(IMAGE_CONFIG['normalize_std'])
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



# newest!!
def find_highest_attention_patch(case_id, stain_slices, attention_weights):
    """Find highest attention patch following stain→slice→patch hierarchy"""
    if 'case_weights' not in attention_weights:
        return None
    
    # 1. Find highest attention stain
    case_weights = attention_weights['case_weights'].cpu().numpy()
    stain_order = attention_weights['stain_order']
    highest_stain_idx = np.argmax(case_weights)
    highest_stain = stain_order[highest_stain_idx]
    
    # 2. Find highest attention slice within that stain
    stain_weights = attention_weights['stain_weights'][highest_stain]
    slice_weights = stain_weights['slice_weights'].cpu().numpy()
    highest_slice_idx = np.argmax(slice_weights)
    
    # 3. Find highest attention patch within that slice
    patch_weights_list = stain_weights['patch_weights']
    patch_weights = patch_weights_list[highest_slice_idx].cpu().numpy()
    highest_patch_idx = np.argmax(patch_weights)
    
    # Get the patch tensor
    slice_tensor = stain_slices[highest_stain][highest_slice_idx]
    
    return {
        'case_id': case_id,
        'stain': highest_stain,
        'slice_idx': highest_slice_idx,
        'patch_idx': highest_patch_idx,
        'attention': patch_weights[highest_patch_idx],
        'patch_tensor': slice_tensor[highest_patch_idx].clone()
    }


def find_lowest_attention_patch(case_id, stain_slices, attention_weights):
    """Find lowest attention patch following stain→slice→patch hierarchy"""
    if 'case_weights' not in attention_weights:
        return None
    
    # 1. Find lowest attention stain
    case_weights = attention_weights['case_weights'].cpu().numpy()
    stain_order = attention_weights['stain_order']
    lowest_stain_idx = np.argmin(case_weights)
    lowest_stain = stain_order[lowest_stain_idx]
    
    # 2. Find lowest attention slice within that stain
    stain_weights = attention_weights['stain_weights'][lowest_stain]
    slice_weights = stain_weights['slice_weights'].cpu().numpy()
    lowest_slice_idx = np.argmin(slice_weights)
    
    # 3. Find lowest attention patch within that slice
    patch_weights_list = stain_weights['patch_weights']
    patch_weights = patch_weights_list[lowest_slice_idx].cpu().numpy()
    lowest_patch_idx = np.argmin(patch_weights)
    
    # Get the patch tensor
    slice_tensor = stain_slices[lowest_stain][lowest_slice_idx]
    
    return {
        'case_id': case_id,
        'stain': lowest_stain,
        'slice_idx': lowest_slice_idx,
        'patch_idx': lowest_patch_idx,
        'attention': patch_weights[lowest_patch_idx],
        'patch_tensor': slice_tensor[lowest_patch_idx].clone()
    }


def visualize_patches_category(patches, category_name: str, output_dir: str):
    """
    Visualize all patches for a given category
    """
    if not patches:
        print(f"No patches found for {category_name}")
        return
    
    n_patches = len(patches)
    n_cols = min(5, n_patches)
    n_rows = (n_patches + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for i, patch_data in enumerate(patches):
        if i >= len(axes):
            break
            
        # Process patch image
        patch_img = patch_data['patch_tensor'].cpu().numpy().transpose(1, 2, 0)
        mean = np.array(IMAGE_CONFIG['normalize_mean'])
        std = np.array(IMAGE_CONFIG['normalize_std'])
        patch_img = patch_img * std + mean
        patch_img = np.clip(patch_img, 0, 1)
        
        axes[i].imshow(patch_img)
        title = (f"Case: {patch_data['case_id']}\n"
                f"Stain: {patch_data['stain']}\n"
                f"Attn: {patch_data['attention']:.4f}")
        axes[i].set_title(title, fontsize=8)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(patches), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"{category_name.replace('_', ' ').title()}\n"
                f"Total: {len(patches)} cases", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save
    filename = f"{category_name}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {category_name}: {len(patches)} patches")