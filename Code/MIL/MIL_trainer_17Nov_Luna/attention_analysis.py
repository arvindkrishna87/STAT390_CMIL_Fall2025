"""
Attention analysis and visualization utilities
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from PIL import Image
import torch
from collections import defaultdict
import csv

from config import IMAGE_CONFIG


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

    plots_dir = os.path.join(attention_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    model.eval()
    attention_summary = []
    # to store patch-level effective attention 
    all_patch_records = []
    case_label_info = {}
    
    with torch.no_grad():
        for batch in test_loader:
            case_data = batch[0]
            case_id = case_data["case_id"]
            stain_slices = case_data["stain_slices"]
            
            # Get predictions with attention weights
            logits, attention_weights = model(stain_slices, return_attn_weights=True)

            # Track per-case labels/predictions for downstream summaries
            true_label = None
            label_tensor = case_data.get("label")

            if isinstance(label_tensor, torch.Tensor):
                true_label = int(label_tensor.detach().view(-1)[0].item())
            else:
                true_label = int(label_tensor)

            pred_label = None

            logits_for_pred = logits.detach()
            if logits_for_pred.dim() == 1:
                logits_for_pred = logits_for_pred.unsqueeze(0)
            elif logits_for_pred.dim() == 0:
                logits_for_pred = logits_for_pred.view(1, 1)
            pred_tensor = torch.argmax(logits_for_pred, dim=1)
            pred_label = int(pred_tensor.view(-1)[0].item())

            case_label_info[case_id] = {
                "true_label": true_label,
                "pred_label": pred_label
            }
            
            # Analyze this case
            case_summary = analyze_case_attention(
                case_id, stain_slices, attention_weights, 
                attention_dir, top_n
            )
            attention_summary.append(case_summary)

            patch_records = compute_effective_patch_attention(case_id, attention_weights)
            all_patch_records.extend(patch_records)

            visualize_case_effective_patches(
                case_id=case_id,
                stain_slices=stain_slices,
                patch_records=patch_records,
                output_dir=attention_dir,
                top_n=top_n
            )
    
    # Save overall summary
    save_attention_summary(attention_summary, attention_dir)

    # plot stain-level attention distribution on all cases 
    plot_attention_distribution(attention_summary, attention_dir)
    
    # effective patch attention distributions
    plot_effective_patch_attention_distribution_per_case(
            all_patch_records,
            case_label_info,
            plots_dir,
            bins = 50
        )

    # Per-case top n% patch analysis
    analyze_top_effective_patches_per_case(
            all_patch_records,
            case_label_info,
            attention_dir,
            top_percent=5.0
        )
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


def visualize_case_effective_patches(case_id: Any,
                                     stain_slices: Dict[str, List[torch.Tensor]],
                                     patch_records: List[Dict[str, Any]],
                                     output_dir: str,
                                     top_n: int = 5):
    """
    Visualize the top/bottom patches across the entire case using effective attention
    weights computed by `compute_effective_patch_attention`.
    """
    sorted_records = sorted(
        patch_records,
        key=lambda r: r.get("effective_weight", 0.0)
    )

    n_select = min(top_n, len(sorted_records))
    top_entries = list(reversed(sorted_records[-n_select:]))
    bottom_entries = sorted_records[:n_select]

    case_effective_dir = os.path.join(output_dir, "case_effective_patches")
    os.makedirs(case_effective_dir, exist_ok=True)

    def _plot_entries(entries: List[Dict[str, Any]], title_prefix: str, filename: str):
        n = len(entries)
        n_cols = min(5, n)
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for ax in axes:
            ax.axis('off')

        for ax, entry in zip(axes, entries):
            stain = entry["stain"]
            slice_idx = entry["slice_idx"]
            patch_idx = entry["patch_idx"]

            stain_slice_list = stain_slices.get(stain, [])
            if slice_idx >= len(stain_slice_list):
                continue
            slice_tensor = stain_slice_list[slice_idx]
            if patch_idx >= slice_tensor.shape[0]:
                continue

            patch_tensor = slice_tensor[patch_idx].cpu().numpy().transpose(1, 2, 0)

            mean = np.array(IMAGE_CONFIG['normalize_mean'])
            std = np.array(IMAGE_CONFIG['normalize_std'])
            patch_img = patch_tensor * std + mean
            patch_img = np.clip(patch_img, 0, 1)

            ax.imshow(patch_img)
            eff_weight = entry.get("effective_weight", 0.0)
            ax.set_title(
                f"{stain} s{slice_idx} p{patch_idx}\nEff: {eff_weight:.4f}",
                fontsize=9
            )
            ax.axis('off')

        plt.suptitle(f"Case {case_id} - {title_prefix} Effective Patches", fontsize=12, fontweight='bold')
        plt.tight_layout()
        filepath = os.path.join(case_effective_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

    _plot_entries(
        top_entries,
        title_prefix="Top",
        filename=f"case_{case_id}_top_effective_patches.png"
    )
    _plot_entries(
        bottom_entries,
        title_prefix="Bottom",
        filename=f"case_{case_id}_bottom_effective_patches.png"
    )


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

def compute_effective_patch_attention(case_id: Any,
                                      attention_weights: Dict) -> List[Dict[str, Any]]:
    """
    Compute effective attention for every patch in a single case

    Returns:
        records: list of dicts with keys:
            - 'case_id'
            - 'stain'
            - 'slice_idx'
            - 'patch_idx'
            - 'patch_attn_weight'
            - 'slice_attn_weight'
            - 'stain_attn_weight'
            - 'effective_weight'
    """
    records: List[Dict[str, Any]] = []

    # Need both case- and stain-level info
    if 'case_weights' not in attention_weights or 'stain_weights' not in attention_weights:
        return records

    case_weights = attention_weights['case_weights'].cpu()   # tensor [num_stains]
    stain_order = attention_weights['stain_order']           # list[str]
    stain_weights_dict = attention_weights['stain_weights']

    # Loop over stains in the same order as case_weights
    for stain_idx, stain in enumerate(stain_order):
        if stain not in stain_weights_dict:
            continue

        weights_dict = stain_weights_dict[stain]
        stain_case_w = case_weights[stain_idx].cpu()         # scalar tensor
        stain_attn_value = float(stain_case_w.item())        # scalar float

        slice_weights = weights_dict.get('slice_weights', None)
        patch_weights_list = weights_dict.get('patch_weights', [])

        if slice_weights is None or len(patch_weights_list) == 0:
            continue

        slice_weights = slice_weights.cpu()                  # [num_slices]

        # For each slice in this stain
        for slice_idx, patch_w in enumerate(patch_weights_list):
            if patch_w is None:
                continue

            patch_w = patch_w.cpu()                          # [num_patches_in_slice]
            slice_attn_value = float(slice_weights[slice_idx].item())

            # One record per patch
            for patch_idx, pw in enumerate(patch_w):
                patch_attn_value = float(pw.item())
                effective_value = stain_attn_value * slice_attn_value * patch_attn_value

                records.append({
                    "case_id": case_id,
                    "stain": stain,
                    "slice_idx": int(slice_idx),
                    "patch_idx": int(patch_idx),
                    "patch_attn_weight": patch_attn_value,
                    "slice_attn_weight": slice_attn_value,
                    "stain_attn_weight": stain_attn_value,
                    "effective_weight": effective_value,
                })

    return records


def plot_effective_patch_attention_distribution_per_case(
    patch_records: List[Dict[str, Any]],
    case_label_info: Dict[Any, Dict[str, int]],
    output_dir: str,
    bins: int = 50,
):
    """
    For each case, plot the distribution (histogram) of effective patch
    attention weights for that case.

    Saves one PNG per case
    """
    if not patch_records:
        print("No effective patch attention data available for per-case plotting.")
        return

    # Group effective weights by case
    case_to_weights: Dict[Any, List[float]] = defaultdict(list)
    for rec in patch_records:
        cid = rec["case_id"]
        case_to_weights[cid].append(rec["effective_weight"])

    for cid, weights_list in case_to_weights.items():
        if not weights_list:
            continue

        weights = np.array(weights_list, dtype=np.float32)
        labels = case_label_info.get(cid, {})
        true_label = labels.get("true_label", None)
        pred_label = labels.get("pred_label", None)

        counts, bin_edges = np.histogram(weights, bins=bins)
        total_patches = counts.sum() if counts.sum() > 0 else 1
        normalized_counts = counts / total_patches
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = np.diff(bin_edges)

        plt.figure(figsize=(8, 5))
        plt.bar(bin_centers, normalized_counts, width=bin_widths, alpha=0.75, align="center")

        # annotate each bar with the raw count of patches contributing to it
        for center, height, count in zip(bin_centers, normalized_counts, counts):
            if count == 0:
                continue
            plt.text(center, height + 0.01, str(count), ha="center", va="bottom", fontsize=8)

        title = f"Case {cid} - Effective Patch Attention"
        if true_label is not None or pred_label is not None:
            title += f"\nTrue: {true_label}, Pred: {pred_label}"

        plt.title(title, fontsize=13)
        plt.xlabel("Effective Patch Attention Weight", fontsize=11)
        plt.ylabel("Normalized Patch Density", fontsize=11)
        plt.grid(axis='y', alpha=0.3)

        fname = f"effective_patch_attn_distro_case_{cid}.png"
        filepath = os.path.join(output_dir, fname)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

    print("Per-case effective patch attention distributions saved.")


def analyze_top_effective_patches_per_case(
    patch_records: List[Dict[str, Any]],
    case_label_info: Dict[Any, Dict[str, int]],
    output_dir: str,
    top_percent: float = 5.0,
):
    """
    For EACH CASE separately, select the top `top_percent` patches by effective_weight
    and summarize where they come from.

    For each case, we get:
      - which patches (case_id, stain, slice_idx, patch_idx),
      - how many top patches are from each stain,
      - how many from each (stain, slice_idx),
      - plus true and predicted labels.

    Args:
        patch_records: list of dicts from compute_effective_patch_attention().
        case_label_info: dict mapping case_id -> {"true_label": int, "pred_label": int}
        output_dir: directory to save CSV and summary.
        top_percent: percentage (0-100) of patches to keep per case.
    """

    # Group records by case
    case_to_records: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for rec in patch_records:
        cid = rec["case_id"]
        case_to_records[cid].append(rec)

    # Collect top-patches across all cases for a combined CSV
    all_top_records: List[Dict[str, Any]] = []

    # Per-case summaries
    per_case_summary = {}

    for cid, recs in case_to_records.items():
        if len(recs) == 0:
            continue

        # Extract effective weights for this case
        weights = np.array([r["effective_weight"] for r in recs], dtype=np.float32)
        num_case_patches = len(weights)

        # Number of patches to keep in this case
        k = max(1, int(num_case_patches * top_percent / 100.0))

        # Indices of top-k weights (within this case)
        top_idx = np.argsort(weights)[-k:]  # last k indices (highest weights)

        top_recs = [recs[i] for i in top_idx]

        # Add labels
        label_info = case_label_info.get(cid, {})
        true_label = label_info.get("true_label", None)
        pred_label = label_info.get("pred_label", None)

        for r in top_recs:
            r_with_labels = {
                **r,
                "true_label": true_label,
                "pred_label": pred_label,
            }
            all_top_records.append(r_with_labels)

        # Build per-case stats
        stain_counts = defaultdict(int)
        slice_counts = defaultdict(int)   # key: (stain, slice_idx)

        for r in top_recs:
            stain = r["stain"]
            slice_idx = r["slice_idx"]
            stain_counts[stain] += 1
            slice_counts[(stain, slice_idx)] += 1

        per_case_summary[cid] = {
            "true_label": true_label,
            "pred_label": pred_label,
            "num_total_patches": num_case_patches,
            "num_top_patches": len(top_recs),
            "stain_counts": dict(stain_counts),
            "slice_counts": dict(slice_counts),
        }


    # --- Save CSV of all top patches across all cases ---
    csv_path = os.path.join(output_dir, f"top_effective_patches_per_case_{top_percent:.1f}pct.csv")
    fieldnames = [
        "case_id",
        "stain",
        "slice_idx",
        "patch_idx",
        "patch_attn_weight",
        "slice_attn_weight",
        "stain_attn_weight",
        "effective_weight",
        "true_label",
        "pred_label",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in all_top_records:
            writer.writerow(rec)

    print(f"Per-case top {top_percent:.1f}% effective patches CSV saved to: {csv_path}")

    # --- Save human-readable per-case summary ---
    summary_path = os.path.join(output_dir, f"top_effective_patches_per_case_summary_{top_percent:.1f}pct.txt")
    with open(summary_path, "w") as f:
        f.write(f"TOP {top_percent:.1f}% EFFECTIVE PATCHES PER CASE\n")
        f.write("=" * 60 + "\n\n")

        for cid, info in per_case_summary.items():
            f.write(f"Case {cid}:\n")
            f.write(f"  True label: {info['true_label']}\n")
            f.write(f"  Pred label: {info['pred_label']}\n")
            f.write(f"  # total patches: {info['num_total_patches']}\n")
            f.write(f"  # top patches (per-case): {info['num_top_patches']}\n")

            f.write("  Top-patch counts by stain:\n")
            for stain, cnt in sorted(info["stain_counts"].items()):
                f.write(f"    {stain}: {cnt}\n")

            f.write("  Top-patch counts by (stain, slice_idx):\n")
            for (stain, s_idx), cnt in sorted(info["slice_counts"].items(),
                                              key=lambda x: (x[0][0], x[0][1])):
                f.write(f"    ({stain}, slice {s_idx}): {cnt}\n")

            f.write("\n")

    print(f"Per-case top {top_percent:.1f}% summary saved to: {summary_path}")
