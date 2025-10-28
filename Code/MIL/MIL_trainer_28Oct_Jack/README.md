# Hierarchical Attention MIL Trainer

A modular implementation of Hierarchical Attention MIL.

## Overview

This implementation converts the original Jupyter notebook into a clean, modular Python structure suitable for Quest. The model uses three levels of attention:

1. **Patch-level attention**: Within each stain-slice
2. **Stain-level attention**: Across slices within each stain  
3. **Case-level attention**: Across different stains (H&E, Melan, SOX10)

## Project Structure

```
MIL_trainer_28Oct_Jack/
├── config.py          # Configuration and constants
├── data_utils.py       # Data loading and preprocessing utilities
├── models.py           # Model architectures (AttentionPool, HierarchicalAttnMIL)
├── dataset.py          # Custom dataset classes and transforms
├── trainer.py          # Training and validation logic
├── utils.py            # Helper functions and utilities
├── main.py             # Main training script
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Usage

### Basic Training

```bash
python main.py
```

### Advanced Usage

```bash
python main.py \
    --epochs 10 \
    --lr 1e-4 \
    --embed_dim 512 \
    --per_slice_cap 800 \
    --patches_dir /path/to/patches \
    --labels_csv /path/to/labels.csv
```

### Resume Training

```bash
python main.py --resume /path/to/checkpoint.pth
```

### Evaluation Only

```bash
python main.py --eval_only --resume /path/to/checkpoint.pth
```

## Data Format

The model expects:
- **Patches**: PNG images organized by case and stain
- **Labels CSV**: Case IDs with corresponding class labels
- **Naming Convention**: `case_{case_id}_{slice_id}_{stain}_patch{n}.png`