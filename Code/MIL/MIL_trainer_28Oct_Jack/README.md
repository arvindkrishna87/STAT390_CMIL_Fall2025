# Hierarchical Attention MIL Trainer

## Overview

This implementation converts the original Jupyter notebook into a modular Python structure suitable for Quest. The model uses three levels of attention:

1. **Patch-level attention**: Within each stain-slice
2. **Stain-level attention**: Across slices within each stain  
3. **Case-level attention**: Across different stains (H&E, Melan, SOX10)


## Usage

### Training with Attention Analysis
```bash
python main.py --analyze_attention --attention_top_n 5
```

### Evaluation Only
```bash
python main.py --eval_only --resume /path/to/checkpoint.pth --analyze_attention
```

### Using Existing Data Splits
```bash
python main.py --load_splits ./runs/run_20241028_143022/data_splits.npz
```

### Advanced Options
```bash
python main.py \
    --epochs 10 \
    --lr 1e-4 \
    --embed_dim 512 \
    --per_slice_cap 800 \
    --analyze_attention \
    --attention_top_n 10 \
    --load_splits /path/to/data_splits.npz
```

## Output Structure

Each run creates a timestamped directory with all results:

```
./runs/run_YYYYMMDD_HHMMSS/
├── results.txt                    # Summary metrics
├── predictions.csv                # Per-case predictions
├── confusion_matrix.png           # Visual confusion matrix
├── data_splits.npz                # Case IDs for reproducibility
├── checkpoints/                   # Model weights
│   └── *.pth                      # If not --eval_only
└── attention_analysis/            # (if --analyze_attention)
    ├── attention_summary.txt
    ├── stain_attention_distribution.png
    └── patch_attention/
        ├── case_*_*_slice*_top_patches.png
        └── case_*_*_slice*_bottom_patches.png
```

## Output Files

### Always Generated

**results.txt**
- Test loss and accuracy
- Number of samples
- Checkpoint information

**predictions.csv**
- `case_id`: Case identifier
- `true_label`: Ground truth (0=benign, 1=high-grade)
- `predicted_label`: Model prediction
- `prob_benign`: Probability for benign class
- `prob_high_grade`: Probability for high-grade class
- `correct`: Boolean indicating correct prediction

**confusion_matrix.png**
- confusion matrix with counts for predictions on the test set

**data_splits.npz**
Contains case IDs for each split:
- `train_cases`: Training set case IDs
- `val_cases`: Validation set case IDs
- `test_cases`: Test set case IDs

### Optional: Attention Analysis (--analyze_attention)

**attention_summary.txt**
- Most attended stain per case
- Stain-level attention weights
- Slice-level attention patterns

**patch_attention/ folder**
- Top N most attended patches per slice
- Bottom N least attended patches per slice
- Denormalized images with attention weights

**stain_attention_distribution.png**
- Box plot showing attention distribution across stains

## Data Format

The model expects:
- **Patches**: PNG images organized by case and stain
- **Labels CSV**: Case IDs with corresponding class labels
- **Naming Convention**: `case_{case_id}_{slice_id}_{stain}_patch{n}.png`

## Project Structure

```
MIL_trainer_28Oct_Jack/
├── config.py              # Configuration and paths
├── data_utils.py          # Data loading and preprocessing
├── models.py              # Model architectures
├── dataset.py             # Dataset classes and transforms
├── trainer.py             # Training and validation logic
├── attention_analysis.py  # Attention visualization
├── utils.py               # Helper functions
├── main.py                # Main training script
└── requirements.txt       # Dependencies
```