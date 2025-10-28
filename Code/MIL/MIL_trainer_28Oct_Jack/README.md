# Hierarchical Attention MIL Trainer

A modular implementation of Hierarchical Attention Multiple Instance Learning (MIL) for multi-stain pathology image classification.

## Overview

This implementation converts the original Jupyter notebook into a clean, modular Python structure suitable for HPC environments. The model uses three levels of attention:

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

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Update data paths in `config.py` to match your HPC environment:
```python
DATA_PATHS = {
    'labels_csv': './data/case_grade_match.csv',
    'patches_dir': './data/CMIL_SP2025_Patches_Apr27',
    'patch_dir_cache': './data/patch_dir.npy',
    'checkpoint_dir': './checkpoints'
}
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

## Configuration

Key configuration options in `config.py`:

- **Model Config**: Embedding dimensions, attention parameters, stain types
- **Training Config**: Learning rate, epochs, batch size, optimizer settings
- **Data Config**: Image preprocessing, data split ratios
- **Paths**: Data directories and checkpoint locations

## Data Format

The model expects:
- **Patches**: PNG images organized by case and stain
- **Labels CSV**: Case IDs with corresponding class labels
- **Naming Convention**: `case_{case_id}_{slice_id}_{stain}_patch{n}.png`

Example patch names:
```
case_1_match_1_h&e_patch1.png
case_1_match_1_melan_patch1.png
case_1_unmatched_2_sox10_patch1.png
```

## Model Architecture

### HierarchicalAttnMIL
- **Base**: DenseNet-121 feature extractor
- **Patch Projector**: Maps CNN features to embeddings
- **Three Attention Levels**: Patch → Stain → Case
- **Classifier**: Final linear layer for binary classification

### AttentionPool
- Learnable attention mechanism
- Weighted aggregation of features
- Returns attention weights for visualization

## Features

- **Modular Design**: Clean separation of concerns
- **HPC Ready**: No Google Drive dependencies
- **Reproducible**: Seed setting and data split saving
- **Monitoring**: Training progress, memory usage, data integrity checks
- **Flexible**: Command-line arguments for easy experimentation
- **Robust**: Error handling and data validation

## Output

Training creates a timestamped run directory containing:
- Model checkpoints
- Training logs
- Data split information
- Final evaluation results

## Memory Considerations

For HPC environments:
- Adjust `per_slice_cap` to control memory usage
- Use `num_workers=0` if encountering multiprocessing issues
- Monitor GPU memory with built-in utilities

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `per_slice_cap` or `batch_size`
2. **Data Loading Errors**: Check file paths and naming conventions
3. **CUDA Issues**: Ensure PyTorch CUDA version matches your environment

### Data Validation

The trainer includes comprehensive data integrity checks:
- Missing labels detection
- Empty cases identification
- Stain coverage analysis
- Data leakage verification

## Citation

If you use this code, please cite the original work and methodology.

## License

[Add your license information here]