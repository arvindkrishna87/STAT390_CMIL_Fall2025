"""
Configuration file for MIL training
"""
import os
from typing import Tuple

# Data paths (adjust these for your HPC environment)
DATA_PATHS = {
    'labels_csv': '/projects/e32998/MIL_training/case_grade_match.csv',
    'patches_dir': '/projects/e32998/patches',
    'runs_dir': '/projects/e32998/MIL_training/runs'  # Base directory for training runs
}

# Model configuration
MODEL_CONFIG = {
    'num_classes': 2,
    'embed_dim': 512,
    'attention_hidden_dim': 128,
    'per_slice_cap': 800,
    'max_slices_per_stain': None,
    'stains': ('h&e', 'melan', 'sox10')
}

# Training configuration
TRAINING_CONFIG = {
    'epochs': 20,
    'batch_size': 1,  # MIL typically uses batch_size=1
    'learning_rate': 2e-4,
    'weight_decay': 1e-5,
    'num_workers': 2,
    'pin_memory': True,
    'random_state': 42,
    'class_weights': [2.0, 1.0]  # [benign_weight, high_grade_weight] - handle class imbalance
}

# Data split configuration
SPLIT_CONFIG = {
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,
    'stratify': True
}

# Image preprocessing
IMAGE_CONFIG = {
    'image_size': (224, 224),
    'normalize_mean': [0.485, 0.456, 0.406],  # DenseNet mean
    'normalize_std': [0.229, 0.224, 0.225]   # DenseNet std
}

# Valid classes for filtering
VALID_CLASSES = [1.0, 3.0, 4.0]

# Device configuration
DEVICE = 'cuda' if os.environ.get('CUDA_AVAILABLE', 'true').lower() == 'true' else 'cpu'
