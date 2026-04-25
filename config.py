"""Configuration for RSNA Pneumonia Detection."""

import os
import torch

# Paths
# Data is at project root (not inside data/ subdirectory)
DATA_DIR = "."
TRAIN_DIR = os.path.join(DATA_DIR, "stage_2_train_images")
TEST_DIR = os.path.join(DATA_DIR, "stage_2_test_images")
TRAIN_LABELS = os.path.join(DATA_DIR, "stage_2_train_labels.csv")
CLASS_INFO = os.path.join(DATA_DIR, "stage_2_detailed_class_info.csv")

# Output
OUTPUT_DIR = "output"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Image
IMAGE_SIZE = 512
ORIGINAL_SIZE = 1024

# Training
# BATCH_SIZE reduced to 2 for RTX 3050 6GB VRAM (~4.5GB free)
BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
# NUM_WORKERS=2+ enables parallel data loading (Windows spawn requires if __name__=='__main__')
NUM_WORKERS = 2

# Model
BACKBONE = "resnet50"
PRETRAINED = True

# Detection
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5

# Data split
TRAIN_VAL_SPLIT = 0.8
RANDOM_SEED = 42

# Augmentation
USE_AUGMENTATION = True

# GPU Configuration
USE_AMP = True  # Automatic Mixed Precision (faster training, less memory)
CUDNN_BENCHMARK = True  # Optimize for fixed input sizes
USE_MULTI_GPU = torch.cuda.device_count() > 1  # Auto-detect multi-GPU
PERSISTENT_WORKERS = True  # Keep DataLoader workers alive between epochs
PREFETCH_FACTOR = 2  # Samples loaded per worker in advance
NON_BLOCKING = True  # Non-blocking GPU transfers
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 2 * 4 = 8 (better convergence, same memory)

# Early stopping
EARLY_STOPPING_PATIENCE = 3  # Stop if val loss doesn't improve for 3 epochs (deadline-friendly)
