# RSNA Pneumonia Detection Challenge

A complete CNN-based object detection solution for detecting pneumonia in chest X-rays and predicting bounding boxes for infected regions.

## Project Structure

```
.
├── config.py              # Configuration settings
├── data_preparation.py    # Data loading, preprocessing, train/val split
├── model.py               # Model architecture (Faster R-CNN + ResNet50-FPN)
├── train.py               # Training loop with loss tracking
├── evaluate.py            # IoU, AP, mAP metrics (MANDATORY)
├── visualize.py           # Bounding box visualization and comparison plots
├── main.py                # Main pipeline orchestration
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Dataset

Download from [Kaggle: RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

Place data in the `data/` directory:
```
data/
  stage_2_train_images/     # DICOM training images
  stage_2_test_images/      # DICOM test images
  stage_2_train_labels.csv  # Bounding box annotations
  stage_2_detailed_class_info.csv  # Class information
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train Model (with all improvements)

```bash
python main.py --mode train --epochs 20 --augmentation --pretrained --lr 1e-4
```

### 2. Evaluate Saved Model

```bash
python main.py --mode evaluate --checkpoint output/best_model.pth
```

### 3. Compare Baseline vs Improved Model

```bash
python main.py --mode compare --epochs 15
```

This runs:
- **Baseline**: No augmentation, no transfer learning, SGD optimizer
- **Improved**: Data augmentation + Transfer learning + AdamW + LR scheduling + Early stopping

### 4. Visualize Predictions

```bash
python main.py --mode visualize --checkpoint output/best_model.pth --num_viz 10
```

## Model Architecture

- **Backbone**: ResNet50 with Feature Pyramid Network (FPN)
- **Detector**: Faster R-CNN
- **Transfer Learning**: Pretrained on COCO, fine-tuned on chest X-rays
- **Classes**: Background (0) + Pneumonia (1)

## Mandatory Evaluation Metrics

The following metrics are computed and reported:

| Metric | Description |
|--------|-------------|
| **IoU** | Intersection over Union for matched detections |
| **Mean IoU** | Average IoU across all matched pairs |
| **AP@0.5** | Average Precision at IoU threshold = 0.5 (11-point) |
| **mAP** | Mean Average Precision (COCO-style, all-point) |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1 Score** | Harmonic mean of precision and recall |

## Improvements Applied

1. **Data Augmentation** (in `data_preparation.py`):
   - Random horizontal flip (p=0.5)
   - Color jitter (brightness, contrast)
   - Random affine transform (rotation, translation, scale)

2. **Transfer Learning** (in `model.py`):
   - Pretrained ResNet50-FPN backbone
   - Progressive unfreezing of layers

3. **Hyperparameter Tuning** (in `train.py`):
   - AdamW optimizer with weight decay
   - ReduceLROnPlateau learning rate scheduling
   - Gradient clipping
   - Early stopping

## Output Files

After running, check the `output/` directory:

```
output/
  best_model.pth              # Best model checkpoint
  baseline_model.pth          # Baseline model checkpoint
  checkpoints/                # Per-epoch checkpoints
  visualizations/             # Predicted bounding boxes
  training_history.png        # Loss curves
  baseline_history.png        # Baseline training curves
  improved_history.png      # Improved training curves
  pr_curve.png               # Precision-Recall curve
  iou_distribution.png       # IoU histogram
  comparison.png             # Baseline vs Improved bar chart
  metrics.json               # Evaluation metrics
  comparison_results.json    # Side-by-side comparison
```

## Evaluation Results

```text
A:\DEEP LEARNING\Assignmenr> python main.py --mode evaluate --checkpoint output/checkpoints/checkpoint_epoch_2.pth

======================================================================
RSNA PNEUMONIA DETECTION - EVALUATION
======================================================================
Using GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU (6.0GB)
  CUDA version: 12.4
  PyTorch CUDA: 90100
  AMP (Mixed Precision): Available

Loading data...
Loaded 30227 annotation rows for 26684 unique patients
Pneumonia cases: 9555
Train: 21347 patients, Val: 5337 patients
DataLoader config: batch_size=2, workers=2, pin_memory=True, persistent=True, prefetch=2

Evaluating...

============================================================
EVALUATION REPORT - IoU Threshold: 0.3
============================================================
--- MANDATORY METRICS ---
  Mean IoU:        0.6266
  Median IoU:      0.6363
  Min IoU:         0.3001
  Max IoU:         0.9296
  AP@0.5 (11-pt):  0.4526
  mAP (COCO):      0.4441

--- Additional Metrics ---
  Precision:       0.3927
  Recall:          0.6569
  F1 Score:        0.4916
  Total GT Boxes:  1886
  Matched Pairs:   1239
============================================================

============================================================
EVALUATION REPORT - IoU Threshold: 0.5
============================================================
--- MANDATORY METRICS ---
  Mean IoU:        0.6769
  Median IoU:      0.6702
  Min IoU:         0.5000
  Max IoU:         0.9296
  AP@0.5 (11-pt):  0.3366
  mAP (COCO):      0.3055

--- Additional Metrics ---
  Precision:       0.3182
  Recall:          0.5323
  F1 Score:        0.3983
  Total GT Boxes:  1886
  Matched Pairs:   1004
============================================================

============================================================
EVALUATION REPORT - IoU Threshold: 0.75
============================================================
--- MANDATORY METRICS ---
  Mean IoU:        0.8087
  Median IoU:      0.8023
  Min IoU:         0.7502
  Max IoU:         0.9296
  AP@0.5 (11-pt):  0.0452
  mAP (COCO):      0.0251

--- Additional Metrics ---
  Precision:       0.0821
  Recall:          0.1373
  F1 Score:        0.1028
  Total GT Boxes:  1886
  Matched Pairs:   259
============================================================

============================================================
Mean Average Precision (mAP) across thresholds: 0.2781
============================================================
```


## Notes

- Training requires GPU for reasonable speed (adjust batch_size if needed)
- The dataset is large; first run will take time to load DICOM files
- For quick testing, reduce `num_epochs` or use a subset of data
- IoU and mAP values depend heavily on training convergence and data quality
