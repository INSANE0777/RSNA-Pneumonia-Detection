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

## Example Results Format

```
============================================================
EVALUATION REPORT - RSNA Pneumonia Detection
============================================================

IoU Threshold: 0.5
Score Threshold: 0.5

--- MANDATORY METRICS ---
  Mean IoU:        0.4521
  Median IoU:      0.4387
  Min IoU:         0.1023
  Max IoU:         0.8912
  AP@0.5 (11-pt):  0.2345
  mAP (COCO):      0.2456

--- Additional Metrics ---
  Precision:       0.3124
  Recall:          0.4567
  F1 Score:        0.3712
  Total GT Boxes:  9555
  Total Predictions: 8234
  Matched Pairs:   3421
============================================================
```

## Notes

- Training requires GPU for reasonable speed (adjust batch_size if needed)
- The dataset is large; first run will take time to load DICOM files
- For quick testing, reduce `num_epochs` or use a subset of data
- IoU and mAP values depend heavily on training convergence and data quality
