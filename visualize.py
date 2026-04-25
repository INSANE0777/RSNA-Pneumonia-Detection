"""Visualization utilities for RSNA Pneumonia Detection.

Show predicted bounding boxes and compare performance before vs after improvements.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pydicom
import torch
import cv2

import config


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image tensor for visualization.

    Args:
        tensor: Normalized image tensor (C, H, W).
        mean: Normalization means.
        std: Normalization stds.

    Returns:
        Denormalized numpy array (H, W, C) in [0, 255].
    """
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img


def draw_boxes_on_image(image, boxes, labels=None, scores=None,
                        color='red', linewidth=2, alpha=1.0,
                        box_format='xyxy', scale=1.0):
    """Draw bounding boxes on an image.

    Args:
        image: PIL Image or numpy array.
        boxes: List of boxes in [x1, y1, x2, y2] format.
        labels: Optional labels for each box.
        scores: Optional scores for each box.
        color: Box color.
        linewidth: Line width.
        alpha: Transparency.
        box_format: 'xyxy' or 'xywh'.
        scale: Scale factor to apply to boxes.

    Returns:
        Image with boxes drawn.
    """
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        x1 *= scale
        y1 *= scale
        x2 *= scale
        y2 *= scale

        width = x2 - x1
        height = y2 - y1

        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=linewidth, edgecolor=color, facecolor='none', alpha=alpha
        )
        ax.add_patch(rect)

        # Add label
        label_text = ""
        if labels is not None:
            label_text += f"Class {labels[i]}"
        if scores is not None:
            label_text += f" {scores[i]:.3f}" if label_text else f"{scores[i]:.3f}"

        if label_text:
            ax.text(x1, y1 - 5, label_text, color=color, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.axis('off')
    return fig, ax


@torch.no_grad()
def visualize_predictions(model, data_loader, device, num_samples=4,
                          save_dir='output/visualizations',
                          score_threshold=0.5, iou_threshold=0.5):
    """Visualize model predictions on validation samples.

    Args:
        model: Detection model.
        data_loader: Data loader.
        device: Device.
        num_samples: Number of samples to visualize.
        save_dir: Directory to save figures.
        score_threshold: Score threshold for predictions.
        iou_threshold: IoU threshold for matching.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    samples_collected = 0
    for images, targets in data_loader:
        if samples_collected >= num_samples:
            break

        images = [img.to(device) for img in images]
        outputs = model(images)

        for i in range(len(images)):
            if samples_collected >= num_samples:
                break

            # Denormalize image
            img = denormalize_image(images[i])

            # Ground truth boxes
            gt_boxes = targets[i]['boxes'].cpu().numpy()
            gt_labels = targets[i]['labels'].cpu().numpy()

            # Predicted boxes
            output = outputs[i]
            scores = output['scores'].cpu().numpy()
            boxes = output['boxes'].cpu().numpy()
            labels = output['labels'].cpu().numpy()

            mask = scores >= score_threshold
            pred_boxes = boxes[mask]
            pred_scores = scores[mask]
            pred_labels = labels[mask]

            # Create figure with subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Original image
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # Ground truth
            axes[1].imshow(img)
            for box in gt_boxes:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor='green', facecolor='none'
                )
                axes[1].add_patch(rect)
            axes[1].set_title(f'Ground Truth ({len(gt_boxes)} boxes)')
            axes[1].axis('off')

            # Predictions
            axes[2].imshow(img)
            for j, box in enumerate(pred_boxes):
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                axes[2].add_patch(rect)
                axes[2].text(x1, y1-5, f'{pred_scores[j]:.3f}',
                            color='red', fontsize=9,
                            bbox=dict(facecolor='white', alpha=0.7))
            axes[2].set_title(f'Predictions ({len(pred_boxes)} boxes)')
            axes[2].axis('off')

            plt.tight_layout()
            save_path = os.path.join(save_dir, f'prediction_{samples_collected}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            samples_collected += 1

    print(f"Saved {samples_collected} visualization(s) to {save_dir}")


def plot_training_history(history, save_path='output/training_history.png'):
    """Plot training and validation loss curves.

    Args:
        history: Dictionary with 'train_loss' and 'val_loss' lists.
        save_path: Path to save figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Learning rate
    if 'learning_rate' in history:
        axes[1].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training history plot to {save_path}")


def plot_pr_curve(metrics, save_path='output/pr_curve.png'):
    """Plot Precision-Recall curve.

    Args:
        metrics: Dictionary with 'precisions' and 'recalls' lists.
        save_path: Path to save figure.
    """
    precisions = metrics.get('precisions', [])
    recalls = metrics.get('recalls', [])
    ap = metrics.get('AP@0.5', 0)

    if len(precisions) == 0 or len(recalls) == 0:
        print("No PR data to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(recalls, precisions, 'b-', linewidth=2)
    ax.fill_between(recalls, precisions, alpha=0.3)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve (AP@0.5 = {ap:.4f})', fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PR curve to {save_path}")


def plot_iou_distribution(ious, save_path='output/iou_distribution.png'):
    """Plot histogram of IoU values.

    Args:
        ious: List of IoU values.
        save_path: Path to save figure.
    """
    if len(ious) == 0:
        print("No IoU data to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(ious, bins=30, range=(0, 1), color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(ious), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(ious):.3f}')
    ax.axvline(np.median(ious), color='green', linestyle='--', linewidth=2, label=f'Median = {np.median(ious):.3f}')
    ax.set_xlabel('IoU', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of IoU Values (Matched Detections)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved IoU distribution plot to {save_path}")


def visualize_purple_style(image, boxes, scores=None, labels=None,
                           save_path=None, scale=1.0, show_axes=True,
                           figsize=(10, 10), dpi=150):
    """Visualize detections with purple boxes matching sample image style.

    Style:
        - Purple bounding boxes with white text labels on purple backgrounds.
        - Black figure/axes background.
        - Visible axis ticks (0-1000 style) when show_axes=True.
        - Grayscale X-ray display.

    Args:
        image: PIL Image or numpy array (grayscale or RGB).
        boxes: List of boxes in [x1, y1, x2, y2] format.
        scores: Optional confidence scores.
        labels: Optional class labels.
        save_path: Path to save figure.
        scale: Scale factor to apply to boxes.
        show_axes: Whether to show axis ticks and labels.
        figsize: Figure size.
        dpi: Save resolution.

    Returns:
        fig, ax: Matplotlib figure and axis.
    """
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()

    fig, ax = plt.subplots(1, figsize=figsize)

    # Set black background
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Display in grayscale
    if len(img.shape) == 2:
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    else:
        ax.imshow(img, cmap='gray')

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        x1 *= scale
        y1 *= scale
        x2 *= scale
        y2 *= scale

        width = x2 - x1
        height = y2 - y1

        # Purple bounding box
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2.5, edgecolor='purple', facecolor='none'
        )
        ax.add_patch(rect)

        # Build label text
        label_text = ""
        if labels is not None:
            label_text += str(labels[i])
        if scores is not None:
            label_text += f" {scores[i]:.2f}" if label_text else f"{scores[i]:.2f}"
        if not label_text:
            label_text = "pneumonia"

        # Purple label background with white text
        ax.text(
            x1, y1 - 5, label_text,
            color='white', fontsize=10, fontweight='bold',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='purple',
                edgecolor='purple',
                alpha=0.9
            ),
            verticalalignment='bottom'
        )

    # Axis styling
    if show_axes:
        h, w = img.shape[:2]
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

        tick_step_x = max(1, w // 5)
        tick_step_y = max(1, h // 5)
        ax.set_xticks(np.arange(0, w + tick_step_x, tick_step_x))
        ax.set_yticks(np.arange(0, h + tick_step_y, tick_step_y))

        ax.tick_params(colors='white', which='both')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
    else:
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='black', edgecolor='none')
        print(f"Saved purple-style visualization to {save_path}")

    return fig, ax


def compare_before_after(baseline_metrics, improved_metrics,
                         save_path='output/comparison.png'):
    """Compare metrics before and after improvements.

    Args:
        baseline_metrics: Metrics dict before improvements.
        improved_metrics: Metrics dict after improvements.
        save_path: Path to save comparison figure.
    """
    metrics_names = ['mAP', 'AP@0.5', 'mean_iou', 'precision', 'recall', 'f1']
    baseline_vals = [baseline_metrics.get(m, 0) for m in metrics_names]
    improved_vals = [improved_metrics.get(m, 0) for m in metrics_names]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='lightcoral', edgecolor='black')
    bars2 = ax.bar(x + width/2, improved_vals, width, label='With Improvements', color='steelblue', edgecolor='black')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Comparison: Baseline vs Improved Model', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim([0, max(max(baseline_vals), max(improved_vals)) * 1.2 + 0.05])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {save_path}")

    # Print comparison table
    print("\n" + "="*60)
    print("COMPARISON: Baseline vs Improved Model")
    print("="*60)
    print(f"{'Metric':<20} {'Baseline':>12} {'Improved':>12} {'Delta':>12}")
    print("-"*60)
    for name, base, imp in zip(metrics_names, baseline_vals, improved_vals):
        delta = imp - base
        print(f"{name:<20} {base:>12.4f} {imp:>12.4f} {delta:>+12.4f}")
    print("="*60)
