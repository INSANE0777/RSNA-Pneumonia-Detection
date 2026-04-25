"""Evaluation metrics for RSNA Pneumonia Detection.

MANDATORY REQUIREMENTS:
- Intersection over Union (IoU)
- Average Precision (AP) / Mean Average Precision (mAP)
- IoU threshold (e.g., 0.5)
- Show calculated IoU scores
- Report final AP/mAP values

GPU Optimizations:
- Automatic Mixed Precision (AMP) for faster evaluation
- Non-blocking GPU transfers
- Batch processing with GPU memory management
"""

import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

import config


def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) between two bounding boxes.

    Boxes are in [x1, y1, x2, y2] format.

    Args:
        box1: First box [x1, y1, x2, y2].
        box2: Second box [x1, y1, x2, y2].

    Returns:
        IoU value between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou


def compute_ious(pred_boxes, gt_boxes):
    """Compute IoU matrix between predicted and ground truth boxes.

    Args:
        pred_boxes: List of predicted boxes.
        gt_boxes: List of ground truth boxes.

    Returns:
        IoU matrix of shape (num_pred, num_gt).
    """
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(pred, gt)
    return iou_matrix


def compute_ap(recalls, precisions):
    """Compute Average Precision using 11-point interpolation.

    Args:
        recalls: Array of recall values.
        precisions: Array of precision values.

    Returns:
        Average Precision score.
    """
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap


def compute_ap_coco(recalls, precisions):
    """Compute Average Precision using COCO-style all-point interpolation.

    Args:
        recalls: Array of recall values.
        precisions: Array of precision values.

    Returns:
        AP score.
    """
    # Append sentinel values
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Area under PR curve
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate_image(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """Evaluate a single image: compute TP, FP, FN.

    Args:
        pred_boxes: Predicted boxes [[x1,y1,x2,y2], ...].
        pred_scores: Confidence scores for predictions.
        gt_boxes: Ground truth boxes.
        iou_threshold: IoU threshold for matching.

    Returns:
        List of (score, is_tp) tuples.
    """
    if len(pred_boxes) == 0:
        return []

    if len(gt_boxes) == 0:
        return [(score, False) for score in pred_scores]

    iou_matrix = compute_ious(pred_boxes, gt_boxes)

    # Track which GT boxes have been matched
    gt_matched = [False] * len(gt_boxes)
    results = []

    # Sort predictions by score descending
    sorted_indices = np.argsort(pred_scores)[::-1]

    for idx in sorted_indices:
        if len(gt_boxes) == 0:
            results.append((pred_scores[idx], False))
            continue

        best_iou = 0
        best_gt = -1
        for j in range(len(gt_boxes)):
            if gt_matched[j]:
                continue
            iou = iou_matrix[idx, j]
            if iou > best_iou:
                best_iou = iou
                best_gt = j

        if best_iou >= iou_threshold and best_gt >= 0:
            gt_matched[best_gt] = True
            results.append((pred_scores[idx], True))
        else:
            results.append((pred_scores[idx], False))

    return results


@torch.no_grad()
def evaluate_model(model, data_loader, device, iou_threshold=0.5, score_threshold=0.5,
                   use_amp=True, non_blocking=True):
    """Evaluate model and compute IoU, AP, and mAP with GPU optimizations.

    MANDATORY: Reports IoU scores, AP, and mAP values.

    Args:
        model: Detection model.
        data_loader: Validation data loader.
        device: Device.
        iou_threshold: IoU threshold for positive detection.
        score_threshold: Score threshold for predictions.
        use_amp: Use Automatic Mixed Precision for faster evaluation.
        non_blocking: Use non-blocking GPU transfers.

    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()

    all_results = []  # List of (score, is_tp) for all images
    all_ious = []  # All IoU values for matched pairs
    num_gt_total = 0
    num_images_with_gt = 0
    num_images_with_preds = 0

    pbar = tqdm(data_loader, desc="Evaluating")
    for images, targets in pbar:
        # Non-blocking GPU transfer
        images = [img.to(device, non_blocking=non_blocking) for img in images]

        # Get predictions with optional AMP
        if use_amp and torch.cuda.is_available():
            with autocast(device_type='cuda'):
                outputs = model(images)
        else:
            outputs = model(images)

        for i, output in enumerate(outputs):
            gt_boxes = targets[i]['boxes'].cpu().numpy()
            num_gt_total += len(gt_boxes)
            if len(gt_boxes) > 0:
                num_images_with_gt += 1

            # Filter predictions by score
            scores = output['scores'].cpu().numpy()
            boxes = output['boxes'].cpu().numpy()
            labels = output['labels'].cpu().numpy()

            mask = scores >= score_threshold
            pred_boxes = boxes[mask]
            pred_scores = scores[mask]
            pred_labels = labels[mask]

            # Only keep pneumonia class (label = 1)
            pneumonia_mask = pred_labels == 1
            pred_boxes = pred_boxes[pneumonia_mask]
            pred_scores = pred_scores[pneumonia_mask]

            if len(pred_boxes) > 0:
                num_images_with_preds += 1

            # Evaluate this image
            results = evaluate_image(pred_boxes, pred_scores, gt_boxes, iou_threshold)
            all_results.extend(results)

            # Collect IoU values for matched pairs
            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                iou_matrix = compute_ious(pred_boxes, gt_boxes)
                for j in range(len(gt_boxes)):
                    best_iou = np.max(iou_matrix[:, j]) if iou_matrix.shape[0] > 0 else 0
                    if best_iou >= iou_threshold:
                        all_ious.append(best_iou)

        # Clear cache periodically during evaluation to prevent OOM
        if torch.cuda.is_available() and len(all_results) % 500 == 0:
            torch.cuda.empty_cache()

    # Compute metrics
    if len(all_results) == 0:
        return {
            'mAP': 0.0,
            'AP@0.5': 0.0,
            'AP_coco': 0.0,
            'mean_iou': 0.0,
            'median_iou': 0.0,
            'min_iou': 0.0,
            'max_iou': 0.0,
            'num_predictions': 0,
            'num_gt': num_gt_total,
            'num_matched': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'num_images_with_gt': num_images_with_gt,
            'num_images_with_preds': num_images_with_preds,
            'ious': [],
            'precisions': [],
            'recalls': [],
        }

    # Sort all results by score descending
    all_results.sort(key=lambda x: x[0], reverse=True)

    # Compute precision-recall curve
    tp_cumsum = np.cumsum([r[1] for r in all_results])
    fp_cumsum = np.cumsum([not r[1] for r in all_results])

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recalls = tp_cumsum / (num_gt_total + 1e-10)

    # Compute AP
    ap_11point = compute_ap(recalls, precisions)
    ap_coco = compute_ap_coco(recalls, precisions)

    # IoU statistics
    mean_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0
    median_iou = np.median(all_ious) if len(all_ious) > 0 else 0.0
    min_iou = np.min(all_ious) if len(all_ious) > 0 else 0.0
    max_iou = np.max(all_ious) if len(all_ious) > 0 else 0.0

    # Final precision, recall, F1
    total_tp = tp_cumsum[-1] if len(tp_cumsum) > 0 else 0
    total_fp = fp_cumsum[-1] if len(fp_cumsum) > 0 else 0
    precision = total_tp / (total_tp + total_fp + 1e-10)
    recall = total_tp / (num_gt_total + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    metrics = {
        'mAP': ap_coco,
        'AP@0.5': ap_11point,
        'AP_coco': ap_coco,
        'mean_iou': mean_iou,
        'median_iou': median_iou,
        'min_iou': min_iou,
        'max_iou': max_iou,
        'num_predictions': len(all_results),
        'num_gt': num_gt_total,
        'num_matched': len(all_ious),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_images_with_gt': num_images_with_gt,
        'num_images_with_preds': num_images_with_preds,
        'ious': all_ious,
        'precisions': precisions.tolist(),
        'recalls': recalls.tolist(),
    }

    return metrics


def print_metrics(metrics, iou_threshold=0.5):
    """Print evaluation metrics in a formatted report.

    Args:
        metrics: Dictionary from evaluate_model().
        iou_threshold: IoU threshold used.
    """
    print("\n" + "="*60)
    print("EVALUATION REPORT - RSNA Pneumonia Detection")
    print("="*60)
    print(f"\nIoU Threshold: {iou_threshold}")
    print(f"Score Threshold: {config.SCORE_THRESHOLD}")
    print("\n--- MANDATORY METRICS ---")
    print(f"  Mean IoU:        {metrics['mean_iou']:.4f}")
    print(f"  Median IoU:      {metrics['median_iou']:.4f}")
    print(f"  Min IoU:         {metrics['min_iou']:.4f}")
    print(f"  Max IoU:         {metrics['max_iou']:.4f}")
    print(f"  AP@0.5 (11-pt):  {metrics['AP@0.5']:.4f}")
    print(f"  mAP (COCO):      {metrics['mAP']:.4f}")
    print("\n--- Additional Metrics ---")
    print(f"  Precision:       {metrics['precision']:.4f}")
    print(f"  Recall:          {metrics['recall']:.4f}")
    print(f"  F1 Score:        {metrics['f1']:.4f}")
    print(f"  Total GT Boxes:  {metrics['num_gt']}")
    print(f"  Total Predictions: {metrics['num_predictions']}")
    print(f"  Matched Pairs:   {metrics['num_matched']}")
    print("="*60)


def evaluate_with_multiple_thresholds(model, data_loader, device,
                                     iou_thresholds=[0.3, 0.5, 0.75],
                                     score_threshold=0.5,
                                     use_amp=True):
    """Evaluate model at multiple IoU thresholds with GPU optimizations.

    Args:
        model: Detection model.
        data_loader: Validation data loader.
        device: Device.
        iou_thresholds: List of IoU thresholds to test.
        score_threshold: Score threshold for predictions.
        use_amp: Use Automatic Mixed Precision.

    Returns:
        Dictionary mapping threshold to metrics.
    """
    results = {}
    for iou_thresh in iou_thresholds:
        print(f"\nEvaluating at IoU threshold = {iou_thresh}...")
        metrics = evaluate_model(model, data_loader, device,
                                  iou_threshold=iou_thresh,
                                  score_threshold=score_threshold,
                                  use_amp=use_amp)
        print_metrics(metrics, iou_thresh)
        results[iou_thresh] = metrics

    # Compute mAP across thresholds (average of APs)
    map_score = np.mean([results[t]['AP@0.5'] for t in iou_thresholds])
    print(f"\n{'='*60}")
    print(f"Mean Average Precision (mAP) across thresholds: {map_score:.4f}")
    print(f"{'='*60}")

    return results
