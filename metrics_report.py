"""Generate a formatted metrics report from evaluation results.

Usage:
    python metrics_report.py --metrics output/metrics.json
    python metrics_report.py --compare output/comparison_results.json
"""

import json
import argparse
import os


def print_iou_report(metrics):
    """Print detailed IoU report."""
    ious = metrics.get('ious', [])

    print("\n" + "="*60)
    print("INTERSECTION OVER UNION (IoU) REPORT")
    print("="*60)

    if not ious:
        print("No IoU data available.")
        return

    import numpy as np
    ious = np.array(ious)

    print(f"\nTotal matched detections: {len(ious)}")
    print(f"Mean IoU:   {np.mean(ious):.4f}")
    print(f"Median IoU: {np.median(ious):.4f}")
    print(f"Std IoU:    {np.std(ious):.4f}")
    print(f"Min IoU:    {np.min(ious):.4f}")
    print(f"Max IoU:    {np.max(ious):.4f}")

    # IoU distribution buckets
    print("\nIoU Distribution:")
    buckets = [
        (0.0, 0.1, "0.0 - 0.1"),
        (0.1, 0.2, "0.1 - 0.2"),
        (0.2, 0.3, "0.2 - 0.3"),
        (0.3, 0.4, "0.3 - 0.4"),
        (0.4, 0.5, "0.4 - 0.5"),
        (0.5, 0.6, "0.5 - 0.6"),
        (0.6, 0.7, "0.6 - 0.7"),
        (0.7, 0.8, "0.7 - 0.8"),
        (0.8, 0.9, "0.8 - 0.9"),
        (0.9, 1.0, "0.9 - 1.0"),
    ]

    for low, high, label in buckets:
        count = np.sum((ious >= low) & (ious < high))
        pct = count / len(ious) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label}: {count:4d} ({pct:5.1f}%) {bar}")

    print("="*60)


def print_ap_report(metrics):
    """Print Average Precision report."""
    print("\n" + "="*60)
    print("AVERAGE PRECISION (AP) / MEAN AVERAGE PRECISION (mAP) REPORT")
    print("="*60)

    ap_11 = metrics.get('AP@0.5', 0)
    ap_coco = metrics.get('AP_coco', 0)
    map_score = metrics.get('mAP', 0)

    print(f"\nAP@0.5 (11-point interpolation): {ap_11:.4f}")
    print(f"AP (COCO all-point interpolation): {ap_coco:.4f}")
    print(f"mAP (Mean Average Precision):      {map_score:.4f}")

    # AP quality assessment
    print("\nAP Quality Assessment:")
    for name, score in [("AP@0.5", ap_11), ("mAP", map_score)]:
        if score >= 0.5:
            quality = "Excellent"
        elif score >= 0.3:
            quality = "Good"
        elif score >= 0.1:
            quality = "Fair"
        else:
            quality = "Poor"
        print(f"  {name}: {score:.4f} - {quality}")

    print("="*60)


def print_full_report(metrics):
    """Print complete evaluation report."""
    print("\n" + "="*60)
    print("RSNA PNEUMONIA DETECTION - FULL EVALUATION REPORT")
    print("="*60)

    print(f"\n--- Detection Statistics ---")
    print(f"  Total Ground Truth Boxes:  {metrics.get('num_gt', 0)}")
    print(f"  Total Predictions:         {metrics.get('num_predictions', 0)}")
    print(f"  Matched Pairs:             {metrics.get('num_matched', 0)}")
    print(f"  Images with GT:            {metrics.get('num_images_with_gt', 0)}")
    print(f"  Images with Predictions:   {metrics.get('num_images_with_preds', 0)}")

    print(f"\n--- Classification Metrics ---")
    print(f"  Precision:  {metrics.get('precision', 0):.4f}")
    print(f"  Recall:     {metrics.get('recall', 0):.4f}")
    print(f"  F1 Score:   {metrics.get('f1', 0):.4f}")

    print_iou_report(metrics)
    print_ap_report(metrics)

    print("\n" + "="*60)
    print("END OF REPORT")
    print("="*60)


def print_comparison_report(data):
    """Print comparison report between baseline and improved."""
    baseline = data.get('baseline', {})
    improved = data.get('improved', {})

    print("\n" + "="*60)
    print("BASELINE vs IMPROVED MODEL COMPARISON")
    print("="*60)

    metrics = ['mAP', 'AP@0.5', 'mean_iou', 'precision', 'recall', 'f1']

    print(f"\n{'Metric':<20} {'Baseline':>12} {'Improved':>12} {'Change':>12} {'% Change':>12}")
    print("-"*70)

    import numpy as np
    for metric in metrics:
        base = baseline.get(metric, 0)
        imp = improved.get(metric, 0)
        change = imp - base
        pct_change = (change / (base + 1e-10)) * 100
        print(f"{metric:<20} {base:>12.4f} {imp:>12.4f} {change:>+12.4f} {pct_change:>+11.1f}%")

    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', type=str, help='Path to metrics JSON file')
    parser.add_argument('--compare', type=str, help='Path to comparison results JSON')
    args = parser.parse_args()

    if args.metrics:
        if not os.path.exists(args.metrics):
            print(f"File not found: {args.metrics}")
            return
        with open(args.metrics, 'r') as f:
            metrics = json.load(f)
        print_full_report(metrics)

    elif args.compare:
        if not os.path.exists(args.compare):
            print(f"File not found: {args.compare}")
            return
        with open(args.compare, 'r') as f:
            data = json.load(f)
        print_comparison_report(data)

    else:
        # Try to auto-find files
        metrics_path = 'output/metrics.json'
        compare_path = 'output/comparison_results.json'

        if os.path.exists(compare_path):
            with open(compare_path, 'r') as f:
                data = json.load(f)
            print_comparison_report(data)
        elif os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            print_full_report(metrics)
        else:
            print("No metrics files found. Please run evaluation first.")


if __name__ == '__main__':
    main()
