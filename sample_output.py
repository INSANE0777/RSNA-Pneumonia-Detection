"""Generate a sample pneumonia detection visualization without needing a trained model.

This script creates a synthetic chest X-ray image and overlays purple bounding
boxes with labels, matching the style shown in the sample image.

Usage:
    python sample_output.py
    python sample_output.py --save output/sample_pneumonia_detection.png
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from visualize import visualize_purple_style


def create_synthetic_xray(size=(1024, 1024), seed=42):
    """Generate a synthetic chest X-ray for demonstration.

    Creates a grayscale image with lung shapes, rib cage outline,
    heart shadow, and spine — realistic enough for visualization demo.

    Args:
        size: (height, width) of the output image.
        seed: Random seed for reproducibility.

    Returns:
        PIL Image in grayscale mode.
    """
    np.random.seed(seed)
    h, w = size

    # Start with dark background (air / outside body)
    xray = np.ones((h, w), dtype=np.float32) * 30

    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2

    # --- Rib cage / lung area (ellipse) ---
    lung_mask = (
        (x - cx) ** 2 / (w * 0.38) ** 2 +
        (y - cy) ** 2 / (h * 0.42) ** 2
    ) <= 1
    xray[lung_mask] = 140  # Lung tissue brightness

    # --- Left lung (slightly darker for realism) ---
    left_lung = (
        (x - cx * 0.75) ** 2 / (w * 0.18) ** 2 +
        (y - cy * 0.9) ** 2 / (h * 0.35) ** 2
    ) <= 1
    xray[left_lung] = 160

    # --- Right lung ---
    right_lung = (
        (x - cx * 1.25) ** 2 / (w * 0.18) ** 2 +
        (y - cy * 0.9) ** 2 / (h * 0.35) ** 2
    ) <= 1
    xray[right_lung] = 155

    # --- Heart shadow (left-mid, darker) ---
    heart = (
        (x - cx * 0.88) ** 2 / (w * 0.14) ** 2 +
        (y - cy * 1.15) ** 2 / (h * 0.20) ** 2
    ) <= 1
    xray[heart] = 90

    # --- Spine shadow (vertical center strip) ---
    spine = np.abs(x - cx) < w * 0.025
    xray[spine] = 70

    # --- Clavicles (horizontal arcs near top) ---
    clavicle_y = cy * 0.45
    clavicle_mask = (
        np.abs(y - clavicle_y) < h * 0.03
    ) & (
        np.abs(x - cx) < w * 0.35
    )
    xray[clavicle_mask] = 110

    # --- Ribs (horizontal stripes) ---
    for rib_y in np.linspace(cy * 0.55, cy * 1.35, 6):
        rib_mask = (
            np.abs(y - rib_y) < h * 0.015
        ) & (
            (x - cx) ** 2 / (w * 0.32) ** 2 +
            (y - cy) ** 2 / (h * 0.38) ** 2 <= 1
        )
        xray[rib_mask] = 85

    # --- Diaphragm curve (bottom arc) ---
    diaphragm_y = cy * 1.45
    diaphragm = (
        (y - diaphragm_y) ** 2 / (h * 0.12) ** 2 +
        (x - cx) ** 2 / (w * 0.35) ** 2 <= 1
    ) & (y > diaphragm_y)
    xray[diaphragm] = 60

    # --- Add subtle noise for realism ---
    noise = np.random.normal(0, 4, size)
    xray = np.clip(xray + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(xray, mode="L")


def get_sample_detections(image_size=(1024, 1024)):
    """Return sample pneumonia bounding boxes for the synthetic X-ray.

    Boxes are in [x1, y1, x2, y2] format, positioned within the lung regions.

    Args:
        image_size: (height, width) of the image.

    Returns:
        boxes: List of bounding boxes.
        scores: List of confidence scores.
        labels: List of labels.
    """
    h, w = image_size
    cy, cx = h // 2, w // 2

    # Sample detections positioned in realistic lung areas
    boxes = [
        # Left upper lung
        [cx * 0.55, cy * 0.55, cx * 0.85, cy * 0.85],
        # Left lower lung
        [cx * 0.50, cy * 1.05, cx * 0.80, cy * 1.35],
        # Right upper lung
        [cx * 1.15, cy * 0.60, cx * 1.45, cy * 0.90],
        # Right mid-lung
        [cx * 1.10, cy * 0.95, cx * 1.40, cy * 1.25],
        # Central lower (near heart border)
        [cx * 0.90, cy * 1.10, cx * 1.20, cy * 1.40],
    ]

    scores = [0.94, 0.89, 0.87, 0.82, 0.76]
    labels = ["pneumonia"] * len(boxes)

    return boxes, scores, labels


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample pneumonia detection visualization"
    )
    parser.add_argument(
        "--size", type=int, default=1024,
        help="Image size (height and width)"
    )
    parser.add_argument(
        "--save", type=str, default="output/sample_pneumonia_detection.png",
        help="Path to save the output image"
    )
    parser.add_argument(
        "--show", action="store_true", default=True,
        help="Display the figure (default: True)"
    )
    parser.add_argument(
        "--no-show", action="store_false", dest="show",
        help="Do not display the figure"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible synthetic image"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Sample Pneumonia Detection Visualization")
    print("=" * 60)

    # Create synthetic chest X-ray
    print(f"\nGenerating synthetic chest X-ray ({args.size}x{args.size})...")
    image = create_synthetic_xray(size=(args.size, args.size), seed=args.seed)

    # Get sample detections
    boxes, scores, labels = get_sample_detections(image_size=(args.size, args.size))

    print(f"Generated {len(boxes)} sample pneumonia detections:")
    for i, (box, score) in enumerate(zip(boxes, scores)):
        print(f"  Box {i+1}: [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}] | Score: {score:.2f}")

    # Visualize with purple style
    print(f"\nRendering purple-style visualization...")
    fig, ax = visualize_purple_style(
        image=image,
        boxes=boxes,
        scores=scores,
        labels=labels,
        save_path=args.save,
        show_axes=True,
        figsize=(10, 10),
        dpi=150,
    )

    print(f"\nSaved to: {args.save}")
    print("=" * 60)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
