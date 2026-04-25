"""Quick demo script to run inference on a single image or batch.

Usage:
    python demo.py --image data/stage_2_train_images/0004cfab-14fd-4e18-9f3f-7569e9e2b6c2.dcm
    python demo.py --image_dir data/stage_2_test_images --output output/demo_results
"""

import os
import argparse
import numpy as np
import pydicom
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import config
from model import get_model
from visualize import visualize_purple_style


def load_dicom(dcm_path, image_size=512):
    """Load and preprocess a DICOM image."""
    dcm = pydicom.dcmread(dcm_path)
    image = dcm.pixel_array

    # Normalize
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255.0
    image = image.astype(np.uint8)

    # RGB
    image = np.stack([image] * 3, axis=-1)
    image_pil = Image.fromarray(image)

    # Transform
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform(image_pil).unsqueeze(0), image


@torch.no_grad()
def predict(model, image_tensor, device, score_threshold=0.5):
    """Run inference on a single image."""
    model.eval()
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)

    output = outputs[0]
    scores = output['scores'].cpu().numpy()
    boxes = output['boxes'].cpu().numpy()
    labels = output['labels'].cpu().numpy()

    mask = scores >= score_threshold
    return boxes[mask], scores[mask], labels[mask]


def visualize_prediction(image, boxes, scores, labels, save_path=None, scale=1.0, style='red'):
    """Draw predictions on image.

    Args:
        image: Image array (grayscale or RGB).
        boxes: Predicted bounding boxes.
        scores: Confidence scores.
        labels: Class labels.
        save_path: Where to save the figure.
        scale: Scale factor for boxes.
        style: 'red' (default) or 'purple' (sample image style).

    Returns:
        fig: Matplotlib figure.
    """
    if style == 'purple':
        # Use the purple-style visualization from visualize.py
        label_texts = ['pneumonia'] * len(boxes)
        fig, ax = visualize_purple_style(
            image=image,
            boxes=boxes.tolist() if hasattr(boxes, 'tolist') else boxes,
            scores=scores.tolist() if hasattr(scores, 'tolist') else scores,
            labels=label_texts,
            save_path=save_path,
            scale=scale,
            show_axes=True,
            figsize=(10, 10),
            dpi=150,
        )
        plt.close(fig)
        return fig

    # Default red style
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image, cmap='gray')

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        x1 *= scale
        y1 *= scale
        x2 *= scale
        y2 *= scale

        width = x2 - x1
        height = y2 - y1

        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f'Pneumonia: {score:.3f}',
                color='red', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax.set_title(f'Detected {len(boxes)} region(s)')
    ax.axis('off')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.close()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Path to single DICOM image')
    parser.add_argument('--image_dir', type=str, help='Directory of DICOM images')
    parser.add_argument('--checkpoint', type=str, default=config.MODEL_SAVE_PATH)
    parser.add_argument('--output', type=str, default='output/demo_results')
    parser.add_argument('--score_threshold', type=float, default=0.5)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--style', type=str, default='red',
                        choices=['red', 'purple'],
                        help='Visualization style: red (default) or purple (sample image style)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Please train the model first: python main.py --mode train")
        return

    model = get_model(num_classes=2, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Loaded model from {args.checkpoint}")

    # Process single image
    if args.image:
        print(f"\nProcessing: {args.image}")
        image_tensor, original_image = load_dicom(args.image, args.image_size)
        boxes, scores, labels = predict(model, image_tensor, device, args.score_threshold)

        scale = original_image.shape[0] / args.image_size
        save_path = os.path.join(args.output, 'prediction.png')
        visualize_prediction(original_image, boxes, scores, labels, save_path, scale, style=args.style)

        print(f"\nPredictions:")
        for i, (box, score) in enumerate(zip(boxes, scores)):
            print(f"  Box {i+1}: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}] | Score: {score:.4f}")

    # Process directory
    elif args.image_dir:
        dcm_files = sorted([f for f in os.listdir(args.image_dir) if f.endswith('.dcm')])
        print(f"\nProcessing {len(dcm_files)} images...")

        for dcm_file in dcm_files:
            dcm_path = os.path.join(args.image_dir, dcm_file)
            image_tensor, original_image = load_dicom(dcm_path, args.image_size)
            boxes, scores, labels = predict(model, image_tensor, device, args.score_threshold)

            scale = original_image.shape[0] / args.image_size
            save_path = os.path.join(args.output, f'{dcm_file.replace(".dcm", ".png")}')
            visualize_prediction(original_image, boxes, scores, labels, save_path, scale, style=args.style)

        print(f"\nSaved all results to {args.output}")


if __name__ == '__main__':
    main()
