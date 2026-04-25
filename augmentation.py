"""Advanced data augmentation for chest X-ray images.

Implements medical imaging specific augmentations using albumentations.
Used as part of the improvement techniques.
"""

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_train_augmentation(image_size=512, p=0.5):
    """Get training augmentation pipeline optimized for chest X-rays.

    Args:
        image_size: Target image size.
        p: Probability of applying each transform.

    Returns:
        Albumentations Compose object.
    """
    return A.Compose([
        # Geometric transforms
        A.HorizontalFlip(p=p),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            p=p
        ),

        # Intensity transforms (medical imaging appropriate)
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=p
        ),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),

        # Noise and blur (simulates image quality variations)
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(intensity=(0.1, 0.5), p=1.0),
        ], p=0.2),

        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.2),

        # Elastic deformation (simulates anatomical variations)
        A.ElasticTransform(
            alpha=1, sigma=50, alpha_affine=50,
            border_mode=cv2.BORDER_CONSTANT, p=0.2
        ),

        # Coarse dropout (simulates sensor artifacts)
        A.CoarseDropout(
            max_holes=8, max_height=32, max_width=32,
            min_holes=1, min_height=8, min_width=8,
            fill_value=0, p=0.1
        ),

        # Normalize and resize
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def get_val_augmentation(image_size=512):
    """Get validation augmentation (only resize and normalize).

    Args:
        image_size: Target image size.

    Returns:
        Albumentations Compose object.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def apply_augmentation(image, bboxes, labels, augmentation):
    """Apply augmentation to image and bounding boxes.

    Args:
        image: Numpy array (H, W, C) or (H, W).
        bboxes: List of boxes in [x1, y1, x2, y2] format.
        labels: List of labels.
        augmentation: Albumentations transform.

    Returns:
        Transformed image, bboxes, labels.
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255
        image = image.astype(np.uint8)

    # Ensure 3 channels
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)

    transformed = augmentation(image=image, bboxes=bboxes, labels=labels)
    return transformed['image'], transformed['bboxes'], transformed['labels']
