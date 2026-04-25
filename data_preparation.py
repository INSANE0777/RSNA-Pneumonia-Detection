"""Data preparation for RSNA Pneumonia Detection Challenge.

Handles loading DICOM images, parsing CSV annotations, preprocessing,
and creating train/validation splits.

GPU Optimizations:
- pin_memory=True for faster CPU->GPU transfers
- persistent_workers=True to keep workers alive between epochs
- prefetch_factor to preload samples ahead of time
- non_blocking transfers in training loop
"""

import os
import pydicom
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import cv2

import config


class RSNADataset(Dataset):
    """PyTorch Dataset for RSNA Pneumonia Detection.

    Loads DICOM chest X-rays and returns image tensors with target dicts
    compatible with torchvision detection models.

    Supports both torchvision transforms (val) and Albumentations (train)
    for bbox-aware augmentation.
    """

    def __init__(self, df, image_dir, transforms=None, albumentations_transform=None, image_size=512):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transforms = transforms
        self.albumentations_transform = albumentations_transform
        self.image_size = image_size
        self.original_size = 1024

        # Group by patientId to collect all boxes per image
        self.patient_ids = self.df['patientId'].unique()

        # Build lookup: patientId -> list of boxes
        self.boxes_dict = {}
        self.labels_dict = {}
        for pid in self.patient_ids:
            rows = self.df[self.df['patientId'] == pid]
            boxes = []
            labels = []
            for _, row in rows.iterrows():
                if pd.notna(row['x']) and pd.notna(row['y']) and pd.notna(row['width']) and pd.notna(row['height']):
                    if row['width'] > 0 and row['height'] > 0:
                        x1 = row['x']
                        y1 = row['y']
                        x2 = x1 + row['width']
                        y2 = y1 + row['height']
                        boxes.append([x1, y1, x2, y2])
                        labels.append(1)  # pneumonia = 1
            self.boxes_dict[pid] = boxes
            self.labels_dict[pid] = labels

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]

        # Load DICOM
        dcm_path = os.path.join(self.image_dir, f"{patient_id}.dcm")
        if not os.path.exists(dcm_path):
            # Fallback: create blank image if file missing
            image = np.zeros((self.original_size, self.original_size), dtype=np.uint16)
        else:
            dcm = pydicom.dcmread(dcm_path)
            image = dcm.pixel_array

        # Normalize to 0-255 and convert to uint8
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255.0
            image = image.astype(np.uint8)

        # Convert grayscale to RGB
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        # Get boxes and labels (in original image coordinates)
        boxes = self.boxes_dict[patient_id]
        labels = self.labels_dict[patient_id]

        # Apply Albumentations transforms (bbox-aware) if provided
        if self.albumentations_transform is not None:
            # Albumentations expects boxes in normalized or pixel coords with labels
            transformed = self.albumentations_transform(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']  # Already tensor (C, H, W) from ToTensorV2
            boxes = transformed['bboxes']
            labels = transformed['labels']

            # Build target dict
            if len(boxes) == 0:
                boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
                labels_tensor = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
                labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

            target = {
                'boxes': boxes_tensor,
                'labels': labels_tensor,
                'image_id': torch.tensor([idx]),
                'area': (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1]) if len(boxes) > 0 else torch.tensor([]),
                'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64),
            }
            return image, target

        # Fallback: torchvision transforms (val mode, no geometric aug)
        image_pil = Image.fromarray(image)

        # Scale factor
        scale = self.image_size / self.original_size

        # Apply transforms
        if self.transforms:
            image_tensor = self.transforms(image_pil)
        else:
            image_tensor = T.ToTensor()(image_pil)
            image_tensor = T.Resize((self.image_size, self.image_size))(image_tensor)

        # Scale boxes to resized image
        boxes_scaled = []
        for box in boxes:
            x1, y1, x2, y2 = box
            boxes_scaled.append([
                x1 * scale,
                y1 * scale,
                x2 * scale,
                y2 * scale
            ])

        # Build target dict
        if len(boxes_scaled) == 0:
            # No pneumonia: return empty targets with dummy box for compatibility
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.as_tensor(boxes_scaled, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([idx]),
            'area': (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1]) if len(boxes_scaled) > 0 else torch.tensor([]),
            'iscrowd': torch.zeros((len(boxes_scaled),), dtype=torch.int64),
        }

        return image_tensor, target


def get_transforms(train=True, use_augmentation=True, image_size=512):
    """Get torchvision transforms for training or validation.

    NOTE: For training with augmentation, use get_albumentations_transform()
    from augmentation.py instead. This function is kept for validation only
    and training without augmentation, since torchvision transforms do NOT
    properly update bounding box coordinates during geometric transforms.

    Args:
        train: Whether this is training mode.
        use_augmentation: Whether to apply data augmentation.
        image_size: Target image size.

    Returns:
        torchvision.transforms.Compose or None
    """
    transforms = [T.Resize((image_size, image_size))]

    # Only use torchvision augmentation for non-bbox transforms (color only)
    # Geometric transforms here do NOT update bounding boxes - use Albumentations instead
    if train and use_augmentation:
        # Color-only augmentations that don't affect bbox coordinates
        transforms.extend([
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ])

    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)


def collate_fn(batch):
    """Custom collate function for detection datasets."""
    return tuple(zip(*batch))


def load_annotations(csv_path):
    """Load and parse the RSNA labels CSV.

    Args:
        csv_path: Path to stage_2_train_labels.csv

    Returns:
        pandas.DataFrame with annotations.
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} annotation rows for {df['patientId'].nunique()} unique patients")
    print(f"Pneumonia cases: {df['Target'].sum()}")
    return df


def prepare_data(csv_path, train_dir, batch_size=8, split_ratio=0.8,
                 use_augmentation=True, image_size=512, num_workers=4,
                 random_seed=42, persistent_workers=True, prefetch_factor=2,
                 max_samples=None):
    """Prepare train and validation DataLoaders with GPU optimizations.

    Args:
        csv_path: Path to labels CSV.
        train_dir: Directory containing DICOM images.
        batch_size: Batch size.
        split_ratio: Fraction for training set.
        use_augmentation: Enable data augmentation on training set.
        image_size: Resize images to this size.
        num_workers: DataLoader workers.
        random_seed: Random seed for reproducibility.
        persistent_workers: Keep workers alive between epochs (faster).
        prefetch_factor: Number of batches to prefetch per worker.
        max_samples: If set, randomly sample this many patients (stratified) before splitting.

    Returns:
        (train_loader, val_loader, train_df, val_df)
    """
    df = load_annotations(csv_path)

    # Get unique patient IDs
    patient_ids = df['patientId'].unique()

    # Optional: subsample for faster training/testing
    if max_samples is not None and max_samples < len(patient_ids):
        print(f"Subsampling {max_samples} patients from {len(patient_ids)} total (stratified)")
        targets = [df[df['patientId'] == pid]['Target'].max() for pid in patient_ids]
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=random_seed)
        for keep_idx, _ in sss.split(patient_ids, targets):
            patient_ids = patient_ids[keep_idx]
        print(f"  -> Subsampled to {len(patient_ids)} patients")

    # Stratified split
    train_ids, val_ids = train_test_split(
        patient_ids, train_size=split_ratio, random_state=random_seed,
        stratify=[df[df['patientId'] == pid]['Target'].max() for pid in patient_ids]
    )

    train_df = df[df['patientId'].isin(train_ids)].copy()
    val_df = df[df['patientId'].isin(val_ids)].copy()

    print(f"Train: {len(train_ids)} patients, Val: {len(val_ids)} patients")

    # Create datasets
    # Use Albumentations for training (bbox-aware geometric augmentation)
    # Use torchvision for validation (no augmentation needed)
    if use_augmentation:
        from augmentation import get_train_augmentation, get_val_augmentation
        train_aug = get_train_augmentation(image_size=image_size, p=0.5)
        val_aug = get_val_augmentation(image_size=image_size)
        train_dataset = RSNADataset(
            train_df, train_dir,
            transforms=get_transforms(train=True, use_augmentation=False, image_size=image_size),
            albumentations_transform=train_aug,
            image_size=image_size
        )
        val_dataset = RSNADataset(
            val_df, train_dir,
            transforms=get_transforms(train=False, use_augmentation=False, image_size=image_size),
            albumentations_transform=val_aug,
            image_size=image_size
        )
    else:
        train_dataset = RSNADataset(
            train_df, train_dir,
            transforms=get_transforms(train=True, use_augmentation=False, image_size=image_size),
            image_size=image_size
        )
        val_dataset = RSNADataset(
            val_df, train_dir,
            transforms=get_transforms(train=False, use_augmentation=False, image_size=image_size),
            image_size=image_size
        )

    # Build DataLoader kwargs with GPU optimizations
    loader_kwargs = {
        'batch_size': batch_size,
        'collate_fn': collate_fn,
        'pin_memory': True,  # Faster CPU->GPU transfer
    }

    # persistent_workers and prefetch_factor require num_workers > 0
    if num_workers > 0:
        loader_kwargs['num_workers'] = num_workers
        if persistent_workers:
            loader_kwargs['persistent_workers'] = True
        if prefetch_factor and num_workers > 0:
            loader_kwargs['prefetch_factor'] = prefetch_factor

    # Create loaders
    train_loader = DataLoader(
        train_dataset, shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, **loader_kwargs
    )

    print(f"DataLoader config: batch_size={batch_size}, workers={num_workers}, "
          f"pin_memory=True, persistent={persistent_workers}, prefetch={prefetch_factor}")

    return train_loader, val_loader, train_df, val_df


def prepare_test_data(test_dir, batch_size=8, image_size=512, num_workers=4,
                      persistent_workers=True, prefetch_factor=2):
    """Prepare test DataLoader (no annotations) with GPU optimizations.

    Args:
        test_dir: Directory with test DICOM images.
        batch_size: Batch size.
        image_size: Resize images to this size.
        num_workers: DataLoader workers.
        persistent_workers: Keep workers alive between epochs.
        prefetch_factor: Number of batches to prefetch per worker.

    Returns:
        test_loader, patient_ids list
    """
    dcm_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.dcm')])
    patient_ids = [f.replace('.dcm', '') for f in dcm_files]

    # Create a dummy dataframe
    dummy_df = pd.DataFrame({
        'patientId': patient_ids,
        'x': [np.nan] * len(patient_ids),
        'y': [np.nan] * len(patient_ids),
        'width': [np.nan] * len(patient_ids),
        'height': [np.nan] * len(patient_ids),
        'Target': [0] * len(patient_ids)
    })

    # Use Albumentations val transform for test (resize + normalize only)
    from augmentation import get_val_augmentation
    val_aug = get_val_augmentation(image_size=image_size)
    test_dataset = RSNADataset(
        dummy_df, test_dir,
        transforms=get_transforms(train=False, use_augmentation=False, image_size=image_size),
        albumentations_transform=val_aug,
        image_size=image_size
    )

    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': False,
        'collate_fn': collate_fn,
        'pin_memory': True,
    }

    if num_workers > 0:
        loader_kwargs['num_workers'] = num_workers
        if persistent_workers:
            loader_kwargs['persistent_workers'] = True
        if prefetch_factor:
            loader_kwargs['prefetch_factor'] = prefetch_factor

    test_loader = DataLoader(test_dataset, **loader_kwargs)

    return test_loader, patient_ids
