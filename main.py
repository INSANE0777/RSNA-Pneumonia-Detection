"""Main pipeline for RSNA Pneumonia Detection.

Orchestrates data preparation, model training, evaluation, and visualization.
Supports running baseline vs improved model comparison.

GPU Optimizations:
- CUDNN benchmark for fixed-size inputs (faster convolutions)
- Automatic Mixed Precision (AMP) training
- Multi-GPU support via DataParallel
- Non-blocking data transfers
- GPU memory management (empty_cache)

Usage:
    python main.py --mode train --epochs 20 --augmentation
    python main.py --mode evaluate --checkpoint output/best_model.pth
    python main.py --mode compare  # Baseline vs Improved
    python main.py --mode visualize --checkpoint output/best_model.pth
"""

import os
import sys
import argparse
import json
import torch
import warnings

warnings.filterwarnings('ignore')

import config
from data_preparation import prepare_data, prepare_test_data
from model import get_model, freeze_backbone, unfreeze_all
from train import train_model, setup_multi_gpu
from evaluate import evaluate_model, evaluate_with_multiple_thresholds, print_metrics
from visualize import (visualize_predictions, plot_training_history,
                       plot_pr_curve, plot_iou_distribution, compare_before_after)


def set_seed(seed=42, benchmark=True):
    """Set random seed for reproducibility and GPU optimizations.

    Args:
        seed: Random seed.
        benchmark: Enable CUDNN benchmark for faster training (when input sizes are fixed).
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # benchmark=True optimizes for fixed input sizes (faster)
        # benchmark=False is needed for variable input sizes (more reproducible)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = not benchmark


def get_device():
    """Get the best available device with info."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        num_gpus = torch.cuda.device_count()
        print(f"Using GPU: {gpu_name} ({gpu_mem:.1f}GB)")
        if num_gpus > 1:
            print(f"  {num_gpus} GPUs detected - multi-GPU support available")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  PyTorch CUDA: {torch.backends.cudnn.version()}")
        print(f"  AMP (Mixed Precision): Available")
    else:
        device = torch.device('cpu')
        print("Using CPU (GPU not available)")
    return device


def train(args):
    """Run training pipeline with GPU optimizations."""
    print("\n" + "="*70)
    print("RSNA PNEUMONIA DETECTION - TRAINING")
    print("="*70)

    set_seed(config.RANDOM_SEED, benchmark=True)
    device = get_device()

    # Prepare data
    print("\n[1/4] Preparing data...")
    train_loader, val_loader, train_df, val_df = prepare_data(
        csv_path=config.TRAIN_LABELS,
        train_dir=config.TRAIN_DIR,
        batch_size=args.batch_size,
        split_ratio=config.TRAIN_VAL_SPLIT,
        use_augmentation=args.augmentation,
        image_size=args.image_size,
        num_workers=args.num_workers,
        random_seed=config.RANDOM_SEED,
        persistent_workers=config.PERSISTENT_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR,
        max_samples=args.max_samples
    )

    # Create model
    print("\n[2/4] Creating model...")
    model = get_model(
        num_classes=2,
        pretrained=args.pretrained,
        model_type=args.model_type
    )

    # Transfer learning: freeze backbone initially
    if args.freeze_backbone:
        freeze_backbone(model, layers_to_freeze=args.freeze_layers)

    # Setup multi-GPU if available
    model, num_gpus = setup_multi_gpu(model)
    model.to(device)

    # Note: torch.compile() requires Triton (Linux-only), skipped on Windows

    # Count parameters
    # Access underlying model if wrapped in DataParallel
    base_model = model.module if hasattr(model, 'module') else model
    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Train
    print("\n[3/4] Training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer,
        scheduler_type=args.scheduler,
        checkpoint_dir=config.CHECKPOINT_DIR,
        model_save_path=config.MODEL_SAVE_PATH,
        early_stopping_patience=args.early_stopping_patience,
        use_early_stopping=args.early_stopping,
        use_amp=config.USE_AMP,
        grad_accum_steps=args.grad_accum_steps,
        resume_path=args.resume
    )

    # Save history
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUT_DIR, 'history.json'), 'w') as f:
        json.dump({k: v for k, v in history.items() if k not in ['ious', 'precisions', 'recalls']}, f, indent=2)

    # Plot training curves
    plot_training_history(history, save_path=os.path.join(config.OUTPUT_DIR, 'training_history.png'))

    # Evaluate
    print("\n[4/4] Evaluating...")
    # Load best model for evaluation
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    model = base_model.to(device)

    metrics = evaluate_with_multiple_thresholds(
        model, val_loader, device,
        iou_thresholds=[0.3, 0.5, 0.75],
        score_threshold=config.SCORE_THRESHOLD,
        use_amp=config.USE_AMP
    )

    # Save metrics
    with open(os.path.join(config.OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump({k: v for k, v in metrics.items() if isinstance(v, (int, float, list))}, f, indent=2)

    # Visualize
    visualize_predictions(
        model, val_loader, device,
        num_samples=args.num_viz,
        save_dir=os.path.join(config.OUTPUT_DIR, 'visualizations'),
        score_threshold=config.SCORE_THRESHOLD
    )

    print("\nTraining complete! Check output/ for results.")
    return model, history, metrics


def evaluate(args):
    """Run evaluation on a saved model with GPU optimizations."""
    print("\n" + "="*70)
    print("RSNA PNEUMONIA DETECTION - EVALUATION")
    print("="*70)

    device = get_device()

    # Load data
    print("\nLoading data...")
    _, val_loader, _, _ = prepare_data(
        csv_path=config.TRAIN_LABELS,
        train_dir=config.TRAIN_DIR,
        batch_size=args.batch_size,
        split_ratio=config.TRAIN_VAL_SPLIT,
        use_augmentation=False,
        image_size=args.image_size,
        num_workers=args.num_workers,
        random_seed=config.RANDOM_SEED,
        persistent_workers=config.PERSISTENT_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR,
        max_samples=args.max_samples
    )

    # Load model
    print("Loading model...")
    model = get_model(num_classes=2, pretrained=False, model_type=args.model_type)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate_with_multiple_thresholds(
        model, val_loader, device,
        iou_thresholds=[0.3, 0.5, 0.75],
        score_threshold=config.SCORE_THRESHOLD,
        use_amp=config.USE_AMP
    )

    # Visualize
    visualize_predictions(
        model, val_loader, device,
        num_samples=args.num_viz,
        save_dir=os.path.join(config.OUTPUT_DIR, 'visualizations'),
        score_threshold=config.SCORE_THRESHOLD
    )

    return metrics


def compare(args):
    """Compare baseline vs improved model with GPU optimizations."""
    print("\n" + "="*70)
    print("RSNA PNEUMONIA DETECTION - BASELINE VS IMPROVED COMPARISON")
    print("="*70)

    device = get_device()

    # Prepare data (same split for fair comparison)
    set_seed(config.RANDOM_SEED, benchmark=True)
    train_loader, val_loader, _, _ = prepare_data(
        csv_path=config.TRAIN_LABELS,
        train_dir=config.TRAIN_DIR,
        batch_size=args.batch_size,
        split_ratio=config.TRAIN_VAL_SPLIT,
        use_augmentation=False,  # Baseline: no augmentation
        image_size=args.image_size,
        num_workers=args.num_workers,
        random_seed=config.RANDOM_SEED,
        persistent_workers=config.PERSISTENT_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR,
        max_samples=args.max_samples
    )

    # --- BASELINE MODEL (no improvements) ---
    print("\n" + "="*70)
    print("TRAINING BASELINE MODEL")
    print("  - No data augmentation")
    print("  - No transfer learning (random init)")
    print("  - Default hyperparameters")
    print("="*70)

    baseline_model = get_model(num_classes=2, pretrained=False, model_type='fasterrcnn')
    baseline_model.to(device)

    baseline_history = train_model(
        model=baseline_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        lr=1e-3,
        weight_decay=1e-4,
        optimizer_type='sgd',
        scheduler_type='none',
        checkpoint_dir=os.path.join(config.OUTPUT_DIR, 'baseline_checkpoints'),
        model_save_path=os.path.join(config.OUTPUT_DIR, 'baseline_model.pth'),
        use_early_stopping=False,
        use_amp=config.USE_AMP
    )

    # Evaluate baseline
    baseline_model.load_state_dict(
        torch.load(os.path.join(config.OUTPUT_DIR, 'baseline_model.pth'), map_location=device)['model_state_dict']
    )
    baseline_metrics = evaluate_model(
        baseline_model, val_loader, device,
        iou_threshold=0.5, score_threshold=config.SCORE_THRESHOLD,
        use_amp=config.USE_AMP
    )
    print_metrics(baseline_metrics, iou_threshold=0.5)

    # --- IMPROVED MODEL (with improvements) ---
    print("\n" + "="*70)
    print("TRAINING IMPROVED MODEL")
    print("  - Data augmentation (flip, jitter, affine)")
    print("  - Transfer learning (pretrained ResNet50)")
    print("  - AdamW optimizer + LR scheduling")
    print("  - Gradient clipping + early stopping")
    print("  - AMP mixed precision training")
    print("="*70)

    # Recreate loaders with augmentation for improved model
    train_loader_aug, val_loader_aug, _, _ = prepare_data(
        csv_path=config.TRAIN_LABELS,
        train_dir=config.TRAIN_DIR,
        batch_size=args.batch_size,
        split_ratio=config.TRAIN_VAL_SPLIT,
        use_augmentation=True,  # Improvement 1: Data augmentation
        image_size=args.image_size,
        num_workers=args.num_workers,
        random_seed=config.RANDOM_SEED,
        persistent_workers=config.PERSISTENT_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR,
        max_samples=args.max_samples
    )

    improved_model = get_model(num_classes=2, pretrained=True, model_type='fasterrcnn')
    freeze_backbone(improved_model, layers_to_freeze=2)  # Improvement 2: Transfer learning
    improved_model.to(device)

    improved_history = train_model(
        model=improved_model,
        train_loader=train_loader_aug,
        val_loader=val_loader_aug,
        device=device,
        num_epochs=args.epochs,
        lr=1e-4,  # Improvement 3: Better LR
        weight_decay=1e-4,
        optimizer_type='adamw',  # Improvement 3: Better optimizer
        scheduler_type='plateau',  # Improvement 3: LR scheduling
        checkpoint_dir=config.CHECKPOINT_DIR,
        model_save_path=config.MODEL_SAVE_PATH,
        use_early_stopping=True,  # Improvement 3: Early stopping
        use_amp=config.USE_AMP
    )

    # Unfreeze all for fine-tuning in later epochs (optional)
    # unfreeze_all(improved_model)

    # Evaluate improved
    improved_model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=device)['model_state_dict']
    )
    improved_metrics = evaluate_model(
        improved_model, val_loader_aug, device,
        iou_threshold=0.5, score_threshold=config.SCORE_THRESHOLD,
        use_amp=config.USE_AMP
    )
    print_metrics(improved_metrics, iou_threshold=0.5)

    # Compare and visualize
    compare_before_after(baseline_metrics, improved_metrics,
                         save_path=os.path.join(config.OUTPUT_DIR, 'comparison.png'))

    # Plot training curves
    plot_training_history(baseline_history,
                          save_path=os.path.join(config.OUTPUT_DIR, 'baseline_history.png'))
    plot_training_history(improved_history,
                          save_path=os.path.join(config.OUTPUT_DIR, 'improved_history.png'))

    # Plot PR curves
    plot_pr_curve(baseline_metrics,
                  save_path=os.path.join(config.OUTPUT_DIR, 'baseline_pr_curve.png'))
    plot_pr_curve(improved_metrics,
                  save_path=os.path.join(config.OUTPUT_DIR, 'improved_pr_curve.png'))

    # Plot IoU distributions
    plot_iou_distribution(baseline_metrics.get('ious', []),
                          save_path=os.path.join(config.OUTPUT_DIR, 'baseline_iou_dist.png'))
    plot_iou_distribution(improved_metrics.get('ious', []),
                          save_path=os.path.join(config.OUTPUT_DIR, 'improved_iou_dist.png'))

    # Save comparison results
    results = {
        'baseline': {k: v for k, v in baseline_metrics.items() if isinstance(v, (int, float, list))},
        'improved': {k: v for k, v in improved_metrics.items() if isinstance(v, (int, float, list))},
    }
    with open(os.path.join(config.OUTPUT_DIR, 'comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\nComparison complete! Check output/ for all results.")
    return baseline_metrics, improved_metrics


def visualize(args):
    """Run visualization on saved model."""
    print("\n" + "="*70)
    print("RSNA PNEUMONIA DETECTION - VISUALIZATION")
    print("="*70)

    device = get_device()

    _, val_loader, _, _ = prepare_data(
        csv_path=config.TRAIN_LABELS,
        train_dir=config.TRAIN_DIR,
        batch_size=args.batch_size,
        split_ratio=config.TRAIN_VAL_SPLIT,
        use_augmentation=False,
        image_size=args.image_size,
        num_workers=args.num_workers,
        random_seed=config.RANDOM_SEED,
        persistent_workers=config.PERSISTENT_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR,
        max_samples=args.max_samples
    )

    model = get_model(num_classes=2, pretrained=False, model_type=args.model_type)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    visualize_predictions(
        model, val_loader, device,
        num_samples=args.num_viz,
        save_dir=os.path.join(config.OUTPUT_DIR, 'visualizations'),
        score_threshold=config.SCORE_THRESHOLD
    )


def main():
    parser = argparse.ArgumentParser(description='RSNA Pneumonia Detection')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'compare', 'visualize'],
                        help='Execution mode')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=config.WEIGHT_DECAY,
                        help='Weight decay')
    parser.add_argument('--image_size', type=int, default=config.IMAGE_SIZE,
                        help='Image size')
    parser.add_argument('--num_workers', type=int, default=config.NUM_WORKERS,
                        help='DataLoader workers')
    parser.add_argument('--model_type', type=str, default='fasterrcnn',
                        choices=['fasterrcnn', 'fasterrcnn_custom'],
                        help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Do not use pretrained weights')
    parser.add_argument('--augmentation', action='store_true', default=True,
                        help='Use data augmentation')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help='Freeze backbone layers')
    parser.add_argument('--freeze_layers', type=int, default=2,
                        help='Number of backbone layers to freeze')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'sgd', 'adam'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'step', 'none'],
                        help='LR scheduler')
    parser.add_argument('--early_stopping', action='store_true', default=True,
                        help='Use early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--checkpoint', type=str, default=config.MODEL_SAVE_PATH,
                        help='Checkpoint path for evaluation/visualization')
    parser.add_argument('--num_viz', type=int, default=8,
                        help='Number of samples to visualize')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps (effective batch = batch * steps)')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable Automatic Mixed Precision')
    parser.add_argument('--no_benchmark', action='store_true',
                        help='Disable CUDNN benchmark')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max patients to use (for quick testing). If set, randomly subsamples before splitting.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint path (e.g., output/checkpoints/checkpoint_epoch_3.pth)')

    args = parser.parse_args()

    if args.no_pretrained:
        args.pretrained = False
    if args.no_augmentation:
        args.augmentation = False
    if args.no_amp:
        config.USE_AMP = False
    if args.no_benchmark:
        config.CUDNN_BENCHMARK = False

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'compare':
        compare(args)
    elif args.mode == 'visualize':
        visualize(args)


if __name__ == '__main__':
    main()
