"""Training script for RSNA Pneumonia Detection.

Handles training loop, loss tracking, optimizer setup, and learning rate scheduling.
Implements improvements: transfer learning, hyperparameter tuning support.

GPU Optimizations:
- Mixed Precision Training (AMP) for faster GPU training and lower memory
- Multi-GPU support via DataParallel / DistributedDataParallel
- Non-blocking GPU transfers (pin_memory + non_blocking=True)
- Gradient accumulation for larger effective batch sizes
- CUDNN benchmark optimization
- Persistent DataLoader workers
- Automatic batch size adjustment based on GPU memory
"""

import os
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

import config


def get_optimizer(model, lr=1e-4, weight_decay=1e-4, optimizer_type='adamw'):
    """Create optimizer with hyperparameter options.

    Args:
        model: Model to optimize.
        lr: Learning rate.
        weight_decay: Weight decay.
        optimizer_type: 'adamw', 'sgd', or 'adam'.

    Returns:
        Optimizer instance.
    """
    params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_type == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def get_scheduler(optimizer, scheduler_type='plateau', patience=3, factor=0.5):
    """Create learning rate scheduler.

    Args:
        optimizer: Optimizer to wrap.
        scheduler_type: 'plateau', 'step', or 'none'.
        patience: Epochs to wait before reducing LR.
        factor: LR reduction factor.

    Returns:
        Scheduler or None.
    """
    if scheduler_type == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='max', factor=factor,
                                  patience=patience, verbose=True)
    elif scheduler_type == 'step':
        return StepLR(optimizer, step_size=5, gamma=0.5)
    else:
        return None


def setup_multi_gpu(model):
    """Wrap model for multi-GPU training if available.

    Args:
        model: PyTorch model.

    Returns:
        Model (possibly wrapped in DataParallel).
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
    return model, num_gpus


def get_gpu_memory_info():
    """Get GPU memory usage info."""
    if not torch.cuda.is_available():
        return "No GPU available"

    info = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        info.append(f"GPU {i}: {allocated:.2f}GB / {total:.2f}GB allocated")
    return " | ".join(info)


def train_one_epoch(model, data_loader, optimizer, device, epoch,
                    clip_grad=1.0, use_amp=True, grad_accum_steps=1,
                    non_blocking=True):
    """Train for one epoch with GPU optimizations.

    Args:
        model: Detection model.
        data_loader: Training data loader.
        optimizer: Optimizer.
        device: Device to train on.
        epoch: Current epoch number.
        clip_grad: Gradient clipping value.
        use_amp: Use Automatic Mixed Precision (faster on modern GPUs).
        grad_accum_steps: Gradient accumulation steps (effective batch = batch * steps).
        non_blocking: Use non-blocking GPU transfers.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(data_loader)

    # Initialize AMP scaler if using GPU and AMP enabled
    scaler = GradScaler() if use_amp and torch.cuda.is_available() else None

    pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (images, targets) in enumerate(pbar):
        # Non-blocking GPU transfer (faster data movement)
        images = [img.to(device, non_blocking=non_blocking) for img in images]
        targets = [{k: v.to(device, non_blocking=non_blocking) for k, v in t.items()} for t in targets]

        # Forward pass with optional AMP
        if use_amp and scaler is not None:
            with autocast(device_type='cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                # Scale loss for gradient accumulation
                losses = losses / grad_accum_steps
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses = losses / grad_accum_steps

        # Backward pass
        if use_amp and scaler is not None:
            scaler.scale(losses).backward()

            # Gradient accumulation: only step every N batches
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
                # Gradient clipping with AMP
                if clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # More memory efficient than zero_grad()
        else:
            losses.backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        loss_val = losses.item() * grad_accum_steps
        total_loss += loss_val

        # Update progress bar with GPU memory info
        mem_info = get_gpu_memory_info() if torch.cuda.is_available() else "CPU"
        pbar.set_postfix({
            'loss': f"{loss_val:.4f}",
            'rpn_loss': f"{loss_dict.get('loss_rpn_box_reg', 0).item():.4f}",
            'cls_loss': f"{loss_dict.get('loss_classifier', 0).item():.4f}",
            'box_loss': f"{loss_dict.get('loss_box_reg', 0).item():.4f}",
            'gpu': mem_info.split(' | ')[0] if ' | ' in mem_info else mem_info[:30]
        })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


@torch.no_grad()
def evaluate_loss(model, data_loader, device, use_amp=True, non_blocking=True):
    """Evaluate model loss on validation set with GPU optimizations.

    Args:
        model: Detection model.
        data_loader: Validation data loader.
        device: Device.
        use_amp: Use Automatic Mixed Precision for evaluation.
        non_blocking: Use non-blocking GPU transfers.

    Returns:
        Average validation loss.
    """
    model.train()  # Need train mode to get losses from torchvision detection models
    total_loss = 0.0
    num_batches = len(data_loader)

    pbar = tqdm(data_loader, desc="[Val Loss]")
    for images, targets in pbar:
        images = [img.to(device, non_blocking=non_blocking) for img in images]
        targets = [{k: v.to(device, non_blocking=non_blocking) for k, v in t.items()} for t in targets]

        if use_amp and torch.cuda.is_available():
            with autocast(device_type='cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        total_loss += losses.item()

        pbar.set_postfix({'loss': f"{losses.item():.4f}"})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def train_model(model, train_loader, val_loader, device,
                num_epochs=20, lr=1e-4, weight_decay=1e-4,
                optimizer_type='adamw', scheduler_type='plateau',
                checkpoint_dir='output/checkpoints', model_save_path='output/best_model.pth',
                early_stopping_patience=5, use_early_stopping=True,
                use_amp=True, grad_accum_steps=1, resume_path=None):
    """Full training loop with checkpointing, early stopping, and GPU optimizations.

    Args:
        model: Detection model.
        train_loader: Training data.
        val_loader: Validation data.
        device: Device.
        num_epochs: Total epochs.
        lr: Learning rate.
        weight_decay: Weight decay.
        optimizer_type: Optimizer choice.
        scheduler_type: LR scheduler choice.
        checkpoint_dir: Where to save checkpoints.
        model_save_path: Best model save path.
        early_stopping_patience: Epochs without improvement before stopping.
        use_early_stopping: Whether to use early stopping.
        use_amp: Enable Automatic Mixed Precision training.
        grad_accum_steps: Gradient accumulation steps.
        resume_path: Path to checkpoint to resume from.

    Returns:
        Dictionary with training history.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = get_optimizer(model, lr, weight_decay, optimizer_type)
    scheduler = get_scheduler(optimizer, scheduler_type)

    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'best_epoch': 0,
        'best_val_loss': float('inf'),
        'epoch_times': [],
    }

    best_val_loss = float('inf')
    patience_counter = 0
    start_epoch = 1

    # Resume from checkpoint if provided
    if resume_path is not None and os.path.exists(resume_path):
        print(f"\n[RESUME] Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)

        # Load model state (handle DataParallel wrapper)
        base_model = model.module if hasattr(model, 'module') else model
        base_model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore training state
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        patience_counter = 0  # Reset patience on resume

        print(f"[RESUME] Resuming from Epoch {start_epoch}")
        print(f"[RESUME] Previous best val_loss: {best_val_loss:.4f}")
    elif resume_path is not None:
        print(f"[RESUME] Warning: Checkpoint not found at {resume_path}, starting from scratch")

    # GPU info
    gpu_info = ""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        num_gpus = torch.cuda.device_count()
        gpu_info = f"{num_gpus}x {gpu_name} ({gpu_mem:.1f}GB each)"
        if use_amp:
            gpu_info += " | AMP enabled"
        if grad_accum_steps > 1:
            gpu_info += f" | GradAccum={grad_accum_steps}"
    else:
        gpu_info = "CPU"

    print(f"\n{'='*70}")
    print(f"Training Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  LR: {lr}, Weight Decay: {weight_decay}")
    print(f"  Optimizer: {optimizer_type}")
    print(f"  Scheduler: {scheduler_type}")
    print(f"  Device: {gpu_info}")
    print(f"  Effective Batch Size: {train_loader.batch_size * grad_accum_steps}")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            clip_grad=1.0, use_amp=use_amp, grad_accum_steps=grad_accum_steps
        )

        # Validate
        val_loss = evaluate_loss(model, val_loader, device, use_amp=use_amp)

        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Scheduler step
        if scheduler_type == 'plateau':
            scheduler.step(-val_loss)
        elif scheduler_type == 'step':
            scheduler.step()

        epoch_time = time.time() - epoch_start
        history['epoch_times'].append(epoch_time)

        # Record
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)

        # GPU memory summary
        mem_summary = get_gpu_memory_info() if torch.cuda.is_available() else "CPU"

        print(f"\nEpoch {epoch}/{num_epochs} | Time: {epoch_time:.1f}s | {mem_summary}")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")

        # Save best model (handle DataParallel wrapper)
        model_to_save = model.module if hasattr(model, 'module') else model

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history['best_epoch'] = epoch
            history['best_val_loss'] = best_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, model_save_path)
            print(f"  -> Saved best model (val_loss: {val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Checkpoint every epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss,
        }, checkpoint_path)

        # Early stopping
        if use_early_stopping and patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {early_stopping_patience} epochs without improvement")
            break

    # Final memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"  Best Epoch: {history['best_epoch']}")
    print(f"  Best Val Loss: {history['best_val_loss']:.4f}")
    print(f"  Avg Epoch Time: {np.mean(history['epoch_times']):.1f}s")
    print(f"{'='*70}")

    return history
