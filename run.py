"""Simple runner script for the most common workflows.

This provides a simplified interface for users who don't want
to use command-line arguments.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from main import set_seed, get_device, train, evaluate, compare, visualize


def run_training():
    """Run full training with recommended settings."""
    print("="*60)
    print("Running: Full Training with Improvements")
    print("="*60)

    class Args:
        mode = 'train'
        epochs = 20
        batch_size = 8
        lr = 1e-4
        weight_decay = 1e-4
        image_size = 512
        num_workers = 4
        model_type = 'fasterrcnn'
        pretrained = True
        no_pretrained = False
        augmentation = True
        no_augmentation = False
        freeze_backbone = True
        freeze_layers = 2
        optimizer = 'adamw'
        scheduler = 'plateau'
        early_stopping = True
        early_stopping_patience = 5
        checkpoint = config.MODEL_SAVE_PATH
        num_viz = 8

    args = Args()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    train(args)


def run_evaluation():
    """Run evaluation on the best saved model."""
    print("="*60)
    print("Running: Model Evaluation")
    print("="*60)

    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Model not found at {config.MODEL_SAVE_PATH}")
        print("Please run training first.")
        return

    class Args:
        mode = 'evaluate'
        batch_size = 8
        image_size = 512
        num_workers = 4
        model_type = 'fasterrcnn'
        checkpoint = config.MODEL_SAVE_PATH
        num_viz = 8
        score_threshold = 0.5

    args = Args()
    evaluate(args)


def run_comparison():
    """Run baseline vs improved comparison."""
    print("="*60)
    print("Running: Baseline vs Improved Comparison")
    print("="*60)

    class Args:
        mode = 'compare'
        epochs = 15
        batch_size = 8
        image_size = 512
        num_workers = 4
        model_type = 'fasterrcnn'
        num_viz = 8

    args = Args()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    compare(args)


def run_visualization():
    """Run visualization on saved model."""
    print("="*60)
    print("Running: Prediction Visualization")
    print("="*60)

    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Model not found at {config.MODEL_SAVE_PATH}")
        print("Please run training first.")
        return

    class Args:
        mode = 'visualize'
        batch_size = 8
        image_size = 512
        num_workers = 4
        model_type = 'fasterrcnn'
        checkpoint = config.MODEL_SAVE_PATH
        num_viz = 12
        score_threshold = 0.5

    args = Args()
    visualize(args)


def print_menu():
    """Print interactive menu."""
    print("\n" + "="*60)
    print("RSNA Pneumonia Detection - Quick Runner")
    print("="*60)
    print("1. Train model (with all improvements)")
    print("2. Evaluate saved model")
    print("3. Compare baseline vs improved")
    print("4. Visualize predictions")
    print("5. Exit")
    print("="*60)


def main():
    """Interactive menu for running the pipeline."""
    while True:
        print_menu()
        choice = input("Enter your choice (1-5): ").strip()

        if choice == '1':
            run_training()
        elif choice == '2':
            run_evaluation()
        elif choice == '3':
            run_comparison()
        elif choice == '4':
            run_visualization()
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1-5.")


if __name__ == '__main__':
    # Allow direct execution: python run.py [train|evaluate|compare|visualize]
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == 'train':
            run_training()
        elif cmd == 'evaluate':
            run_evaluation()
        elif cmd == 'compare':
            run_comparison()
        elif cmd == 'visualize':
            run_visualization()
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python run.py [train|evaluate|compare|visualize]")
    else:
        main()
