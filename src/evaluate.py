"""
Evaluation script for CNN-LSTM microstructure evolution prediction.
Loads trained model and evaluates on validation set with detailed metrics and visualizations.
"""

import torch
import torch.nn as nn
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from .config import Config
from .model import get_model
from .dataset import create_dataloaders, MicrostructureDataset, create_sample_pairs
from .utils import (
    get_device, load_checkpoint, calculate_metrics, print_metrics,
    visualize_sample, denormalize_image
)


class Evaluator:
    """Evaluator class for comprehensive model evaluation."""

    def __init__(
        self,
        model: nn.Module,
        val_loader,
        config: Config,
        device: torch.device
    ):
        self.model = model
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.criterion = nn.MSELoss()

    @torch.no_grad()
    def evaluate(self, save_visualizations: bool = True) -> dict:
        """
        Comprehensive evaluation on validation set.

        Args:
            save_visualizations: Whether to save visualizations

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        all_losses = []
        all_metrics = {
            "mse": [], "mae": [], "psnr": [],
            "ipfx_mse": [], "ipfy_mse": [], "ipfz_mse": [], "oriindx_mse": []
        }

        print("\nEvaluating model...")
        pbar = tqdm(self.val_loader, desc="Evaluation")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            all_losses.append(loss.item())

            # Calculate metrics
            batch_metrics = calculate_metrics(outputs, targets)
            for key in all_metrics:
                all_metrics[key].append(batch_metrics[key])

            # Save visualizations for first few batches
            if save_visualizations and batch_idx < 5:
                for sample_idx in range(min(2, inputs.size(0))):
                    vis_path = self.config.OUTPUT_DIR / f"eval_batch_{batch_idx}_sample_{sample_idx}.png"
                    visualize_sample(
                        inputs,
                        outputs,
                        targets,
                        save_path=vis_path,
                        sample_idx=sample_idx
                    )

        # Calculate average metrics
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        avg_metrics['loss'] = np.mean(all_losses)

        # Calculate standard deviations
        std_metrics = {f"{key}_std": np.std(values) for key, values in all_metrics.items()}
        avg_metrics.update(std_metrics)

        return avg_metrics

    @torch.no_grad()
    def evaluate_per_component(self) -> dict:
        """
        Evaluate each output component separately.

        Returns:
            Dictionary with per-component metrics
        """
        self.model.eval()

        component_metrics = {
            "ipfx": {"mse": [], "mae": []},
            "ipfy": {"mse": [], "mae": []},
            "ipfz": {"mse": [], "mae": []},
            "oriindx": {"mse": [], "mae": []}
        }

        print("\nEvaluating per component...")
        pbar = tqdm(self.val_loader, desc="Component Evaluation")

        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)

            # Split into components (each has 3 channels)
            components = {
                "ipfx": (outputs[:, 0:3], targets[:, 0:3]),
                "ipfy": (outputs[:, 3:6], targets[:, 3:6]),
                "ipfz": (outputs[:, 6:9], targets[:, 6:9]),
                "oriindx": (outputs[:, 9:12], targets[:, 9:12])
            }

            for comp_name, (pred, tgt) in components.items():
                mse = torch.mean((pred - tgt) ** 2).item()
                mae = torch.mean(torch.abs(pred - tgt)).item()
                component_metrics[comp_name]["mse"].append(mse)
                component_metrics[comp_name]["mae"].append(mae)

        # Average per component
        avg_component_metrics = {}
        for comp_name, metrics in component_metrics.items():
            avg_component_metrics[f"{comp_name}_mse"] = np.mean(metrics["mse"])
            avg_component_metrics[f"{comp_name}_mae"] = np.mean(metrics["mae"])
            avg_component_metrics[f"{comp_name}_mse_std"] = np.std(metrics["mse"])
            avg_component_metrics[f"{comp_name}_mae_std"] = np.std(metrics["mae"])

        return avg_component_metrics

    def create_comparison_grid(self, num_samples: int = 4):
        """
        Create a grid comparing predictions with targets.

        Args:
            num_samples: Number of samples to include in grid
        """
        self.model.eval()

        samples_collected = 0
        all_inputs = []
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)

                all_inputs.append(inputs.cpu())
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())

                samples_collected += inputs.size(0)
                if samples_collected >= num_samples:
                    break

        # Concatenate
        all_inputs = torch.cat(all_inputs, dim=0)[:num_samples]
        all_outputs = torch.cat(all_outputs, dim=0)[:num_samples]
        all_targets = torch.cat(all_targets, dim=0)[:num_samples]

        # Create grid visualization
        fig, axes = plt.subplots(num_samples, 6, figsize=(18, 3 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        component_names = ["IPF X", "IPF Y", "IPF Z", "Orientation Index"]
        channel_ranges = [(0, 3), (3, 6), (6, 9), (9, 12)]

        for sample_idx in range(num_samples):
            pred = all_outputs[sample_idx]
            tgt = all_targets[sample_idx]

            for comp_idx, (name, (start, end)) in enumerate(zip(component_names, channel_ranges)):
                # Prediction
                pred_comp = denormalize_image(pred[start:end])
                axes[sample_idx, comp_idx].imshow(pred_comp)
                if sample_idx == 0:
                    axes[sample_idx, comp_idx].set_title(f"{name}\n(Prediction)")
                axes[sample_idx, comp_idx].axis('off')

                # Target
                if comp_idx == 0:  # Show target only for first component to save space
                    tgt_comp = denormalize_image(tgt[start:end])
                    axes[sample_idx, 4].imshow(tgt_comp)
                    if sample_idx == 0:
                        axes[sample_idx, 4].set_title(f"{component_names[0]}\n(Target)")
                    axes[sample_idx, 4].axis('off')

            # Difference map for IPF X
            pred_ipfx = pred[0:3].numpy().transpose(1, 2, 0)
            tgt_ipfx = tgt[0:3].numpy().transpose(1, 2, 0)
            diff = np.abs(pred_ipfx - tgt_ipfx).mean(axis=2)

            im = axes[sample_idx, 5].imshow(diff, cmap='hot', vmin=0, vmax=0.2)
            if sample_idx == 0:
                axes[sample_idx, 5].set_title("Difference\n(IPF X)")
            axes[sample_idx, 5].axis('off')

        plt.tight_layout()
        save_path = self.config.OUTPUT_DIR / "comparison_grid.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nComparison grid saved to {save_path}")
        plt.close()

    def plot_metric_distributions(self, metrics: dict):
        """
        Plot distributions of metrics.

        Args:
            metrics: Dictionary of metrics
        """
        # Extract MSE metrics
        mse_metrics = {
            "Overall": metrics["mse"],
            "IPF X": metrics["ipfx_mse"],
            "IPF Y": metrics["ipfy_mse"],
            "IPF Z": metrics["ipfz_mse"],
            "Orientation": metrics["oriindx_mse"]
        }

        fig, ax = plt.subplots(figsize=(10, 6))

        names = list(mse_metrics.keys())
        values = list(mse_metrics.values())

        bars = ax.bar(names, values, color=['blue', 'green', 'red', 'orange', 'purple'])
        ax.set_ylabel('MSE', fontsize=12)
        ax.set_title('MSE by Component', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.6f}',
                   ha='center', va='bottom', fontsize=10)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        save_path = self.config.OUTPUT_DIR / "metric_distributions.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metric distributions saved to {save_path}")
        plt.close()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained microstructure evolution model")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--model_type', type=str, default='cnn_lstm',
                       choices=['cnn_lstm', 'conv_lstm'], help='Model architecture type')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'mps', 'cpu'], help='Device to use')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Directory to save evaluation results')

    args = parser.parse_args()

    # Load configuration
    config = Config()
    config.DATA_DIR = Path(args.data_dir)
    config.OUTPUT_DIR = Path(args.output_dir)
    config.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size

    print("=" * 80)
    print("Model Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data directory: {config.DATA_DIR}")
    print(f"Output directory: {config.OUTPUT_DIR}")

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    _, val_loader = create_dataloaders(config)

    # Create model
    print(f"\nCreating {args.model_type} model...")
    model = get_model(config, args.model_type)
    model = model.to(device)

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = load_checkpoint(Path(args.checkpoint), model, device=device)

    # Create evaluator
    evaluator = Evaluator(model, val_loader, config, device)

    # Run evaluation
    print("\n" + "=" * 80)
    print("Running evaluation...")
    print("=" * 80)

    metrics = evaluator.evaluate(save_visualizations=True)
    print_metrics(metrics, "Overall")

    # Per-component evaluation
    component_metrics = evaluator.evaluate_per_component()
    print_metrics(component_metrics, "Per-Component")

    # Create visualizations
    print("\nCreating visualizations...")
    evaluator.create_comparison_grid(num_samples=4)
    evaluator.plot_metric_distributions(metrics)

    # Save metrics to file
    metrics_file = config.OUTPUT_DIR / "evaluation_metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Evaluation Metrics\n")
        f.write("=" * 80 + "\n\n")
        f.write("Overall Metrics:\n")
        for key, value in metrics.items():
            f.write(f"{key:20s}: {value:.6f}\n")
        f.write("\n" + "-" * 80 + "\n\n")
        f.write("Per-Component Metrics:\n")
        for key, value in component_metrics.items():
            f.write(f"{key:20s}: {value:.6f}\n")

    print(f"\nMetrics saved to {metrics_file}")

    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
