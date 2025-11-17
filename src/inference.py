"""
Inference script for CNN-LSTM microstructure evolution prediction.
Performs sliding window inference on full-resolution images and saves results as TIFF files.
"""

import torch
import torch.nn as nn
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from .config import Config
from .model import get_model
from .utils import get_device, load_checkpoint


class SlidingWindowInference:
    """Sliding window inference for full-resolution images."""

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: torch.device,
        patch_size: int = None,
        overlap: int = None
    ):
        """
        Args:
            model: Trained model
            config: Configuration object
            device: Device to run inference on
            patch_size: Size of patches (if None, use config.PATCH_SIZE)
            overlap: Overlap between patches (if None, use config.INFERENCE_OVERLAP)
        """
        self.model = model
        self.config = config
        self.device = device
        self.patch_size = patch_size or config.PATCH_SIZE
        self.overlap = overlap or config.INFERENCE_OVERLAP
        self.stride = self.patch_size - self.overlap

    def load_sample(self, sample_id: str) -> dict:
        """
        Load all images for a sample.

        Args:
            sample_id: Sample ID (e.g., "00", "01")

        Returns:
            Dictionary of loaded images
        """
        sample = {}
        for data_type in ["ipfxMag", "ipfyMag", "ipfzMag", "oriindx", "t"]:
            filepath = self.config.get_data_path(sample_id, data_type)
            img = Image.open(filepath)
            img_array = np.array(img)

            # Convert temperature to grayscale if RGB
            if data_type == "t" and len(img_array.shape) == 3:
                img_array = np.mean(img_array, axis=2, keepdims=True)
            elif len(img_array.shape) == 2:
                img_array = img_array[:, :, np.newaxis]

            sample[data_type] = img_array

        return sample

    def prepare_input(self, sample_t: dict, sample_t1: dict) -> np.ndarray:
        """
        Prepare input array from two samples.

        Args:
            sample_t: Current sample
            sample_t1: Next sample (for temperature)

        Returns:
            Input array of shape (H, W, C)
        """
        # Concatenate: current state + future temperature
        input_components = [
            sample_t["ipfxMag"],
            sample_t["ipfyMag"],
            sample_t["ipfzMag"],
            sample_t["oriindx"],
            sample_t["t"],
            sample_t1["t"]
        ]
        input_array = np.concatenate(input_components, axis=2)
        return input_array

    def extract_patches(self, image: np.ndarray) -> tuple:
        """
        Extract overlapping patches from image.

        Args:
            image: Image array of shape (H, W, C)

        Returns:
            Tuple of (patches, positions) where:
                - patches: List of patch arrays
                - positions: List of (top, left) positions
        """
        h, w, c = image.shape
        patches = []
        positions = []

        # Calculate number of patches needed
        n_patches_h = (h - self.overlap) // self.stride + 1
        n_patches_w = (w - self.overlap) // self.stride + 1

        for i in range(n_patches_h):
            for j in range(n_patches_w):
                top = i * self.stride
                left = j * self.stride

                # Ensure patch doesn't go out of bounds
                if top + self.patch_size > h:
                    top = h - self.patch_size
                if left + self.patch_size > w:
                    left = w - self.patch_size

                patch = image[top:top + self.patch_size, left:left + self.patch_size]
                patches.append(patch)
                positions.append((top, left))

        return patches, positions

    def merge_patches(
        self,
        patches: list,
        positions: list,
        output_shape: tuple
    ) -> np.ndarray:
        """
        Merge overlapping patches into full image using weighted averaging.

        Args:
            patches: List of patch arrays
            positions: List of (top, left) positions
            output_shape: Output image shape (H, W, C)

        Returns:
            Merged image array
        """
        h, w, c = output_shape
        output = np.zeros((h, w, c), dtype=np.float32)
        weights = np.zeros((h, w, 1), dtype=np.float32)

        # Create weight mask (higher weight in center, lower at edges)
        weight_mask = self._create_weight_mask(self.patch_size)

        for patch, (top, left) in zip(patches, positions):
            output[top:top + self.patch_size, left:left + self.patch_size] += patch * weight_mask
            weights[top:top + self.patch_size, left:left + self.patch_size] += weight_mask

        # Normalize by weights
        output = np.divide(output, weights, where=weights > 0)

        return output

    def _create_weight_mask(self, size: int) -> np.ndarray:
        """
        Create a weight mask with smooth transition at edges.

        Args:
            size: Size of the square mask

        Returns:
            Weight mask of shape (size, size, 1)
        """
        # Create 1D weight profile (higher in center, lower at edges)
        x = np.linspace(-1, 1, size)
        weight_1d = np.exp(-x**2 / 0.5)  # Gaussian-like weights

        # Create 2D weight mask
        weight_2d = np.outer(weight_1d, weight_1d)
        weight_mask = weight_2d[:, :, np.newaxis]

        return weight_mask

    @torch.no_grad()
    def predict_full_image(
        self,
        sample_id_t: str,
        sample_id_t1: str,
        batch_size: int = None
    ) -> np.ndarray:
        """
        Predict full-resolution microstructure evolution.

        Args:
            sample_id_t: Current sample ID
            sample_id_t1: Next sample ID (for temperature)
            batch_size: Batch size for processing patches

        Returns:
            Predicted microstructure array of shape (H, W, 12)
        """
        self.model.eval()

        if batch_size is None:
            batch_size = self.config.INFERENCE_BATCH_SIZE

        # Load samples
        print(f"Loading samples {sample_id_t} and {sample_id_t1}...")
        sample_t = self.load_sample(sample_id_t)
        sample_t1 = self.load_sample(sample_id_t1)

        # Prepare input
        input_array = self.prepare_input(sample_t, sample_t1)
        h, w = input_array.shape[:2]

        print(f"Image size: {h} x {w}")
        print(f"Extracting patches (size={self.patch_size}, overlap={self.overlap})...")

        # Extract patches
        patches, positions = self.extract_patches(input_array)
        print(f"Number of patches: {len(patches)}")

        # Process patches in batches
        predicted_patches = []

        print("Processing patches...")
        for i in tqdm(range(0, len(patches), batch_size)):
            batch_patches = patches[i:i + batch_size]

            # Convert to tensor
            batch_tensor = []
            for patch in batch_patches:
                # Normalize and convert to tensor
                patch_normalized = patch.astype(np.float32) / 255.0
                patch_tensor = torch.from_numpy(patch_normalized).permute(2, 0, 1)  # (C, H, W)
                batch_tensor.append(patch_tensor)

            batch_tensor = torch.stack(batch_tensor).to(self.device)  # (B, C, H, W)

            # Predict
            output = self.model(batch_tensor)  # (B, 12, H, W)

            # Convert back to numpy
            output = output.cpu().numpy()

            for j in range(output.shape[0]):
                pred_patch = output[j].transpose(1, 2, 0)  # (H, W, 12)
                predicted_patches.append(pred_patch)

        # Merge patches
        print("Merging patches...")
        output_shape = (h, w, 12)  # 12 output channels
        merged_output = self.merge_patches(predicted_patches, positions, output_shape)

        # Convert back to [0, 255] range
        merged_output = np.clip(merged_output * 255, 0, 255).astype(np.uint8)

        return merged_output

    def save_output(self, output: np.ndarray, sample_id: str, output_dir: Path):
        """
        Save predicted output as TIFF files.

        Args:
            output: Predicted output array (H, W, 12)
            sample_id: Sample ID for naming
            output_dir: Directory to save outputs
        """
        output_dir.mkdir(exist_ok=True, parents=True)

        # Split into components
        components = {
            "ipfxMag": output[:, :, 0:3],
            "ipfyMag": output[:, :, 3:6],
            "ipfzMag": output[:, :, 6:9],
            "oriindx": output[:, :, 9:12]
        }

        print(f"\nSaving outputs to {output_dir}...")
        for name, data in components.items():
            filepath = output_dir / f"{sample_id}_{name}_predicted.tiff"
            img = Image.fromarray(data)
            img.save(filepath)
            print(f"Saved: {filepath}")

    def visualize_comparison(
        self,
        prediction: np.ndarray,
        target_sample_id: str,
        output_dir: Path
    ):
        """
        Create visualization comparing prediction with ground truth.

        Args:
            prediction: Predicted array (H, W, 12)
            target_sample_id: ID of target sample for comparison
            output_dir: Directory to save visualization
        """
        # Load ground truth
        target = self.load_sample(target_sample_id)

        # Create comparison plot
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        fig.suptitle(f'Prediction vs Ground Truth (Sample {target_sample_id})', fontsize=16)

        components = [
            ("IPF X", prediction[:, :, 0:3], target["ipfxMag"]),
            ("IPF Y", prediction[:, :, 3:6], target["ipfyMag"]),
            ("IPF Z", prediction[:, :, 6:9], target["ipfzMag"]),
            ("Orientation Index", prediction[:, :, 9:12], target["oriindx"])
        ]

        for i, (name, pred, tgt) in enumerate(components):
            # Prediction
            axes[i, 0].imshow(pred)
            axes[i, 0].set_title(f'{name} - Prediction')
            axes[i, 0].axis('off')

            # Ground truth
            axes[i, 1].imshow(tgt)
            axes[i, 1].set_title(f'{name} - Ground Truth')
            axes[i, 1].axis('off')

            # Difference
            diff = np.abs(pred.astype(float) - tgt.astype(float)).mean(axis=2)
            im = axes[i, 2].imshow(diff, cmap='hot', vmin=0, vmax=50)
            axes[i, 2].set_title(f'{name} - Difference')
            axes[i, 2].axis('off')
            plt.colorbar(im, ax=axes[i, 2], fraction=0.046)

        plt.tight_layout()
        save_path = output_dir / f"comparison_{target_sample_id}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nComparison visualization saved to {save_path}")
        plt.close()


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Run inference on full-resolution images")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--sample_t', type=str, required=True,
                       help='Current sample ID (e.g., "00")')
    parser.add_argument('--sample_t1', type=str, required=True,
                       help='Next sample ID for temperature (e.g., "01")')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='outputs/inference',
                       help='Directory to save predictions')
    parser.add_argument('--model_type', type=str, default='cnn_lstm',
                       choices=['cnn_lstm', 'conv_lstm'], help='Model architecture type')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'mps', 'cpu'], help='Device to use')
    parser.add_argument('--patch_size', type=int, default=None,
                       help='Patch size for inference')
    parser.add_argument('--overlap', type=int, default=None,
                       help='Overlap between patches')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for processing patches')
    parser.add_argument('--compare', action='store_true',
                       help='Create comparison visualization with ground truth')

    args = parser.parse_args()

    # Load configuration
    config = Config()
    config.DATA_DIR = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print("=" * 80)
    print("Full-Resolution Inference")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Current sample: {args.sample_t}")
    print(f"Next sample: {args.sample_t1}")
    print(f"Output directory: {output_dir}")

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Create model
    print(f"\nCreating {args.model_type} model...")
    model = get_model(config, args.model_type)
    model = model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    load_checkpoint(Path(args.checkpoint), model, device=device)

    # Create inference engine
    inference = SlidingWindowInference(
        model,
        config,
        device,
        patch_size=args.patch_size,
        overlap=args.overlap
    )

    # Run inference
    print("\n" + "=" * 80)
    print("Running inference...")
    print("=" * 80)

    prediction = inference.predict_full_image(
        args.sample_t,
        args.sample_t1,
        batch_size=args.batch_size
    )

    # Save outputs
    inference.save_output(prediction, f"{args.sample_t}_to_{args.sample_t1}", output_dir)

    # Create comparison visualization if requested
    if args.compare:
        print("\nCreating comparison visualization...")
        inference.visualize_comparison(prediction, args.sample_t1, output_dir)

    print("\n" + "=" * 80)
    print("Inference completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
