import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lasernet.data.dataset import LaserDataset
from lasernet.utils import compute_index, get_num_of_slices, load_model_from_path

# Font size configuration
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 24
plt.rcParams.update(
    {
        "font.size": SMALL_SIZE,
        "axes.titlesize": MEDIUM_SIZE,
        "axes.labelsize": MEDIUM_SIZE,
        "xtick.labelsize": SMALL_SIZE,
        "ytick.labelsize": SMALL_SIZE,
        "legend.fontsize": SMALL_SIZE,
        "figure.titlesize": BIGGER_SIZE,
    }
)

sys.path.insert(0, "../src")


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    plane = "xz"
    num_slices = get_num_of_slices(plane)  # 94 slices for xz plane
    print(f"Plane: {plane}, Number of slices: {num_slices}")

    # =========================================================================
    # Load Dataset
    # =========================================================================
    print("\nLoading microstructure datasets...")

    # Load microstructure training dataset to get normalizer
    micro_train_dataset = LaserDataset(
        data_path=Path("./data/processed/"),
        field_type="microstructure",
        split="train",
        normalize=True,
        plane="xz",
        sequence_length=3,
        target_offset=1,
    )
    print(f"Microstructure train dataset: {len(micro_train_dataset)} samples")

    # Load microstructure test dataset with same normalizer
    micro_test_dataset = LaserDataset(
        data_path=Path("./data/processed/"),
        field_type="microstructure",
        split="test",
        normalize=True,
        normalizer=micro_train_dataset.normalizer,
        plane="xz",
        sequence_length=3,
        target_offset=1,
    )
    print(f"Microstructure test dataset: {len(micro_test_dataset)} samples")
    print(f"Microstructure data shape: {micro_test_dataset.shape}")

    # =========================================================================
    # Load Models
    # =========================================================================
    print("\nLoading microstructure models...")

    checkpoint_paths = {
        "CNN-MLP": Path("./models/best_deepcnn_mlp_medium_microstructure_mseloss.ckpt"),
        "CNN-PredRNN": Path("./models/best_predrnn_shallow3l_microstructure_mseloss.ckpt"),
        "Base-ConvLSTM": Path("./models/best_baselineconvlstm_microstructure_mseloss.ckpt"),
    }

    # Load all models
    models = {}
    for name, path in checkpoint_paths.items():
        if not path.exists():
            print(f"Warning: {name} checkpoint not found at {path}")
            continue
        model = load_model_from_path(path)
        model = model.to(device)
        model = model.half()
        model.eval()
        models[name] = model
        print(f"Loaded {name}: {model.__class__.__name__} ({model.count_parameters():,} params)")

    print(f"\nLoaded {len(models)} models for comparison")

    if len(models) == 0:
        print("Error: No models found. Please check the checkpoint paths.")
        return

    # =========================================================================
    # Generate Predictions
    # =========================================================================
    timestep = 21
    viz_slice_idx = num_slices // 2  # Use middle slice
    slice_index = compute_index(
        timestep, micro_test_dataset.split, micro_test_dataset.plane, viz_slice_idx
    )  # Just to show usage
    print(f"\nGenerating predictions for slice {viz_slice_idx}...")

    # Get input data
    micro_input, micro_target, _, micro_mask = micro_test_dataset[slice_index]
    micro_input_batch = micro_input.unsqueeze(0).half().to(device)

    # Generate predictions from each model
    predictions = {}
    for name, model in models.items():
        with torch.no_grad():
            pred = model(micro_input_batch)
        predictions[name] = pred[0].cpu().float()

    # Denormalize for visualization
    micro_target_denorm = micro_test_dataset.denormalize_target(micro_target)
    predictions_denorm = {name: micro_test_dataset.denormalize_target(pred) for name, pred in predictions.items()}

    # =========================================================================
    # Create Comparison Figure
    # =========================================================================
    print("\nCreating comparison figure...")

    # Create comparison figure: 1 column × (1 + num_models) rows with shared axes
    num_rows = 1 + len(models)
    fig, axes = plt.subplots(num_rows, 1, figsize=(7, 1.5 * num_rows), sharex=True, sharey=True)

    # Ground truth
    target_rgb = np.clip(np.transpose(micro_target_denorm[0:3].numpy(), (2, 1, 0)), 0, 1).astype(np.float32)
    axes[0].imshow(target_rgb, aspect="equal", origin="lower")
    axes[0].set_title("Ground Truth")

    # Predictions from each model
    for idx, (name, pred_denorm) in enumerate(predictions_denorm.items()):
        pred_rgb = np.clip(np.transpose(pred_denorm[0:3].numpy(), (2, 1, 0)), 0, 1).astype(np.float32)

        # Compute MSE for this prediction
        mse = torch.nn.functional.mse_loss(predictions[name], micro_target).item()

        axes[idx + 1].imshow(pred_rgb, aspect="equal", origin="lower")
        axes[idx + 1].set_title(f"{name}")  # \nMSE: {mse:.6f}

    # Set common axis labels
    ylabel = fig.supylabel("Z coordinate")
    ylabel.set_fontsize(SMALL_SIZE)

    axes[-1].xaxis.label.set_fontsize(SMALL_SIZE)

    plt.tight_layout()

    output_path = "model_prediction_comparison_3.png"
    plt.savefig(output_path, format="png", bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    plt.show()

    # =========================================================================
    # Print Summary Metrics
    # =========================================================================
    print("\nModel Comparison Summary:")
    print("-" * 60)
    print(f"{'Model':<25} {'MSE (norm)':<15} {'MAE (IPF-X)':<15}")
    print("-" * 60)
    for name, pred in predictions.items():
        pred_denorm = predictions_denorm[name]
        mse = torch.nn.functional.mse_loss(pred, micro_target).item()
        mae = torch.mean(torch.abs(pred_denorm[0:3] - micro_target_denorm[0:3])).item()
        print(f"{name:<25} {mse:<15.6f} {mae:<15.4f}")
    print("-" * 60)

    # =========================================================================
    # Timestep Evolution Figure (t=18 to t=21)
    # =========================================================================
    print("\nCreating timestep evolution figure...")

    # Timesteps 18-21 correspond to test split temporal offsets 0-3
    # Dataset index = slice_idx + temporal_offset * num_slices
    timesteps = [18, 19, 20, 21]
    num_evolution_rows = len(timesteps)

    fig2, axes2 = plt.subplots(num_evolution_rows, 1, figsize=(7, 1.5 * num_evolution_rows), sharex=True, sharey=True)

    for row, timestep in enumerate(timesteps):
        # Compute dataset index for this timestep
        # temporal_offset = timestep - 18 (since test split starts at t=18)
        # But we need to account for sequence_length: the target at temporal_offset=0
        # corresponds to the target after the first 3 input frames
        temporal_offset = timestep - 18
        dataset_idx = viz_slice_idx + temporal_offset * num_slices

        # Get the target (ground truth) for this timestep
        _, micro_target_t, _, _ = micro_test_dataset[dataset_idx]
        micro_target_t_denorm = micro_test_dataset.denormalize_target(micro_target_t)

        # Convert to RGB
        target_rgb_t = np.clip(np.transpose(micro_target_t_denorm[0:3].numpy(), (2, 1, 0)), 0, 1).astype(np.float32)

        axes2[row].imshow(target_rgb_t, aspect="equal", origin="lower")
        axes2[row].set_title(f"Timestep {timestep}")

    # Set common axis labels
    fig2.supylabel("Z coordinate")
    axes2[-1].set_xlabel("X coordinate")

    # plt.suptitle("Microstructure Evolution")
    plt.tight_layout()

    output_path2 = "timestep_evolution_18_21.png"
    plt.savefig(output_path2, format="png", bbox_inches="tight")
    print(f"Saved figure to {output_path2}")
    plt.show()

    # =========================================================================
    # MSE Error Maps Figure
    # =========================================================================
    print("\nCreating MSE error maps figure...")

    num_error_rows = len(models)
    fig3, axes3 = plt.subplots(num_error_rows, 1, figsize=(7, 1.5 * num_error_rows), sharex=True, sharey=True)

    # Compute error maps for each model
    target_np = micro_target_denorm[0:3].numpy()  # [3, H, W]

    # Find global min/max for consistent colorbar across all error maps
    all_errors = []
    for name, pred_denorm in predictions_denorm.items():
        pred_np = pred_denorm[0:3].numpy()  # [3, H, W]
        error = np.mean((target_np - pred_np) ** 2, axis=0).astype(np.float32)  # [H, W]
        all_errors.append(error)
    vmin = 0
    vmax = max(e.max() for e in all_errors)

    for idx, (name, pred_denorm) in enumerate(predictions_denorm.items()):
        pred_np = pred_denorm[0:3].numpy()  # [3, H, W]
        error = np.mean((target_np - pred_np) ** 2, axis=0).astype(np.float32)  # [H, W]

        im = axes3[idx].imshow(
            error.T, cmap="RdYlBu_r", aspect="equal", interpolation="nearest", origin="lower", vmin=vmin, vmax=vmax
        )
        axes3[idx].set_title(f"{name}")

    # Set common axis labels
    fig3.supylabel("Z coordinate")
    axes3[-1].set_xlabel("X coordinate")

    # fig3.suptitle("MSE Error Maps")
    plt.tight_layout()

    # Add common colorbar horizontally at the bottom
    fig3.colorbar(im, ax=axes3, orientation="horizontal", fraction=0.05, pad=0.08, label="MSE Error")

    output_path3 = "model_mse_error_maps.png"
    plt.savefig(output_path3, format="png", bbox_inches="tight")
    print(f"Saved figure to {output_path3}")
    plt.show()


if __name__ == "__main__":
    main()
