"""
Configuration file for CNN-LSTM microstructure evolution prediction.
Contains all hyperparameters and settings for training, evaluation, and inference.
"""

import os
from pathlib import Path


class Config:
    """Configuration class containing all hyperparameters and settings."""

    # Data paths
    DATA_DIR = Path("data")
    CHECKPOINT_DIR = Path("checkpoints")
    LOG_DIR = Path("logs")
    OUTPUT_DIR = Path("outputs")

    # Dataset parameters
    NUM_SAMPLES = 10  # Samples 00-09
    SAMPLE_IDS = [f"{i:02d}" for i in range(NUM_SAMPLES)]
    IMAGE_HEIGHT = 1554
    IMAGE_WIDTH = 2916

    # Input channels
    IPFX_CHANNELS = 3  # RGB
    IPFY_CHANNELS = 3  # RGB
    IPFZ_CHANNELS = 3  # RGB
    ORIINDX_CHANNELS = 3  # RGB
    TEMP_CHANNELS = 1  # Grayscale (will convert RGB to grayscale if needed)

    # Total input: current_state (IPFx + IPFy + IPFz + oriindx + temp) + future_temp
    INPUT_CHANNELS = IPFX_CHANNELS + IPFY_CHANNELS + IPFZ_CHANNELS + ORIINDX_CHANNELS + TEMP_CHANNELS + TEMP_CHANNELS
    # Total output: next_state (IPFx + IPFy + IPFz + oriindx)
    OUTPUT_CHANNELS = IPFX_CHANNELS + IPFY_CHANNELS + IPFZ_CHANNELS + ORIINDX_CHANNELS

    # Patch-based training (due to large image size)
    PATCH_SIZE = 256  # Size of patches to extract during training
    PATCHES_PER_IMAGE = 16  # Number of random patches to extract per image pair

    # Model architecture
    ENCODER_CHANNELS = [32, 64, 128, 256, 512]  # CNN encoder channels
    LSTM_HIDDEN_SIZE = 512  # LSTM hidden state size
    LSTM_NUM_LAYERS = 2  # Number of LSTM layers
    LSTM_DROPOUT = 0.3  # Dropout in LSTM
    USE_SKIP_CONNECTIONS = True  # Use skip connections in decoder

    # Training parameters
    BATCH_SIZE = 4  # Batch size for training
    NUM_EPOCHS = 300  # Number of training epochs
    LEARNING_RATE = 1e-4  # Initial learning rate
    WEIGHT_DECAY = 1e-5  # L2 regularization
    GRAD_CLIP_MAX_NORM = 1.0  # Gradient clipping

    # Learning rate scheduler
    LR_SCHEDULER = "ReduceLROnPlateau"  # Options: "ReduceLROnPlateau", "CosineAnnealing"
    LR_PATIENCE = 10  # Patience for ReduceLROnPlateau
    LR_FACTOR = 0.5  # Factor to reduce LR
    LR_MIN = 1e-7  # Minimum learning rate

    # Loss function weights
    LOSS_TYPE = "MSE"  # Options: "MSE", "L1", "Combined"
    IPF_LOSS_WEIGHT = 1.0  # Weight for IPF channels
    ORIINDX_LOSS_WEIGHT = 1.0  # Weight for orientation index
    USE_SSIM_LOSS = False  # Add SSIM loss component
    SSIM_WEIGHT = 0.1  # Weight for SSIM loss if enabled

    # Data augmentation
    USE_AUGMENTATION = True
    AUG_ROTATION = True  # Random 90° rotations
    AUG_FLIP_H = True  # Horizontal flip
    AUG_FLIP_V = True  # Vertical flip
    AUG_PROBABILITY = 0.5  # Probability of applying augmentation

    # Train/validation split
    TRAIN_SPLIT = 0.7  # 70% for training
    VAL_SPLIT = 0.3  # 30% for validation
    RANDOM_SEED = 42  # For reproducibility

    # Early stopping
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 30  # Epochs without improvement

    # Checkpointing
    SAVE_BEST_MODEL = True
    SAVE_EVERY_N_EPOCHS = 10  # Save checkpoint every N epochs

    # Logging
    LOG_EVERY_N_STEPS = 10  # Log training metrics every N steps
    VAL_EVERY_N_EPOCHS = 1  # Validate every N epochs

    # Visualization during training
    VISUALIZE_EVERY_N_EPOCHS = 5  # Save visualization every N epochs
    NUM_VIS_SAMPLES = 2  # Number of samples to visualize

    # Inference parameters
    INFERENCE_OVERLAP = 64  # Overlap between patches during sliding window inference
    INFERENCE_BATCH_SIZE = 8  # Batch size for inference

    # Device
    DEVICE = "cuda"  # Options: "cuda", "cpu", "mps"
    USE_MIXED_PRECISION = True  # Use automatic mixed precision training

    # Number of workers for data loading
    NUM_WORKERS = 4
    PIN_MEMORY = True

    @classmethod
    def create_dirs(cls):
        """Create necessary directories if they don't exist."""
        cls.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        cls.LOG_DIR.mkdir(exist_ok=True, parents=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    @classmethod
    def get_data_path(cls, sample_id: str, data_type: str) -> Path:
        """
        Get the path to a specific data file.

        Args:
            sample_id: Sample ID (e.g., "00", "01", ...)
            data_type: Type of data (e.g., "ipfxMag", "ipfyMag", "ipfzMag", "oriindx", "t")

        Returns:
            Path to the data file
        """
        return cls.DATA_DIR / sample_id / f"{sample_id}_{data_type}_1.tiff"

    @classmethod
    def print_config(cls):
        """Print configuration summary."""
        print("=" * 80)
        print("Configuration Summary")
        print("=" * 80)
        print(f"Data Directory: {cls.DATA_DIR}")
        print(f"Number of Samples: {cls.NUM_SAMPLES}")
        print(f"Image Size: {cls.IMAGE_HEIGHT} x {cls.IMAGE_WIDTH}")
        print(f"Patch Size: {cls.PATCH_SIZE}")
        print(f"Input Channels: {cls.INPUT_CHANNELS}")
        print(f"Output Channels: {cls.OUTPUT_CHANNELS}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Number of Epochs: {cls.NUM_EPOCHS}")
        print(f"Device: {cls.DEVICE}")
        print(f"Mixed Precision: {cls.USE_MIXED_PRECISION}")
        print("=" * 80)


if __name__ == "__main__":
    # Test configuration
    Config.create_dirs()
    Config.print_config()

    # Test data path retrieval
    print("\nSample data paths:")
    for sample_id in ["00", "01"]:
        for data_type in ["ipfxMag", "ipfyMag", "ipfzMag", "oriindx", "t"]:
            path = Config.get_data_path(sample_id, data_type)
            exists = "✓" if path.exists() else "✗"
            print(f"{exists} {path}")
