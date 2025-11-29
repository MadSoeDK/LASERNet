"""
Device utilities for LASERNet.

This module provides utilities for device selection and management across
CUDA (NVIDIA GPUs), Apple Silicon (MPS), and CPU.
"""

import os
from typing import Optional

import torch


def get_device(device_name: Optional[str] = None) -> torch.device:
    """
    Get the best available device for PyTorch.

    Supports CUDA (NVIDIA GPUs), Apple Silicon (MPS), and CPU.
    Device selection priority: CUDA > MPS > CPU

    Args:
        device_name: Optional device override. Can be:
            - "cuda" or "gpu": Force CUDA (raises error if unavailable)
            - "mps": Force Apple Silicon MPS (raises error if unavailable)
            - "cpu": Force CPU
            - None: Auto-detect best available device

    Returns:
        torch.device object

    Raises:
        RuntimeError: If requested device is not available

    Environment Variables:
        TORCH_DEVICE: Override device selection (same values as device_name)

    Examples:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device("cuda")  # Force CUDA
        >>> device = get_device("cpu")  # Force CPU
    """
    # Check environment variable override
    env_device = os.environ.get("TORCH_DEVICE")
    requested_device = device_name or env_device

    if requested_device:
        requested_device = requested_device.lower().strip()

        # CUDA
        if requested_device in ("cuda", "gpu"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA device requested but CUDA is not available. "
                    "Please check your PyTorch installation and GPU drivers."
                )
            device = torch.device("cuda")
            print(f"Using CUDA (forced): {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return device

        # Apple Silicon MPS
        elif requested_device == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError(
                    "MPS device requested but MPS is not available. "
                    "Please ensure you're running on Apple Silicon with PyTorch 1.12+."
                )
            device = torch.device("mps")
            print("Using Apple Silicon MPS (forced)")
            return device

        # CPU
        elif requested_device == "cpu":
            device = torch.device("cpu")
            print("Using CPU (forced)")
            return device

        else:
            raise ValueError(
                f"Invalid device name: '{requested_device}'. "
                "Valid options: 'cuda', 'gpu', 'mps', 'cpu'"
            )

    # Auto-detection (no device specified)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return device

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS")
        return device

    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU detected)")
        return device


def get_device_info() -> dict:
    """
    Get detailed information about available devices.

    Returns:
        Dictionary containing device availability and capabilities

    Example:
        >>> info = get_device_info()
        >>> print(info)
        {
            'cuda_available': True,
            'cuda_device_count': 1,
            'cuda_device_name': 'NVIDIA A100-SXM4-40GB',
            'cuda_memory_gb': 40.0,
            'mps_available': False,
            'cpu_count': 64
        }
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': 0,
        'cuda_device_name': None,
        'cuda_memory_gb': None,
        'mps_available': hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        'cpu_count': os.cpu_count(),
    }

    if info['cuda_available']:
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3

    return info


def print_device_info() -> None:
    """
    Print detailed information about available devices.

    Example:
        >>> print_device_info()
        Device Information:
        ==================
        CUDA (NVIDIA GPU):
          Available: Yes
          Devices: 1
          Device 0: NVIDIA A100-SXM4-40GB
          Memory: 40.0 GB
        Apple Silicon MPS:
          Available: No
        CPU:
          Cores: 64
    """
    info = get_device_info()

    print("Device Information:")
    print("=" * 50)

    # CUDA
    print("CUDA (NVIDIA GPU):")
    if info['cuda_available']:
        print(f"  Available: Yes")
        print(f"  Devices: {info['cuda_device_count']}")
        print(f"  Device 0: {info['cuda_device_name']}")
        print(f"  Memory: {info['cuda_memory_gb']:.1f} GB")
    else:
        print(f"  Available: No")

    # MPS
    print("\nApple Silicon MPS:")
    if info['mps_available']:
        print(f"  Available: Yes")
    else:
        print(f"  Available: No")

    # CPU
    print("\nCPU:")
    print(f"  Cores: {info['cpu_count']}")

    print("=" * 50)
