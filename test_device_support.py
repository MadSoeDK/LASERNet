#!/usr/bin/env python
"""
Test script to demonstrate multi-device support in LASERNet.

This script shows how to use the device utilities to detect and select
different compute devices (CUDA, Apple Silicon MPS, CPU).
"""

from lasernet.utils import get_device, get_device_info, print_device_info


def main():
    print("=" * 70)
    print("LASERNet Multi-Device Support Test")
    print("=" * 70)
    print()

    # Print detailed device information
    print_device_info()
    print()

    # Test auto-detection
    print("Auto-detection test:")
    print("-" * 50)
    device = get_device()
    print(f"Selected device: {device}")
    print()

    # Test manual CPU selection
    print("Manual CPU selection test:")
    print("-" * 50)
    device_cpu = get_device("cpu")
    print(f"Selected device: {device_cpu}")
    print()

    # Get device info programmatically
    print("Device information (programmatic):")
    print("-" * 50)
    info = get_device_info()
    print(f"CUDA available: {info['cuda_available']}")
    print(f"MPS available: {info['mps_available']}")
    print(f"CPU cores: {info['cpu_count']}")

    if info['cuda_available']:
        print(f"CUDA device: {info['cuda_device_name']}")
        print(f"CUDA memory: {info['cuda_memory_gb']:.1f} GB")
    print()

    # Test environment variable override
    import os
    print("Environment variable override test:")
    print("-" * 50)

    env_device = os.environ.get("TORCH_DEVICE")
    if env_device:
        print(f"TORCH_DEVICE is set to: {env_device}")
        device_env = get_device()
        print(f"Using device: {device_env}")
    else:
        print("TORCH_DEVICE not set (using auto-detection)")
    print()

    print("=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
