"""Device selection utility for PyTorch with Apple Silicon MPS support."""

import os

import torch


def get_device() -> torch.device:
    """Auto-detect the best available device: MPS > CUDA > CPU.

    Sets PYTORCH_ENABLE_MPS_FALLBACK=1 for operations not yet supported on MPS.

    Returns:
        torch.device for the best available accelerator.
    """
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def device_info() -> dict:
    """Return device information for logging."""
    device = get_device()
    info = {
        "device": str(device),
        "pytorch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
    return info
