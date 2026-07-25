"""Device selection prefers MPS, then CUDA, then CPU.

The session-wide MPS mock in conftest is overridden here explicitly, so this is the
only place that asserts the MPS branch — and it never allocates an MPS tensor.
"""

from __future__ import annotations

from unittest.mock import patch

from src.device import device_info, get_device


def test_prefers_mps_when_available():
    with patch("torch.backends.mps.is_available", return_value=True):
        assert get_device().type == "mps"


def test_falls_back_to_cuda_then_cpu():
    with patch("torch.backends.mps.is_available", return_value=False):
        with patch("torch.cuda.is_available", return_value=True):
            assert get_device().type == "cuda"
        with patch("torch.cuda.is_available", return_value=False):
            assert get_device().type == "cpu"


def test_sets_the_mps_fallback_env_var():
    """Some ops still have no MPS kernel; without this they hard-error."""
    import os

    get_device()
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_device_info_reports_what_a_checkpoint_needs():
    info = device_info()
    for key in ("device", "pytorch_version", "mps_available", "cuda_available"):
        assert key in info
