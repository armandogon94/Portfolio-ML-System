"""Tests for device detection utility."""

import torch

from src.device import device_info, get_device


class TestGetDevice:

    def test_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_device_is_valid(self):
        device = get_device()
        assert device.type in ("cpu", "mps", "cuda")


class TestDeviceInfo:

    def test_returns_dict(self):
        info = device_info()
        assert isinstance(info, dict)

    def test_has_required_keys(self):
        info = device_info()
        assert "device" in info
        assert "pytorch_version" in info
        assert "mps_available" in info
        assert "cuda_available" in info

    def test_device_is_string(self):
        info = device_info()
        assert isinstance(info["device"], str)

    def test_booleans_are_bool(self):
        info = device_info()
        assert isinstance(info["mps_available"], bool)
        assert isinstance(info["cuda_available"], bool)
