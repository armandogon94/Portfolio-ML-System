"""Screenshot command failures must never be mistaken for captured images."""

from __future__ import annotations

import builtins
import sys

from scripts import capture_screenshots


def test_missing_playwright_returns_zero_captures(monkeypatch):
    real_import = builtins.__import__

    def import_without_playwright(name, *args, **kwargs):
        if name == "playwright.sync_api":
            raise ImportError("Playwright intentionally hidden by this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_playwright)

    captured = capture_screenshots.capture(
        "http://localhost:3070",
        "http://localhost:5070",
        timeout_ms=1,
    )

    assert captured == 0


def test_missing_playwright_makes_the_command_fail(monkeypatch):
    real_import = builtins.__import__

    def import_without_playwright(name, *args, **kwargs):
        if name == "playwright.sync_api":
            raise ImportError("Playwright intentionally hidden by this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_playwright)
    monkeypatch.setattr(sys, "argv", ["capture_screenshots.py"])

    assert capture_screenshots.main() == 1


def test_partial_capture_makes_the_command_fail(monkeypatch):
    monkeypatch.setattr(capture_screenshots, "capture", lambda *args, **kwargs: 1)
    monkeypatch.setattr(sys, "argv", ["capture_screenshots.py"])

    assert capture_screenshots.main() == 1
