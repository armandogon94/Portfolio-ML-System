#!/usr/bin/env python
"""Capture the README screenshots with Playwright. Committed so they are regenerable.

Appendix A2 requires that the README show the application actually working, and
that the captures be reproducible rather than hand-taken one-offs.

**Seeded demo data only.** Every form is filled from the schema defaults in
`src/serving/schemas.py`, which are illustrative values chosen for the demo. No
real transaction, no real borrower, no real cardholder, and nothing personal ever
appears in an image committed to this repository.

Prerequisites — both services running, and at least one trained checkpoint:

    make docker-up                 # or: make serve  &&  make web-dev
    uv run python scripts/train.py --model all

Usage:
    uv run --extra dev python scripts/capture_screenshots.py
    uv run --extra dev python scripts/capture_screenshots.py --base-url http://localhost:3070
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from src.config import get_project_root

console = Console()

OUTPUT_DIR = get_project_root() / "docs" / "images"

#: Host-side web port for this repository. See docs/ports.example.md.
DEFAULT_BASE_URL = "http://localhost:3070"
DEFAULT_MLFLOW_URL = "http://localhost:5070"

#: Desktop viewport. Wide enough that the two-column model pages do not collapse
#: into the mobile stack, which is not what the README is illustrating.
VIEWPORT = {"width": 1440, "height": 900}

SHOTS = [
    {
        "name": "dashboard",
        "path": "/dashboard",
        "wait_for": "table",
        "full_page": True,
        "caption": "Model dashboard: status, key metric, and MLflow sparkline per model",
    },
    {
        "name": "fraud-prediction",
        "path": "/fintech/fraud",
        "submit": True,
        "wait_for": "[data-testid='decision']",
        "full_page": True,
        "caption": "Fraud scoring with its SHAP attribution and the git SHA that trained it",
    },
    {
        "name": "credit-risk-prediction",
        "path": "/fintech/credit-risk",
        "submit": True,
        "wait_for": "[data-testid='decision']",
        "full_page": True,
        "caption": "Credit decision from origination-time fields only",
    },
    {
        "name": "landing",
        "path": "/",
        "wait_for": "main",
        "full_page": False,
        "caption": "Landing page",
    },
]


def capture(base_url: str, mlflow_url: str, *, timeout_ms: int) -> int:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeout
        from playwright.sync_api import sync_playwright
    except ImportError:
        console.print(
            "[bold red]Playwright is not installed.[/bold red]\n  make screenshots-install"
        )
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    captured = 0

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)

        for shot in SHOTS:
            url = f"{base_url}{shot['path']}"
            console.print(f"==> {shot['name']}  {url}")
            try:
                page.goto(url, wait_until="networkidle", timeout=timeout_ms)

                if shot.get("submit"):
                    # The form is pre-filled from the schema defaults, so a bare
                    # submit exercises the whole predict + explain path with
                    # seeded values and no typing.
                    page.get_by_role("button", name="Score").click()

                page.wait_for_selector(shot["wait_for"], timeout=timeout_ms)
                # Let Recharts finish its enter animation before the shutter.
                page.wait_for_timeout(700)

                path = OUTPUT_DIR / f"{shot['name']}.png"
                page.screenshot(path=str(path), full_page=shot["full_page"])
                console.print(f"    [green]{path.relative_to(get_project_root())}[/green]")
                captured += 1
            except PlaywrightTimeout:
                console.print(
                    f"    [yellow]timed out waiting for {shot['wait_for']!r}. "
                    f"Is the model trained? A 503 renders UntrainedNotice, which "
                    f"has no decision element.[/yellow]"
                )

        # MLflow's own UI, captured for the experiment-tracking section.
        console.print(f"==> mlflow  {mlflow_url}")
        try:
            page.goto(mlflow_url, wait_until="networkidle", timeout=timeout_ms)
            page.wait_for_timeout(1500)
            path = OUTPUT_DIR / "mlflow-runs.png"
            page.screenshot(path=str(path), full_page=False)
            console.print(f"    [green]{path.relative_to(get_project_root())}[/green]")
            captured += 1
        except PlaywrightTimeout:
            console.print("    [yellow]MLflow not reachable — skipped.[/yellow]")

        browser.close()

    return captured


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--mlflow-url", default=DEFAULT_MLFLOW_URL)
    parser.add_argument("--timeout", type=int, default=15_000, help="Per-step timeout in ms.")
    args = parser.parse_args()

    captured = capture(args.base_url, args.mlflow_url, timeout_ms=args.timeout)

    expected = len(SHOTS) + 1  # application views plus the MLflow UI
    if captured != expected:
        console.print(
            f"\n[bold red]Incomplete capture: {captured}/{expected} screenshots.[/bold red]\n"
            "  1. make docker-up            (or: make serve && make web-dev)\n"
            "  2. uv run python scripts/train.py --model all\n"
            "Every documented application view and the MLflow UI must be captured in "
            "one run.\n"
            "Screenshots of empty states are not worth committing — see docs/PROGRESS.md."
        )
        return 1

    console.print(f"\n[green]{captured} screenshot(s) written to docs/images/.[/green]")
    console.print("[dim]Seeded demo values only. No real or personal data.[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
