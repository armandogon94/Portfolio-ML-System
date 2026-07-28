#!/usr/bin/env python
"""Run the documented pipeline end to end: download -> train -> evaluate.

A thin orchestrator over the three scripts that do the work, so the README can
say "make all" and mean it. Each stage's exit code is honoured: a failed download
does not silently proceed to training on nothing.

Usage:
    uv run python scripts/run_all.py
    uv run python scripts/run_all.py --sample     # fixtures only, no downloads
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(stage: str, command: list[str]) -> int:
    print(f"\n{'=' * 70}\n==> {stage}\n{'=' * 70}", flush=True)
    return subprocess.call([sys.executable, *command], cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Skip downloads and train on the committed CI fixtures (writes nothing).",
    )
    args = parser.parse_args()

    if not args.sample:
        code = _run("download", ["scripts/download_data.py", "--dataset", "all"])
        if code != 0:
            print(
                "\nDownload failed. Nothing was trained and no metric was written.\n"
                "See the remediation above, or docs/PROGRESS.md.",
                file=sys.stderr,
            )
            return code

    train = ["scripts/train.py", "--model", "all"] + (["--sample"] if args.sample else [])
    code = _run("train", train)
    if code != 0:
        return code

    return _run("evaluate", ["scripts/evaluate.py"])


if __name__ == "__main__":
    raise SystemExit(main())
