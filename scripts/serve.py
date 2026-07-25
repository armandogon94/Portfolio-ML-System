#!/usr/bin/env python
"""Launch the FastAPI inference server.

Binds 8070 by default, per docs/ports.example.md and the master port allocation.
Never 8000: that is a framework default and it collides with every other project
on this machine. Override with BACKEND_PORT.

Inside Docker the container listens on 8000 and compose maps
${BACKEND_PORT:-8070}:8000 — the host-side port is the same either way.

Usage:
    uv run python scripts/serve.py
    BACKEND_PORT=8071 uv run python scripts/serve.py --reload
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: Host-side default for this repository. See docs/ports.example.md.
DEFAULT_PORT = 8070

#: Host ports reserved by docs/ports.example.md. Keep this named constant so a
#: parameterized test can prevent the documentation and launcher from drifting.
REFUSED_PORTS = {80, 443, 3000, 5000, 5432, 6379, 7000, 8000, 11434}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("BACKEND_PORT", DEFAULT_PORT)),
        help=f"Port to bind (default {DEFAULT_PORT}, or $BACKEND_PORT).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind.")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")
    args = parser.parse_args()

    if args.port in REFUSED_PORTS:
        parser.error(
            f"Port {args.port} is reserved: 5000/7000 are macOS AirPlay Receiver, "
            "11434 is Ollama, 3000/8000/5432/6379 are framework defaults that "
            "collide across projects, and 80/443 need root privileges and belong "
            "to a reverse proxy. Use 8070. See docs/ports.example.md."
        )

    import uvicorn

    uvicorn.run("src.serving.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
