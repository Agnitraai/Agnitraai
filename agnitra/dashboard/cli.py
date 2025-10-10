"""CLI entry-point for launching the Agnitra dashboard server."""

from __future__ import annotations

import argparse
from typing import Optional

import uvicorn

from .app import create_app


def run_dashboard(argv: Optional[list[str]] = None) -> None:
    """Run the Agnitra dashboard using Uvicorn."""
    parser = argparse.ArgumentParser(description="Launch the Agnitra web dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number for the dashboard server.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable autoreload (for development only).",
    )
    args = parser.parse_args(argv)

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    run_dashboard()
