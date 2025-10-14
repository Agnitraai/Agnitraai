"""CLI entry point for the Agentic Optimization API server."""

from __future__ import annotations

import argparse
from typing import Optional

import uvicorn

from .app import create_app


def run_api(argv: Optional[list[str]] = None) -> None:
    """Launch the Agentic Optimization API."""

    parser = argparse.ArgumentParser(description="Run the Agnitra Agentic Optimization API server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8080, help="Port number for the API server.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable autoreload (development only).",
    )
    args = parser.parse_args(argv)

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":  # pragma: no cover
    run_api()

