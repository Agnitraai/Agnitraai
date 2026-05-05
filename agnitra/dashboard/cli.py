"""CLI entry-point for launching the Agnitra dashboard server."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import uvicorn

from .app import create_app


_LOG = logging.getLogger(__name__)


def run_dashboard(argv: Optional[list[str]] = None) -> None:
    """Run the Agnitra dashboard using Uvicorn.

    Defaults bind to ``127.0.0.1``. Override via ``--host`` or
    ``AGNITRA_DASHBOARD_HOST``; ``AGNITRA_DASHBOARD_PORT`` overrides the port.
    Binding to a public interface emits a warning unless
    ``AGNITRA_ALLOW_PUBLIC_BIND=1`` is set.
    """
    parser = argparse.ArgumentParser(description="Launch the Agnitra web dashboard.")
    parser.add_argument(
        "--host",
        default=os.environ.get("AGNITRA_DASHBOARD_HOST", "127.0.0.1"),
        help="Host interface to bind. Set AGNITRA_DASHBOARD_HOST to override.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("AGNITRA_DASHBOARD_PORT", 8000)),
        help="Port number for the dashboard server. Set AGNITRA_DASHBOARD_PORT to override.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable autoreload (for development only).",
    )
    args = parser.parse_args(argv)

    if args.host in {"0.0.0.0", "::"} and os.environ.get("AGNITRA_ALLOW_PUBLIC_BIND") != "1":
        _LOG.warning(
            "Binding dashboard to public interface %s. Set AGNITRA_ALLOW_PUBLIC_BIND=1 "
            "to silence this warning if exposure is intentional.",
            args.host,
        )

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    run_dashboard()
