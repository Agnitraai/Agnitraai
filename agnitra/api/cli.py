"""CLI entry point for the Agentic Optimization API server."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import uvicorn

from .app import create_app


_LOG = logging.getLogger(__name__)

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8080


def run_api(argv: Optional[list[str]] = None) -> None:
    """Launch the Agentic Optimization API.

    Defaults bind to ``127.0.0.1`` so a casual ``agnitra-api`` invocation
    cannot accidentally expose the optimizer to the network. Override via
    ``--host`` (or ``AGNITRA_API_HOST``) to bind publicly; doing so emits
    a warning unless ``AGNITRA_ALLOW_PUBLIC_BIND=1`` is set.
    """

    parser = argparse.ArgumentParser(description="Run the Agnitra AI Agentic Optimization API server.")
    parser.add_argument(
        "--host",
        default=os.environ.get("AGNITRA_API_HOST", _DEFAULT_HOST),
        help="Host interface to bind. Set AGNITRA_API_HOST to override the default.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("AGNITRA_API_PORT", _DEFAULT_PORT)),
        help="Port number for the API server. Set AGNITRA_API_PORT to override the default.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable autoreload (development only).",
    )
    args = parser.parse_args(argv)

    if args.host in {"0.0.0.0", "::"} and os.environ.get("AGNITRA_ALLOW_PUBLIC_BIND") != "1":
        _LOG.warning(
            "Binding API to public interface %s without authentication may expose "
            "the optimizer. Set AGNITRA_ALLOW_PUBLIC_BIND=1 to silence this warning.",
            args.host,
        )

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":  # pragma: no cover
    run_api()

