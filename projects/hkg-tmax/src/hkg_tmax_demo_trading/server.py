"""Uvicorn entrypoint for the local HKG Polymarket demo backtester."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .api import create_app, default_database_url, default_repo_root, default_static_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the HKG Polymarket demo backtester")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--database-url", default=default_database_url())
    parser.add_argument("--repo-root", type=Path, default=default_repo_root())
    parser.add_argument("--static-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    static_dir = args.static_dir or default_static_dir(args.repo_root)
    app = create_app(
        repo_root=args.repo_root,
        database_url=args.database_url,
        static_dir=static_dir,
        apply_schema_on_startup=True,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
