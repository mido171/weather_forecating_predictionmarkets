"""Training CLI entrypoint (stub)."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Weather ML training pipeline (scaffold)."
    )
    parser.add_argument(
        "--config",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--stage",
        choices=["dataset", "train", "eval", "all"],
        default="all",
        help="Pipeline stage to run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_value = args.config if args.config else "<none>"
    print(f"weather_ml.train scaffold: stage={args.stage}, config={config_value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())