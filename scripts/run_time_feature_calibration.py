from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from weather_ml import time_sweep_calibration


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate time sweep experiment.")
    parser.add_argument("--sweep-root", required=True, help="Path to sweep root folder.")
    parser.add_argument("--experiment-id", required=True, help="Experiment id (e.g., E24).")
    parser.add_argument(
        "--run-dir",
        help="Optional experiment run dir (overrides sweep lookup).",
    )
    parser.add_argument(
        "--cal-start", default="2025-01-01", help="Calibration start date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--cal-end", default="2025-12-31", help="Calibration end date (YYYY-MM-DD)."
    )
    parser.add_argument("--ddof", type=int, default=1, help="Std ddof.")
    parser.add_argument("--truth-lag", type=int, default=2, help="Truth lag days.")
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow calibration window to overlap training/validation.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory (defaults to experiment run dir).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sweep_root = Path(args.sweep_root)
    output_dir = Path(args.output_dir) if args.output_dir else None
    run_dir_override = Path(args.run_dir) if args.run_dir else None
    time_sweep_calibration.run_experiment_calibration(
        sweep_root=sweep_root,
        experiment_id=args.experiment_id,
        run_dir_override=run_dir_override,
        cal_start=_parse_date(args.cal_start),
        cal_end=_parse_date(args.cal_end),
        ddof=int(args.ddof),
        truth_lag=int(args.truth_lag),
        allow_overlap=bool(args.allow_overlap),
        output_dir=output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
