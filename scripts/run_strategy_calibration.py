from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from weather_ml import strategy_calibration


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute residual calibration for a sweep strategy."
    )
    parser.add_argument(
        "--sweep-root",
        required=True,
        help="Path to sweep output root (contains feature_strategy_sweep.json).",
    )
    parser.add_argument(
        "--strategy-id",
        default="S02",
        help="Strategy id to calibrate (default: S02).",
    )
    parser.add_argument(
        "--cal-start",
        default="2025-01-01",
        help="Calibration window start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--cal-end",
        default="2025-12-31",
        help="Calibration window end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--ddof",
        type=int,
        default=1,
        help="Degrees of freedom for sigma (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory (default: strategy run dir).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    strategy_calibration.run_strategy_calibration(
        sweep_root=Path(args.sweep_root),
        strategy_id=str(args.strategy_id),
        cal_start=_parse_date(args.cal_start),
        cal_end=_parse_date(args.cal_end),
        ddof=int(args.ddof),
        output_dir=output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
