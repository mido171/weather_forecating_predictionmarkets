from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from klga_tmax.config import load_settings
from klga_tmax.db.engine import make_engine
from klga_tmax.evaluation.gribstream_bias_backtest import (
    DEFAULT_CUTOFF_ID,
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_LABEL_LAG_DAYS,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MIN_TEST_DAYS,
    format_markdown_table,
    run_backtest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Leakage-safe KLGA Tmax GribStream 45-day half-life bias backtest."
    )
    parser.add_argument("--cutoff-id", default=DEFAULT_CUTOFF_ID)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--half-life-days", type=float, default=DEFAULT_HALF_LIFE_DAYS)
    parser.add_argument("--label-lag-days", type=int, default=DEFAULT_LABEL_LAG_DAYS)
    parser.add_argument("--min-test-days", type=int, default=DEFAULT_MIN_TEST_DAYS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summary.json and CSV outputs. Defaults under KLGA_ARTIFACT_ROOT.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Acknowledge the read-only database scan and local artifact writes.",
    )
    args = parser.parse_args()
    if not args.execute:
        parser.error("database backtest is disabled; re-run with --execute")

    settings = load_settings(require_db=True)
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = settings.artifact_root / "reports" / "gribstream_bias_backtest" / stamp

    engine = make_engine(settings.database_url)
    with engine.connect() as connection:
        result = run_backtest(
            connection,
            cutoff_id=args.cutoff_id,
            lookback_days=args.lookback_days,
            half_life_days=args.half_life_days,
            label_lag_days=args.label_lag_days,
            min_test_days=args.min_test_days,
            output_dir=output_dir,
        )

    print("KLGA GribStream Tmax bias-correction backtest")
    print(f"cutoff_id={args.cutoff_id}")
    print("model_run_buffer_hours=4")
    print(f"lookback_days={args.lookback_days}")
    print(f"half_life_days={args.half_life_days}")
    print(f"label_lag_days={args.label_lag_days}")
    print(f"min_test_days={args.min_test_days}")
    print(f"output_dir={output_dir}")
    print()
    print(format_markdown_table(result.summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
