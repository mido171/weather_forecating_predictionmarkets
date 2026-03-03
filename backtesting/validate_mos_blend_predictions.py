from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_FILES = [
    "dev_predictions.parquet",
    "test_predictions.parquet",
    "metrics_dev.json",
    "metrics_test.json",
    "blend_weights.json",
]

REQUIRED_COLUMNS = [
    "target_date_local",
    "y_tmax",
    "q_0.05",
    "q_0.10",
    "q_0.25",
    "q_0.50",
    "q_0.75",
    "q_0.90",
    "q_0.95",
]

Q_COLS = ["q_0.05", "q_0.10", "q_0.25", "q_0.50", "q_0.75", "q_0.90", "q_0.95"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate MOS blend prediction artifacts.")
    p.add_argument("--blend-dir", required=True, help="Directory containing blend artifacts.")
    p.add_argument("--coverage-start", default="2024-10-01")
    p.add_argument("--coverage-end", default="2025-12-31")
    p.add_argument("--out-json", required=True)
    return p.parse_args()


def validate_blend(blend_dir: Path, cov_start: str, cov_end: str) -> dict:
    result: dict = {
        "blend_dir": str(blend_dir),
        "coverage_start": cov_start,
        "coverage_end": cov_end,
        "checks": {},
    }

    # 1) Required files
    missing_files = [name for name in REQUIRED_FILES if not (blend_dir / name).exists()]
    result["checks"]["required_files"] = {
        "ok": len(missing_files) == 0,
        "missing": missing_files,
    }
    if missing_files:
        result["ok"] = False
        return result

    dev = pd.read_parquet(blend_dir / "dev_predictions.parquet")
    test = pd.read_parquet(blend_dir / "test_predictions.parquet")

    # 2) Required columns
    dev_missing = [c for c in REQUIRED_COLUMNS if c not in dev.columns]
    test_missing = [c for c in REQUIRED_COLUMNS if c not in test.columns]
    result["checks"]["required_columns"] = {
        "ok": (len(dev_missing) == 0 and len(test_missing) == 0),
        "dev_missing": dev_missing,
        "test_missing": test_missing,
    }

    # 3) Coverage in test predictions
    test_dates = pd.to_datetime(test["target_date_local"], errors="coerce").dt.normalize()
    cov_mask = test_dates.between(pd.Timestamp(cov_start), pd.Timestamp(cov_end))
    cov_count = int(cov_mask.sum())
    expected_count = int(
        len(pd.date_range(pd.Timestamp(cov_start), pd.Timestamp(cov_end), freq="D"))
    )
    result["checks"]["coverage"] = {
        "ok": cov_count == expected_count,
        "covered_days": cov_count,
        "expected_days": expected_count,
        "test_min_date": str(test_dates.min().date()) if len(test_dates) else None,
        "test_max_date": str(test_dates.max().date()) if len(test_dates) else None,
    }

    # 4) Quantile monotonicity (dev + test)
    qdf_dev = dev[Q_COLS].copy()
    qdf_test = test[Q_COLS].copy()
    nondec_dev = (qdf_dev.diff(axis=1).iloc[:, 1:] >= 0).all(axis=1)
    nondec_test = (qdf_test.diff(axis=1).iloc[:, 1:] >= 0).all(axis=1)
    bad_rows_dev = int((~nondec_dev).sum())
    bad_rows_test = int((~nondec_test).sum())
    result["checks"]["quantile_non_decreasing"] = {
        "ok": (bad_rows_dev == 0 and bad_rows_test == 0),
        "violating_rows_dev": bad_rows_dev,
        "violating_rows_test": bad_rows_test,
    }

    # 5) Duplicate date check (dev + test)
    dev_dates = pd.to_datetime(dev["target_date_local"], errors="coerce").dt.normalize()
    test_dates = pd.to_datetime(test["target_date_local"], errors="coerce").dt.normalize()
    dev_dups = int(dev_dates.duplicated().sum())
    test_dups = int(test_dates.duplicated().sum())
    result["checks"]["duplicate_target_date_local"] = {
        "ok": (dev_dups == 0 and test_dups == 0),
        "dev_duplicates": dev_dups,
        "test_duplicates": test_dups,
    }

    result["ok"] = all(c.get("ok", False) for c in result["checks"].values())
    return result


def main() -> None:
    args = parse_args()
    out = validate_blend(
        blend_dir=Path(args.blend_dir),
        cov_start=str(args.coverage_start),
        cov_end=str(args.coverage_end),
    )
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    if not out.get("ok", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
