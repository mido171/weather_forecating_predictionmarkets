from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

from weather_ml.tfs2 import data, experiments
from weather_ml.tfs2.config import SplitConfig


LOGGER = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _run_knn_for_split(df, run_root: Path, split: SplitConfig) -> dict:
    ctx = experiments.build_context(df, run_root=run_root, split_override=split)
    result = experiments.run_knn_residual_correction(ctx)
    out = {
        "split": {
            "train_start": str(split.train_start),
            "train_end": str(split.train_end),
            "val_start": str(split.val_start),
            "val_end": str(split.val_end),
            "test_start": str(split.test_start),
            "test_end": str(split.test_end),
            "gap_dates": [str(d) for d in split.gap_dates],
        },
        "baseline_b": ctx.baseline_b.metrics,
        "experiment": {
            "experiment_id": result.get("experiment_id"),
            "name": result.get("name"),
            "metrics": result.get("metrics", {}),
        },
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare KNN experiment with different train_start.")
    parser.add_argument("--station", default="KMIA")
    parser.add_argument("--train-start-a", default="2021-02-23")
    parser.add_argument("--train-start-b", default="2021-11-30")
    parser.add_argument("--train-end", default="2024-06-30")
    parser.add_argument("--val-start", default="2024-07-01")
    parser.add_argument("--val-end", default="2025-01-30")
    parser.add_argument("--test-start", default="2025-02-01")
    parser.add_argument("--test-end", default="2025-12-31")
    parser.add_argument("--gap-date", default="2025-01-31")
    args = parser.parse_args()

    _setup_logging()

    train_a = date.fromisoformat(args.train_start_a)
    train_b = date.fromisoformat(args.train_start_b)
    train_end = date.fromisoformat(args.train_end)
    val_start = date.fromisoformat(args.val_start)
    val_end = date.fromisoformat(args.val_end)
    test_start = date.fromisoformat(args.test_start)
    test_end = date.fromisoformat(args.test_end)
    gap_date = date.fromisoformat(args.gap_date)

    dataset_start = min(train_a, train_b)
    dataset_end = test_end

    engine = data.create_engine_from_url(None)
    bundle = data.build_dataset(engine, args.station, dataset_start, dataset_end)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path("artifacts") / "knn_train_range_compare" / run_id
    root.mkdir(parents=True, exist_ok=True)

    split_a = SplitConfig(
        train_start=train_a,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
        gap_dates=(gap_date,),
    )
    split_b = SplitConfig(
        train_start=train_b,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
        gap_dates=(gap_date,),
    )

    LOGGER.info("RUN_A train_start=%s", train_a)
    res_a = _run_knn_for_split(bundle.df, root / f"train_start_{train_a}", split_a)
    LOGGER.info("RUN_B train_start=%s", train_b)
    res_b = _run_knn_for_split(bundle.df, root / f"train_start_{train_b}", split_b)

    summary = {
        "station_id": args.station,
        "dataset_ref": asdict(bundle.dataset_ref),
        "run_id": run_id,
        "result_a": res_a,
        "result_b": res_b,
    }
    out_path = root / "knn_train_range_compare.json"
    out_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    def _extract_mae(res: dict) -> tuple[float | None, float | None]:
        metrics = res.get("experiment", {}).get("metrics", {})
        val = metrics.get("validation", {}).get("mae")
        test = metrics.get("test", {}).get("mae")
        return val, test

    val_a, test_a = _extract_mae(res_a)
    val_b, test_b = _extract_mae(res_b)
    print("Summary")
    print(f"A train_start={train_a} val_mae={val_a} test_mae={test_a}")
    print(f"B train_start={train_b} val_mae={val_b} test_mae={test_b}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
