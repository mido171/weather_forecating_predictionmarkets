from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_beastmode_signal_discovery import (  # noqa: E402
    CONFIRMATION_START,
    RESEARCH_ROOT,
    markdown_table,
    require_no_confirmation_dates,
    score_prediction_frame,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_official_anchor_expert_blend_screen import (  # noqa: E402
    build_expert_frame,
    load_top_candidate_predictions,
    past_only_expert_blend,
)

FOLDER_NAME = "0023_composite_expert_stack"
REGIME_DIR = RESEARCH_ROOT / "0020_regime_experts" / "artifacts"
FAILURE_DIR = RESEARCH_ROOT / "0022_failure_specialists" / "artifacts"
EXTRA_TOP_N = 8
MIN_HISTORIES = (60, 120, 240)


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_optional_expert_family(
    *,
    prefix: str,
    directory: Path,
    scoreboard_name: str,
    predictions_name: str,
    top_n: int = EXTRA_TOP_N,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scoreboard_path = directory / scoreboard_name
    predictions_path = directory / predictions_name
    if not scoreboard_path.exists() or not predictions_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    return load_top_candidate_predictions(
        scoreboard_path=scoreboard_path,
        predictions_path=predictions_path,
        prefix=prefix,
        top_n=top_n,
    )


def merge_extra_experts(frame: pd.DataFrame, expert_long: pd.DataFrame) -> pd.DataFrame:
    if expert_long.empty:
        return frame
    required = {"target_date", "expert_id", "candidate_prediction_c"}
    missing = required.difference(expert_long.columns)
    if missing:
        raise ValueError(f"Extra expert predictions missing columns: {sorted(missing)}")
    long = expert_long.copy()
    long["target_date"] = pd.to_datetime(long["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(long["target_date"], context="composite extra expert predictions")
    wide = (
        long.pivot_table(index="target_date", columns="expert_id", values="candidate_prediction_c", aggfunc="last")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return frame.merge(wide, on="target_date", how="left").sort_values("target_date").reset_index(drop=True)


def build_composite_expert_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame, mapping = build_expert_frame()
    require_no_confirmation_dates(frame["target_date"], context="composite base expert frame")

    extra_longs: list[pd.DataFrame] = []
    extra_mappings: list[pd.DataFrame] = []
    for prefix, directory, scoreboard_name, predictions_name in [
        ("regime", REGIME_DIR, "regime_scoreboard.csv", "top_regime_predictions.csv"),
        ("failure", FAILURE_DIR, "specialist_scoreboard.csv", "top_specialist_predictions.csv"),
    ]:
        long, extra_mapping = load_optional_expert_family(
            prefix=prefix,
            directory=directory,
            scoreboard_name=scoreboard_name,
            predictions_name=predictions_name,
        )
        if not long.empty:
            extra_longs.append(long)
            extra_mappings.append(extra_mapping)

    if extra_longs:
        frame = merge_extra_experts(frame, pd.concat(extra_longs, ignore_index=True))
    if extra_mappings:
        mapping = pd.concat([mapping, *extra_mappings], ignore_index=True)
    return frame, mapping.sort_values(["source_family", "rank", "expert_id"]).reset_index(drop=True)


def score_blend_predictions(blend_predictions: pd.DataFrame, candidate_id: str, *, mode: str, same_source: bool, min_history: int) -> dict[str, object]:
    candidate = score_prediction_frame(blend_predictions.rename(columns={"expert_prediction_c": "prediction"}), "prediction")
    official = score_prediction_frame(blend_predictions.rename(columns={"official_raw": "official_prediction"}), "official_prediction")
    return {
        "candidate_id": candidate_id,
        "mode": mode,
        "same_source": same_source,
        "min_history": min_history,
        **candidate,
        "official_same_rows_mae": official["mae"],
        "official_same_rows_rmse": official["rmse"],
        "delta_vs_official_same_rows": float(candidate["mae"] - official["mae"]),
        "fallback_rows": int(blend_predictions["selected_expert"].eq("official_raw_fallback").sum()),
    }


def run_blend_grid(
    frame: pd.DataFrame,
    *,
    experts: list[str],
    min_histories: tuple[int, ...] = MIN_HISTORIES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for min_history in min_histories:
        for mode in ("best", "inverse_mae"):
            for same_source in (False, True):
                candidate_id = f"composite_{mode}_{'same_source' if same_source else 'all_prior'}_min{min_history}"
                blend_predictions = past_only_expert_blend(
                    frame,
                    experts=experts,
                    mode=mode,
                    same_source=same_source,
                    min_history=min_history,
                )
                blend_predictions["candidate_id"] = candidate_id
                score_rows.append(
                    score_blend_predictions(
                        blend_predictions,
                        candidate_id,
                        mode=mode,
                        same_source=same_source,
                        min_history=min_history,
                    )
                )
                prediction_rows.append(blend_predictions)
    scoreboard = pd.DataFrame(score_rows).sort_values(["delta_vs_official_same_rows", "mae"]).reset_index(drop=True)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    return scoreboard, predictions


def expert_inventory(mapping: pd.DataFrame) -> pd.DataFrame:
    if mapping.empty:
        return pd.DataFrame(columns=["source_family", "experts", "best_prior_candidate_mae"])
    grouped = (
        mapping.groupby("source_family", observed=True)
        .agg(
            experts=("expert_id", "count"),
            best_prior_candidate_mae=("candidate_mae", "min"),
            median_prior_candidate_mae=("candidate_mae", "median"),
        )
        .reset_index()
    )
    grouped["best_prior_candidate_mae"] = grouped["best_prior_candidate_mae"].where(
        grouped["best_prior_candidate_mae"].notna(),
        math.nan,
    )
    return grouped.sort_values(["source_family"]).reset_index(drop=True)


def write_outputs(
    *,
    frame: pd.DataFrame,
    mapping: pd.DataFrame,
    scoreboard: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, object]:
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    inventory = expert_inventory(mapping)
    write_csv(artifacts / "expert_mapping.csv", mapping)
    write_csv(artifacts / "expert_inventory.csv", inventory)
    write_csv(artifacts / "blend_scoreboard.csv", scoreboard)
    write_csv(artifacts / "blend_predictions.csv", predictions)

    best = scoreboard.iloc[0] if not scoreboard.empty else None
    manifest = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "official_rows": int(len(frame)),
        "first_target_date": str(frame["target_date"].min().date()),
        "last_target_date": str(frame["target_date"].max().date()),
        "expert_count": int(len(mapping)),
        "blend_candidates": int(len(scoreboard)),
        "best_candidate": "" if best is None else str(best["candidate_id"]),
        "best_mae": None if best is None else float(best["mae"]),
        "best_rmse": None if best is None else float(best["rmse"]),
        "best_delta_vs_official": None if best is None else float(best["delta_vs_official_same_rows"]),
        "confirmation_start": str(CONFIRMATION_START.date()),
    }
    write_json(RESEARCH_ROOT / "composite_expert_stack_manifest.json", manifest)

    best_text = "No composite blend candidate was scoreable."
    if best is not None:
        best_text = (
            f"Best composite candidate: `{best['candidate_id']}` with MAE `{best['mae']:.4f}`, "
            f"RMSE `{best['rmse']:.4f}`, and same-row official delta "
            f"`{best['delta_vs_official_same_rows']:.4f}`."
        )

    readme = f"""# Composite Official-Anchor Expert Stack

Generated: `{manifest['generated_at_utc']}`

## What Was Tested

This insight tests whether the previous expert families compound when blended together. It combines:

- official raw forecast anchor;
- top past-only bucket/ladder experts from `0016`;
- top past-only analog experts from `0017`;
- top regime specialists from `0020`;
- top fold-local failure-mode specialists from `0022`.

The screen then reruns prior-performance expert selection and inverse-MAE blending using minimum history gates of `60`, `120`, and `240` prior realized rows.

## Leakage Contract

- All rows are earlier than `{CONFIRMATION_START.date()}`.
- Every input expert prediction already uses prior-only correction logic.
- The composite layer does not fit on future rows; it chooses or weights experts using only prior realized expert error.
- Same-date rows are not used for the current prediction.
- No 2024+ confirmation labels are touched.

## Main Result

{best_text}

## Expert Inventory

{markdown_table(inventory, max_rows=20)}

## Blend Scoreboard

{markdown_table(scoreboard, max_rows=20)}

## Expert Mapping

{markdown_table(mapping, max_rows=30)}

## Interpretation

This is the first direct test of whether the newer regime/failure specialists add incremental value on top of the stronger ladder-plus-analog expert stack from `0018`. If the composite result does not beat `0018`, the evidence says the current specialist definitions are not strong enough yet and that the bigger bottleneck remains continuous official forecast coverage plus richer local residual models.
"""
    write_text(folder / "README.md", readme)
    update_master_index(manifest)
    return manifest


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Composite Official-Anchor Expert Stack\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_composite_expert_stack.py`:

- `{FOLDER_NAME}`: composite prior-performance blend across official raw, ladder, analog, regime, and failure-mode experts.

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Expert count | {manifest['expert_count']} |
| Blend candidates | {manifest['blend_candidates']} |
| Best candidate | {manifest['best_candidate']} |
| Best MAE | {manifest['best_mae']} |
| Best RMSE | {manifest['best_rmse']} |
| Best delta vs official | {manifest['best_delta_vs_official']} |

Leakage contract: all rows are before `{CONFIRMATION_START.date()}`; composite expert selection and weights use strictly prior realized expert error.
"""
    write_text(index_path, text)


def run() -> dict[str, object]:
    frame, mapping = build_composite_expert_frame()
    experts = ["official_raw", *[column for column in frame.columns if column not in {"target_date", "forecast_source_family", "target_tmax_c", "forecast_max_c", "official_raw"}]]
    scoreboard, predictions = run_blend_grid(frame, experts=experts)
    return write_outputs(frame=frame, mapping=mapping, scoreboard=scoreboard, predictions=predictions)


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 composite expert-stack screen.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
