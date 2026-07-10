from __future__ import annotations

import argparse
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPO_ROOT / "experiments"
LADDER_DIR = RESEARCH_ROOT / "0016_past_only_official_anchor_correction_ladder" / "artifacts"
ANALOG_DIR = RESEARCH_ROOT / "0017_past_only_official_residual_analog_screen" / "artifacts"
CONFIRMATION_START = pd.Timestamp("2024-01-01")
TOP_PER_FAMILY = 8
MIN_HISTORY = 120


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def markdown_table(frame: pd.DataFrame, *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    clipped = frame.head(max_rows).copy()
    columns = [str(col) for col in clipped.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in clipped.itertuples(index=False, name=None):
        cells = ["" if pd.isna(value) else str(value) for value in row]
        lines.append("| " + " | ".join(cell.replace("|", "\\|").replace("\n", " ") for cell in cells) + " |")
    return "\n".join(lines)


def require_no_confirmation_dates(dates: pd.Series, *, context: str) -> None:
    normalized = pd.to_datetime(dates, errors="coerce").dt.normalize()
    bad = normalized[normalized >= CONFIRMATION_START]
    if not bad.empty:
        examples = ", ".join(str(value.date()) for value in bad.head(10))
        raise RuntimeError(f"{context} attempted to use confirmation dates >= 2024-01-01: {examples}")


def safe_expert_label(prefix: str, rank: int, candidate_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", candidate_id).strip("_").lower()
    return f"{prefix}_{rank:02d}_{slug[:48]}"


def load_official_rows() -> pd.DataFrame:
    path = LADDER_DIR / "official_rows_used.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing official rows artifact: {path}")
    rows = pd.read_csv(path)
    rows["target_date"] = pd.to_datetime(rows["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(rows["target_date"], context="expert blend official rows")
    keep = ["target_date", "forecast_source_family", "target_tmax_c", "forecast_max_c"]
    rows = rows[keep].drop_duplicates("target_date", keep="last").sort_values("target_date").reset_index(drop=True)
    rows["official_raw"] = pd.to_numeric(rows["forecast_max_c"], errors="coerce")
    return rows


def load_top_candidate_predictions(
    *,
    scoreboard_path: Path,
    predictions_path: Path,
    prefix: str,
    top_n: int = TOP_PER_FAMILY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scoreboard = pd.read_csv(
        scoreboard_path,
        usecols=["candidate_id", "mae", "delta_vs_official_same_rows"],
    ).head(top_n).copy()
    predictions = pd.read_csv(
        predictions_path,
        usecols=["target_date", "candidate_id", "candidate_prediction_c"],
    )
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(predictions["target_date"], context=f"{prefix} expert predictions")

    mapping_rows: list[dict[str, object]] = []
    long_rows: list[pd.DataFrame] = []
    for rank, row in enumerate(scoreboard.itertuples(index=False), start=1):
        candidate_id = str(row.candidate_id)
        expert_id = safe_expert_label(prefix, rank, candidate_id)
        mapping_rows.append(
            {
                "expert_id": expert_id,
                "candidate_id": candidate_id,
                "source_family": prefix,
                "rank": rank,
                "candidate_mae": float(row.mae),
                "delta_vs_official_same_rows": float(row.delta_vs_official_same_rows),
            }
        )
        subset = predictions[predictions["candidate_id"].eq(candidate_id)][
            ["target_date", "candidate_prediction_c"]
        ].copy()
        subset["expert_id"] = expert_id
        long_rows.append(subset)

    long = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    mapping = pd.DataFrame(mapping_rows)
    return long, mapping


def build_expert_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    official = load_official_rows()
    expert_longs: list[pd.DataFrame] = []
    mappings: list[pd.DataFrame] = [
        pd.DataFrame(
            [
                {
                    "expert_id": "official_raw",
                    "candidate_id": "official_raw",
                    "source_family": "official",
                    "rank": 0,
                    "candidate_mae": math.nan,
                    "delta_vs_official_same_rows": 0.0,
                }
            ]
        )
    ]
    for prefix, directory in [("ladder", LADDER_DIR), ("analog", ANALOG_DIR)]:
        long, mapping = load_top_candidate_predictions(
            scoreboard_path=directory / "scoreboard.csv",
            predictions_path=directory / "all_candidate_predictions.csv",
            prefix=prefix,
        )
        expert_longs.append(long)
        mappings.append(mapping)

    frame = official.copy()
    if expert_longs:
        long_all = pd.concat(expert_longs, ignore_index=True)
        wide = (
            long_all.pivot_table(
                index="target_date",
                columns="expert_id",
                values="candidate_prediction_c",
                aggfunc="last",
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
        frame = frame.merge(wide, on="target_date", how="left")
    mapping = pd.concat(mappings, ignore_index=True)
    return frame.sort_values("target_date").reset_index(drop=True), mapping


def score_prediction(frame: pd.DataFrame, prediction_col: str) -> dict[str, object]:
    scored = frame[["target_date", "target_tmax_c", prediction_col]].dropna().copy()
    if scored.empty:
        return {"n": 0, "first_date": "", "last_date": "", "mae": math.nan, "rmse": math.nan, "bias": math.nan}
    error = pd.to_numeric(scored[prediction_col], errors="coerce") - pd.to_numeric(scored["target_tmax_c"], errors="coerce")
    return {
        "n": int(len(scored)),
        "first_date": str(scored["target_date"].min().date()),
        "last_date": str(scored["target_date"].max().date()),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(error.mean()),
    }


def prior_mae(
    prior: pd.DataFrame,
    *,
    expert: str,
    source: str | None,
) -> tuple[int, float]:
    work = prior if source is None else prior[prior["forecast_source_family"].eq(source)]
    work = work.dropna(subset=[expert, "target_tmax_c"])
    if work.empty:
        return 0, math.nan
    error = pd.to_numeric(work[expert], errors="coerce") - pd.to_numeric(work["target_tmax_c"], errors="coerce")
    return int(len(work)), float(error.abs().mean())


def past_only_expert_blend(
    frame: pd.DataFrame,
    *,
    experts: list[str],
    mode: str,
    same_source: bool,
    min_history: int = MIN_HISTORY,
) -> pd.DataFrame:
    ordered = frame.sort_values("target_date").reset_index(drop=True).copy()
    target = pd.to_numeric(ordered["target_tmax_c"], errors="coerce").to_numpy(dtype=float)
    official = pd.to_numeric(ordered["official_raw"], errors="coerce").to_numpy(dtype=float)
    expert_values = {
        expert: pd.to_numeric(ordered[expert], errors="coerce").to_numpy(dtype=float)
        for expert in experts
    }
    prior_counts: dict[str, np.ndarray] = {}
    prior_maes: dict[str, np.ndarray] = {}
    source_groups = list(ordered.groupby("forecast_source_family", sort=False).indices.values()) if same_source else []

    for expert, values in expert_values.items():
        valid = np.isfinite(values) & np.isfinite(target)
        valid_int = valid.astype(float)
        abs_error = np.where(valid, np.abs(values - target), 0.0)
        counts = np.zeros(len(ordered), dtype=float)
        sums = np.zeros(len(ordered), dtype=float)
        if same_source:
            for group_index in source_groups:
                index_array = np.asarray(group_index, dtype=int)
                group_valid = valid_int[index_array]
                group_error = abs_error[index_array]
                counts[index_array] = np.cumsum(group_valid) - group_valid
                sums[index_array] = np.cumsum(group_error) - group_error
        else:
            counts = np.cumsum(valid_int) - valid_int
            sums = np.cumsum(abs_error) - abs_error
        prior_counts[expert] = counts
        prior_maes[expert] = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)

    predictions: list[float] = []
    selected: list[str] = []
    eligible_counts: list[int] = []
    for index in range(len(ordered)):
        current_available = [expert for expert in experts if np.isfinite(expert_values[expert][index])]
        if not current_available:
            predictions.append(math.nan)
            selected.append("")
            eligible_counts.append(0)
            continue
        scored: list[tuple[str, int, float]] = []
        for expert in current_available:
            n = int(prior_counts[expert][index])
            mae = float(prior_maes[expert][index])
            if n >= min_history and np.isfinite(mae):
                scored.append((expert, n, mae))
        if not scored:
            predictions.append(float(official[index]) if np.isfinite(official[index]) else math.nan)
            selected.append("official_raw_fallback")
            eligible_counts.append(0)
            continue

        eligible_counts.append(len(scored))
        if mode == "best":
            best = min(scored, key=lambda item: (item[2], item[0]))
            predictions.append(float(expert_values[best[0]][index]))
            selected.append(best[0])
        elif mode == "inverse_mae":
            weights = np.array([1.0 / max(item[2], 1e-6) for item in scored], dtype=float)
            values = np.array([float(expert_values[item[0]][index]) for item in scored], dtype=float)
            weights = weights / weights.sum()
            predictions.append(float(np.dot(weights, values)))
            selected.append("inverse_mae_blend")
        else:
            raise ValueError(f"Unknown mode: {mode}")

    out = ordered[["target_date", "forecast_source_family", "target_tmax_c", "official_raw"]].copy()
    out["expert_prediction_c"] = predictions
    out["selected_expert"] = selected
    out["eligible_expert_count"] = eligible_counts
    out["mode"] = mode
    out["same_source"] = same_source
    return out


def run_screen() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame, mapping = build_expert_frame()
    experts = ["official_raw", *[col for col in frame.columns if col.startswith("ladder_") or col.startswith("analog_")]]
    score_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for mode in ("best", "inverse_mae"):
        for same_source in (False, True):
            candidate_id = f"{mode}_{'same_source' if same_source else 'all_prior'}"
            predictions = past_only_expert_blend(frame, experts=experts, mode=mode, same_source=same_source)
            predictions["candidate_id"] = candidate_id
            candidate = score_prediction(predictions, "expert_prediction_c")
            official = score_prediction(predictions.rename(columns={"official_raw": "official_prediction_c"}), "official_prediction_c")
            score_rows.append(
                {
                    "candidate_id": candidate_id,
                    "mode": mode,
                    "same_source": same_source,
                    **candidate,
                    "official_same_rows_mae": official["mae"],
                    "official_same_rows_rmse": official["rmse"],
                    "delta_vs_official_same_rows": float(candidate["mae"] - official["mae"]),
                    "fallback_rows": int(predictions["selected_expert"].eq("official_raw_fallback").sum()),
                }
            )
            prediction_rows.append(predictions)

    scoreboard = pd.DataFrame(score_rows).sort_values(["delta_vs_official_same_rows", "mae"]).reset_index(drop=True)
    all_predictions = pd.concat(prediction_rows, ignore_index=True)
    return scoreboard, all_predictions, mapping


def write_outputs(scoreboard: pd.DataFrame, predictions: pd.DataFrame, mapping: pd.DataFrame) -> None:
    folder = RESEARCH_ROOT / "0018_past_only_official_expert_blend_screen"
    artifacts = folder / "artifacts"
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "predictions.csv", predictions)
    write_csv(artifacts / "expert_mapping.csv", mapping)

    selection_counts = (
        predictions.groupby(["candidate_id", "selected_expert"], dropna=False, observed=True)
        .agg(rows=("target_date", "count"))
        .reset_index()
        .sort_values(["candidate_id", "rows"], ascending=[True, False])
    )
    write_csv(artifacts / "selection_counts.csv", selection_counts)

    best = scoreboard.iloc[0] if not scoreboard.empty else None
    best_text = "No scoreable expert blend was produced."
    if best is not None:
        best_text = (
            f"Best expert blend: `{best['candidate_id']}` with MAE `{best['mae']:.4f}` "
            f"versus same-row official MAE `{best['official_same_rows_mae']:.4f}` "
            f"(delta `{best['delta_vs_official_same_rows']:.4f}`)."
        )

    text = f"""# Past-Only Official Expert Blend Screen

Generated: `{now_utc()}`

## What Was Tested

This insight combines the official raw forecast, the best past-only bucket-correction experts, and the best past-only analog experts. For every target date, expert choice or expert weights are based only on rows earlier than the target date.

## Leakage Contract

- Every upstream correction expert already uses only earlier rows.
- This blend layer also uses only earlier rows to estimate expert MAE.
- Target labels from `{CONFIRMATION_START.date()}` onward are blocked upstream.
- Fixed oracle expert selection is not used.
- This is a research screen, not an accepted production model.

## Main Result

{best_text}

## Scoreboard

{markdown_table(scoreboard, max_rows=20)}

## Expert Map

{markdown_table(mapping, max_rows=25)}

## Selection Counts

{markdown_table(selection_counts, max_rows=30)}

## Interpretation

If this screen does not materially beat the best individual analog expert, the next lift probably requires better forecast archive continuity and richer regime-specific expert definitions, not just blending the same weak corrections.
"""
    write_text(folder / "README.md", text)


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Official Expert Blend Screen\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_official_anchor_expert_blend_screen.py`:

- `0018_past_only_official_expert_blend_screen`: prior-performance selection/blending across official raw, bucket-correction, and analog experts.

| Metric | Value |
|---|---:|
| Candidate blend rows | {manifest['candidate_rows']} |
| Prediction rows | {manifest['prediction_rows']} |
| Expert count | {manifest['expert_count']} |
| Best candidate MAE | {manifest['best_mae']} |
| Best delta vs same-row official | {manifest['best_delta_vs_official']} |

Leakage contract: blend selection and blend weights are estimated only from rows earlier than each target date.
"""
    write_text(index_path, text)


def run() -> dict[str, object]:
    scoreboard, predictions, mapping = run_screen()
    write_outputs(scoreboard, predictions, mapping)
    best = scoreboard.iloc[0] if not scoreboard.empty else None
    manifest = {
        "generated_at_utc": now_utc(),
        "candidate_rows": int(len(scoreboard)),
        "prediction_rows": int(len(predictions)),
        "expert_count": int(len(mapping)),
        "best_candidate_id": "" if best is None else str(best["candidate_id"]),
        "best_mae": None if best is None else float(best["mae"]),
        "best_delta_vs_official": None if best is None else float(best["delta_vs_official_same_rows"]),
        "folder": "0018_past_only_official_expert_blend_screen",
    }
    write_json(RESEARCH_ROOT / "official_expert_blend_screen_manifest.json", manifest)
    update_master_index(manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run past-only official expert blend screen.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
