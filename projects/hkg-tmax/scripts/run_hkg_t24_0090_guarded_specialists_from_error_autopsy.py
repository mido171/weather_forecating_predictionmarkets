from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_0083_expanded_frame_official_anchor_replay import date_text, score_arrays
from scripts.run_hkg_t24_0085_long_history_feature_station_residual_bridge import FEATURE_MATRIX_PATH
from scripts.run_hkg_t24_0086_guarded_long_history_residual_specialists import (
    SpecialistSpec,
    apply_specialist,
    assign_bucket,
    pre2000_thresholds,
)
from scripts.run_hkg_t24_beastmode_signal_discovery import (
    CONFIRMATION_START,
    RESEARCH_ROOT,
    markdown_table,
    require_no_confirmation_dates,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import (
    classify_feature_family,
    station_ids_in_feature,
    update_markdown_section,
)

FOLDER_NAME = "0090_guarded_specialists_from_error_autopsy"
INPUT_0088_TOP_PATH = (
    RESEARCH_ROOT / "0088_prior_gated_specialist_stack" / "artifacts" / "top_predictions.csv"
)
INPUT_0089_LEADS_PATH = (
    RESEARCH_ROOT / "0089_remaining_error_regime_autopsy" / "artifacts" / "next_specialist_leads.csv"
)
BASE_ID = "0088_0087_interaction_champion"
MAX_LEADS = 14


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def safe_token(value: str) -> str:
    return (
        value.replace("_", "-")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "-")
        .replace(".", "p")[:72]
    )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [path for path in (FEATURE_MATRIX_PATH, INPUT_0088_TOP_PATH, INPUT_0089_LEADS_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"0090 requires 0088, 0089, and feature artifacts first: {missing}")

    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    features = features.loc[:, ~features.columns.duplicated()].copy()
    features["target_date"] = pd.to_datetime(features["target_date"], errors="coerce").dt.normalize()
    features = features[features["target_date"].notna() & (features["target_date"] < CONFIRMATION_START)].copy()

    base = pd.read_csv(INPUT_0088_TOP_PATH)
    base["target_date"] = pd.to_datetime(base["target_date"], errors="coerce").dt.normalize()
    base = base[base["target_date"].notna() & (base["target_date"] < CONFIRMATION_START)].copy()
    required = {
        "target_date",
        "forecast_source_family",
        "target_tmax_c",
        "forecast_max_c",
        "candidate_prediction_c",
        "season",
        "frame_segment",
        "era_bucket",
    }
    missing_cols = required.difference(base.columns)
    if missing_cols:
        raise RuntimeError(f"0088 top predictions missing required columns: {sorted(missing_cols)}")
    for column in ("target_tmax_c", "forecast_max_c", "candidate_prediction_c"):
        base[column] = pd.to_numeric(base[column], errors="coerce")
    base = base[base[["target_tmax_c", "forecast_max_c", "candidate_prediction_c"]].notna().all(axis=1)].copy()
    for column in ("forecast_source_family", "season", "frame_segment", "era_bucket"):
        base[column] = base[column].astype(str)

    leads = pd.read_csv(INPUT_0089_LEADS_PATH)
    leads = leads[leads["feature"].isin(features.columns)].copy()
    leads["lead_rank"] = pd.to_numeric(leads["lead_rank"], errors="coerce")
    leads = leads.sort_values(["lead_rank", "contrast_priority"]).head(MAX_LEADS).reset_index(drop=True)

    require_no_confirmation_dates(features["target_date"], context="0090 feature matrix")
    require_no_confirmation_dates(base["target_date"], context="0090 0088 prediction input")
    return features, base, leads


def build_working_frame(
    features: pd.DataFrame,
    base: pd.DataFrame,
    leads: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_names = leads["feature"].astype(str).tolist()
    joined = base.merge(features[["target_date", *feature_names]], on="target_date", how="left")
    joined["base_residual_c"] = joined["candidate_prediction_c"] - joined["target_tmax_c"]
    threshold_rows: list[dict[str, object]] = []
    for lead in leads.to_dict("records"):
        feature = str(lead["feature"])
        thresholds = pre2000_thresholds(features, feature)
        if thresholds is None:
            joined[f"{feature}__bucket"] = np.nan
            continue
        joined[f"{feature}__bucket"] = assign_bucket(joined[feature], thresholds)
        threshold_rows.append(
            {
                "lead_rank": lead.get("lead_rank", ""),
                "feature": feature,
                "family": classify_feature_family(feature),
                "station_ids": station_ids_in_feature(feature),
                "contrast_priority": lead.get("contrast_priority", ""),
                "recommended_action": lead.get("recommended_action", ""),
                "thresholds": ";".join(f"{value:.6g}" for value in thresholds),
                "threshold_count": int(len(thresholds)),
                "bucketed_rows": int(joined[f"{feature}__bucket"].notna().sum()),
            }
        )
    require_no_confirmation_dates(joined["target_date"], context="0090 joined frame")
    return joined.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True), pd.DataFrame(threshold_rows)


def make_specs(leads: pd.DataFrame, thresholds: pd.DataFrame) -> list[SpecialistSpec]:
    features_with_thresholds = set(thresholds["feature"].astype(str).tolist())
    specs: list[SpecialistSpec] = []
    for lead in leads.to_dict("records"):
        feature = str(lead["feature"])
        if feature not in features_with_thresholds:
            continue
        family = classify_feature_family(feature)
        cap = 0.30 if family == "calendar_climatology" else 0.55
        shrink = 150.0 if family == "calendar_climatology" else 100.0
        lead_rank = int(float(lead.get("lead_rank", 999)))
        context_modes = (
            ("feature", "source_frame_feature", "source_season_feature")
            if lead_rank <= 8
            else ("feature", "source_season_feature")
        )
        for context_mode in context_modes:
            min_history = 120
            specs.append(
                SpecialistSpec(
                    candidate_id=(
                        f"autopsy_{safe_token(feature)}_{context_mode}_m{min_history}_"
                        f"cap{str(cap).replace('.', 'p')}"
                    ),
                    feature=feature,
                    context_mode=context_mode,
                    min_history=min_history,
                    shrink_rows=shrink,
                    correction_cap_c=cap,
                )
            )
    return specs


def evaluation_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    source = frame["forecast_source_family"].astype(str)
    segment = frame["frame_segment"].astype(str)
    season = frame["season"].astype(str)
    mask_map: dict[str, np.ndarray] = {
        "old_frame_": segment.eq("current_0081_frame").to_numpy(dtype=bool),
        "newly_available_": segment.eq("newly_available_official_frame").to_numpy(dtype=bool),
        "press_": source.eq("press_archive").to_numpy(dtype=bool),
        "rss_": source.eq("rss_archive").to_numpy(dtype=bool),
    }
    for season_name in sorted(season.dropna().unique().tolist()):
        safe_name = safe_token(str(season_name)).replace("-", "_")
        mask_map[f"season_{safe_name}_"] = season.eq(season_name).to_numpy(dtype=bool)
    return mask_map


def score_candidate(
    frame: pd.DataFrame,
    *,
    candidate_id: str,
    candidate_class: str,
    prediction: np.ndarray,
    mask_map: dict[str, np.ndarray],
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    target = frame["target_tmax_c"].to_numpy(dtype=float)
    base = frame["candidate_prediction_c"].to_numpy(dtype=float)
    raw = frame["forecast_max_c"].to_numpy(dtype=float)
    dates = pd.to_datetime(frame["target_date"], errors="coerce")
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "candidate_class": candidate_class,
        **score_arrays(target=target, prediction=prediction, dates=dates),
    }
    base_score = score_arrays(target=target, prediction=base, dates=dates)
    raw_score = score_arrays(target=target, prediction=raw, dates=dates)
    row["delta_mae_vs_0088_base"] = float(row["mae"]) - float(base_score["mae"])
    row["delta_mae_vs_official_raw"] = float(row["mae"]) - float(raw_score["mae"])
    delta_keys: list[str] = ["delta_mae_vs_0088_base"]
    for prefix, mask in mask_map.items():
        score = score_arrays(target=target[mask], prediction=prediction[mask], dates=dates[mask], prefix=prefix)
        base_segment = score_arrays(target=target[mask], prediction=base[mask], dates=dates[mask], prefix=prefix)
        row.update(score)
        delta_key = f"{prefix}delta_mae_vs_0088_base"
        row[delta_key] = float(score[f"{prefix}mae"]) - float(base_segment[f"{prefix}mae"])
        delta_keys.append(delta_key)
    if extra:
        row.update(extra)
    row["hardened_gate_passed"] = (
        float(row["delta_mae_vs_0088_base"]) < 0.0
        and all(float(row[key]) <= 0.0 for key in delta_keys[1:])
    )
    row["season_no_regression_passed"] = all(
        float(row[key]) <= 0.0 for key in delta_keys if key.startswith("season_")
    )
    return row


def build_outputs(
    frame: pd.DataFrame,
    leads: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mask_map = evaluation_masks(frame)
    raw = frame["forecast_max_c"].to_numpy(dtype=float)
    base = frame["candidate_prediction_c"].to_numpy(dtype=float)
    rows = [
        score_candidate(
            frame,
            candidate_id="official_raw",
            candidate_class="official_raw",
            prediction=raw,
            mask_map=mask_map,
        ),
        score_candidate(
            frame,
            candidate_id=BASE_ID,
            candidate_class="0088_base",
            prediction=base,
            mask_map=mask_map,
        ),
    ]
    definitions: list[dict[str, object]] = []
    predictions = {"official_raw": raw, BASE_ID: base}
    diagnostics_by_id: dict[str, pd.DataFrame] = {}
    lead_lookup = leads.set_index("feature").to_dict("index") if not leads.empty else {}
    for spec in make_specs(leads, thresholds):
        prediction, diagnostics = apply_specialist(frame, spec)
        predictions[spec.candidate_id] = prediction
        diagnostics_by_id[spec.candidate_id] = diagnostics
        lead = lead_lookup.get(spec.feature, {})
        definitions.append(
            {
                "candidate_id": spec.candidate_id,
                "candidate_class": "0090_autopsy_feature_specialist",
                "feature": spec.feature,
                "family": classify_feature_family(spec.feature),
                "station_ids": station_ids_in_feature(spec.feature),
                "lead_rank": lead.get("lead_rank", ""),
                "context_mode": spec.context_mode,
                "min_history": spec.min_history,
                "shrink_rows": spec.shrink_rows,
                "correction_cap_c": spec.correction_cap_c,
                "recommended_action": lead.get("recommended_action", ""),
            }
        )
        rows.append(
            score_candidate(
                frame,
                candidate_id=spec.candidate_id,
                candidate_class="0090_autopsy_feature_specialist",
                prediction=prediction,
                mask_map=mask_map,
                extra={
                    "feature": spec.feature,
                    "family": classify_feature_family(spec.feature),
                    "station_ids": station_ids_in_feature(spec.feature),
                    "lead_rank": lead.get("lead_rank", ""),
                    "context_mode": spec.context_mode,
                    "min_history": spec.min_history,
                    "shrink_rows": spec.shrink_rows,
                    "correction_cap_c": spec.correction_cap_c,
                },
            )
        )
    scoreboard = pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    hardened = scoreboard[scoreboard["hardened_gate_passed"].astype(bool)].copy()
    if hardened.empty:
        best_id = BASE_ID
    else:
        best_id = str(hardened.sort_values(["mae", "rmse"]).iloc[0]["candidate_id"])
    best_prediction = predictions[best_id]
    diagnostics = diagnostics_by_id.get(best_id, pd.DataFrame())
    top_predictions = frame[
        [
            "target_date",
            "forecast_source_family",
            "target_tmax_c",
            "forecast_max_c",
            "season",
            "frame_segment",
            "era_bucket",
        ]
    ].copy()
    top_predictions["candidate_id"] = best_id
    top_predictions["candidate_prediction_c"] = best_prediction
    top_predictions["candidate_error_c"] = best_prediction - top_predictions["target_tmax_c"]
    return scoreboard, pd.DataFrame(definitions), diagnostics, top_predictions


def build_summary(
    *,
    generated_at: str,
    frame: pd.DataFrame,
    leads: pd.DataFrame,
    thresholds: pd.DataFrame,
    scoreboard: pd.DataFrame,
) -> dict[str, object]:
    dates = pd.to_datetime(frame["target_date"], errors="coerce")
    base = scoreboard[scoreboard["candidate_id"].eq(BASE_ID)].iloc[0].to_dict()
    hardened = scoreboard[scoreboard["hardened_gate_passed"].astype(bool)].copy()
    if hardened.empty:
        best = base
        best_hardened = {}
    else:
        best = hardened.sort_values(["mae", "rmse"]).iloc[0].to_dict()
        best_hardened = best
    return {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "rows": int(len(frame)),
        "first_target_date": date_text(dates.min()),
        "last_target_date": date_text(dates.max()),
        "lead_count": int(len(leads)),
        "threshold_feature_count": int(len(thresholds)),
        "candidate_count": int(len(scoreboard)),
        "hardened_candidate_count": int(len(hardened)),
        "base_0088_mae": float(base["mae"]),
        "base_0088_rmse": float(base["rmse"]),
        "best_candidate": str(best["candidate_id"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_delta_mae_vs_0088_base": float(best["delta_mae_vs_0088_base"]),
        "best_hardened_candidate": best_hardened.get("candidate_id", ""),
        "best_hardened_feature": best_hardened.get("feature", ""),
        "best_hardened_context_mode": best_hardened.get("context_mode", ""),
        "best_hardened_delta_mae_vs_0088_base": best_hardened.get("delta_mae_vs_0088_base"),
        "confirmation_start": CONFIRMATION_START.date().isoformat(),
        "uses_2024_plus_rows": False,
        "status": "guarded_specialists_from_error_autopsy_complete",
        "next_recommended_task": (
            "Run 0091 to inspect the best non-hardened 0090 candidates by failing slice, then design either "
            "narrower source-season specialists or a conservative ensemble constrained by the 0090 no-regression gate."
        ),
    }


def build_readme(
    *,
    generated_at: str,
    summary: dict[str, object],
    leads: pd.DataFrame,
    thresholds: pd.DataFrame,
    definitions: pd.DataFrame,
    scoreboard: pd.DataFrame,
) -> str:
    return f"""# 0090 Guarded Specialists From Error Autopsy

Generated: `{generated_at}`

## Purpose

`0089` identified the remaining high-error regimes and the features that separate the bad tail from the good tail. `0090` converts those leads into actual past-only residual specialists and scores them against the current `0088`/`0087` champion.

The gate is deliberately strict: a candidate must improve full-frame MAE and must not worsen the source slices, frame slices, or any season slice. This protects against a specialist that only helps the bad MAM/new-press regime by quietly damaging another part of the history.

## Main Result

| Field | Value |
|---|---|
| Status | `{summary['status']}` |
| Rows | `{summary['rows']}` |
| Date range | `{summary['first_target_date']}` to `{summary['last_target_date']}` |
| 0088 base MAE | `{summary['base_0088_mae']}` |
| 0088 base RMSE | `{summary['base_0088_rmse']}` |
| Leads tested | `{summary['lead_count']}` |
| Features with pre-2000 thresholds | `{summary['threshold_feature_count']}` |
| Candidate count | `{summary['candidate_count']}` |
| Hardened passing candidates | `{summary['hardened_candidate_count']}` |
| Best candidate | `{summary['best_candidate']}` |
| Best MAE | `{summary['best_mae']}` |
| Best RMSE | `{summary['best_rmse']}` |
| Delta vs 0088 base | `{summary['best_delta_mae_vs_0088_base']}` |
| Best hardened feature | `{summary['best_hardened_feature']}` |
| Best hardened context | `{summary['best_hardened_context_mode']}` |
| 2024+ rows used | `{summary['uses_2024_plus_rows']}` |

## 0089 Leads Used

{markdown_table(leads, max_rows=20)}

## Pre-2000 Thresholds

{markdown_table(thresholds, max_rows=20)}

## Candidate Definitions

{markdown_table(definitions.head(40), max_rows=40)}

## Top Scoreboard Rows

{markdown_table(scoreboard.head(30), max_rows=30)}

## Interpretation

If no candidate passes, the right conclusion is not that the features are useless. It means the simple residual specialist form is not robust enough under source, frame, and season gates. The next step should inspect which slice failed for the best near-misses, then narrow the specialist or use the signal as a guard rather than a direct correction.

## Exact Next Recommended Task

{summary['next_recommended_task']}
"""


def update_milestones(summary: dict[str, object], scoreboard: pd.DataFrame) -> None:
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0090_guarded_specialists_from_error_autopsy.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Current evidence | Status |
|---|---|---|
| Autopsy-derived specialists | `{summary['candidate_count']}` candidates over `{summary['rows']}` rows | Pre-2024 only |
| Leads tested | `{summary['lead_count']}` | From 0089 |
| Thresholded features | `{summary['threshold_feature_count']}` | Pre-2000 buckets |
| Hardened passing candidates | `{summary['hardened_candidate_count']}` | Source/frame/season gate |
| Base 0088 MAE | `{summary['base_0088_mae']}` | Benchmark |
| Best candidate | `{summary['best_candidate']}` | `{summary['best_mae']}` MAE |
| Delta vs 0088 base | `{summary['best_delta_mae_vs_0088_base']}` | Specialist value |
| Leakage | `0` 2024+ rows | PASS |

Top 0090 candidates:

{markdown_table(scoreboard.head(10), max_rows=10)}
"""
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="0090 Guarded Specialists From Error Autopsy",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Exact Next Recommended Codex Task",
        section=(
            "Implement `0091_near_miss_specialist_failure_analysis`: inspect which source/frame/season gates block "
            "the best 0090 near-misses, then design narrower candidates or conservative guard-only use. Keep current "
            "RSS data only until the forecast backfill completes; keep 2024+ sealed."
        ),
    )


def run() -> dict[str, object]:
    generated_at = now_utc()
    features, base, leads = load_inputs()
    frame, thresholds = build_working_frame(features, base, leads)
    scoreboard, definitions, diagnostics, top_predictions = build_outputs(frame, leads, thresholds)
    summary = build_summary(
        generated_at=generated_at,
        frame=frame,
        leads=leads,
        thresholds=thresholds,
        scoreboard=scoreboard,
    )
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    write_csv(artifacts / "selected_0089_leads.csv", leads)
    write_csv(artifacts / "feature_thresholds.csv", thresholds)
    write_csv(artifacts / "candidate_definitions.csv", definitions)
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "best_gate_diagnostics.csv", diagnostics)
    write_csv(artifacts / "top_predictions.csv", top_predictions)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "guarded_specialists_from_error_autopsy_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            generated_at=generated_at,
            summary=summary,
            leads=leads,
            thresholds=thresholds,
            definitions=definitions,
            scoreboard=scoreboard,
        ),
    )
    update_milestones(summary, scoreboard)
    require_no_confirmation_dates(top_predictions["target_date"], context="0090 top predictions")
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Convert 0089 HKG Tmax error-autopsy leads into guarded past-only specialists."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
