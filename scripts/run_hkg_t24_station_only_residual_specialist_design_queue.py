from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_beastmode_signal_discovery import (  # noqa: E402
    RESEARCH_ROOT,
    markdown_table,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import (  # noqa: E402
    update_markdown_section,
)

FOLDER_NAME = "0057_station_only_residual_specialist_design_queue"
ARTIFACT_0056 = RESEARCH_ROOT / "0056_station_only_failure_mode_analysis" / "artifacts"
FAILURE_RANK_PATH = ARTIFACT_0056 / "failure_regime_rank.csv"
FEATURE_CORR_PATH = ARTIFACT_0056 / "feature_error_correlations.csv"
WORST_DAYS_PATH = ARTIFACT_0056 / "worst_days.csv"
THRESHOLDS_PATH = ARTIFACT_0056 / "regime_thresholds.csv"


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required 0056 artifact: {path}")
    return pd.read_csv(path)


def first_match(frame: pd.DataFrame, **equals: object) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=object)
    mask = pd.Series(True, index=frame.index)
    for column, value in equals.items():
        if column not in frame.columns:
            return pd.Series(dtype=object)
        mask &= frame[column].astype(str).eq(str(value))
    matched = frame[mask].copy()
    return matched.iloc[0] if not matched.empty else pd.Series(dtype=object)


def safe_float(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def build_design_queue(failure_rank: pd.DataFrame, feature_corr: pd.DataFrame) -> pd.DataFrame:
    late = first_match(failure_rank, analysis_name="fold", fold_id="fold_2018_2023")
    winter_mid = first_match(
        failure_rank,
        analysis_name="season_x_heat",
        season="DJF",
        heat_bucket_pre2000_target="mid",
    )
    jja_mid = first_match(
        failure_rank,
        analysis_name="season_x_heat",
        season="JJA",
        heat_bucket_pre2000_target="mid",
    )
    feb = first_match(failure_rank, analysis_name="month", month="2.0")
    mam = first_match(failure_rank, analysis_name="season", season="MAM")
    pressure_high = first_match(failure_rank, analysis_name="pressure_pair_spread_bucket", pressure_pair_spread_bucket="high")
    top_abs_feature = feature_corr.iloc[0] if not feature_corr.empty else pd.Series(dtype=object)

    rows = [
        {
            "candidate_id": "late_period_bias_repair",
            "priority": 1,
            "source_evidence": "0056 fold_2018_2023 bias and MAE lift",
            "evidence_n": int(safe_float(late.get("n", 0))),
            "evidence_mae": safe_float(late.get("mae")),
            "evidence_bias": safe_float(late.get("bias")),
            "mechanism_hypothesis": "Station-only static calibration drifts low in the recent period; fold-local trend/intercept repair may reduce systematic cool bias.",
            "deployable_inputs": "target_date year/month, 0054 station feature availability, fold-local prior residual history",
            "diagnostic_inputs_forbidden_in_model": "",
            "leakage_status": "deployable_if_fit_inside_fold",
            "next_test": "Fold-local OOF residual intercept/slope specialist fitted only on prior dates before each scored row; compare to ridge_all_station residuals.",
            "promotion_gate": "Must reduce 2018-2023 MAE without degrading 2000-2017 by more than 0.02 C.",
        },
        {
            "candidate_id": "winter_mid_heat_proxy_specialist",
            "priority": 2,
            "source_evidence": "0056 worst regime DJF + diagnostic mid target heat bucket",
            "evidence_n": int(safe_float(winter_mid.get("n", 0))),
            "evidence_mae": safe_float(winter_mid.get("mae")),
            "evidence_bias": safe_float(winter_mid.get("bias")),
            "mechanism_hypothesis": "Winter moderate-temperature days are often overpredicted by the station-only model; the model may confuse cool-season recovery with warmer regimes.",
            "deployable_inputs": "DJF calendar, pre-2000-threshold station temperature/dew/pressure buckets, latest-before-1500 station levels",
            "diagnostic_inputs_forbidden_in_model": "realized target heat bucket",
            "leakage_status": "requires_proxy_validation",
            "next_test": "Translate diagnostic mid-heat bucket into pre-cutoff station proxy buckets, then test prior-only residual shrinkage.",
            "promotion_gate": "Must improve DJF residual MAE and keep target-label heat buckets out of model features.",
        },
        {
            "candidate_id": "warm_season_mid_proxy_specialist",
            "priority": 3,
            "source_evidence": "0056 JJA + diagnostic mid target heat bucket positive bias",
            "evidence_n": int(safe_float(jja_mid.get("n", 0))),
            "evidence_mae": safe_float(jja_mid.get("mae")),
            "evidence_bias": safe_float(jja_mid.get("bias")),
            "mechanism_hypothesis": "Mid-level summer days are underpredicted by station-only Ridge; maritime/cloud/rain suppression proxies may need separate treatment.",
            "deployable_inputs": "JJA calendar, station temperature trajectory buckets, wind buckets, pressure spread buckets",
            "diagnostic_inputs_forbidden_in_model": "realized target heat bucket",
            "leakage_status": "requires_proxy_validation",
            "next_test": "Build deployable summer mid-heat proxy from station trajectory and wind regimes, then test fold-local residual correction.",
            "promotion_gate": "Must reduce positive JJA-mid bias without adding label-derived heat buckets.",
        },
        {
            "candidate_id": "february_march_transition_specialist",
            "priority": 4,
            "source_evidence": "0056 February and March month buckets",
            "evidence_n": int(safe_float(feb.get("n", 0))),
            "evidence_mae": safe_float(feb.get("mae")),
            "evidence_bias": safe_float(feb.get("bias")),
            "mechanism_hypothesis": "Late-winter to spring transition has large miss frequency; seasonal phase alone is too coarse.",
            "deployable_inputs": "month/day-of-year, station dew trajectory, pressure pair spread, wind regime buckets",
            "diagnostic_inputs_forbidden_in_model": "",
            "leakage_status": "deployable_if_fit_inside_fold",
            "next_test": "Fold-local month-specific residual specialist with smooth day-of-year taper and pressure/dew gates.",
            "promotion_gate": "Must improve Feb-Mar MAE with no deterioration in adjacent Jan/Apr months.",
        },
        {
            "candidate_id": "spring_transition_pressure_dew_specialist",
            "priority": 5,
            "source_evidence": "0056 MAM season and high dew/pressure regime misses",
            "evidence_n": int(safe_float(mam.get("n", 0))),
            "evidence_mae": safe_float(mam.get("mae")),
            "evidence_bias": safe_float(mam.get("bias")),
            "mechanism_hypothesis": "Spring errors may reflect humidity/cloud-front regimes not captured by linear station-only averaging.",
            "deployable_inputs": "MAM calendar, dew_14d_trajectory_bucket, pressure_pair_spread_bucket, wind_station_attribute_bucket",
            "diagnostic_inputs_forbidden_in_model": "",
            "leakage_status": "deployable_if_fit_inside_fold",
            "next_test": "Fold-local smooth residual model inside MAM using dew and pressure buckets from pre-2000 thresholds.",
            "promotion_gate": "Must beat global station-only residual correction on MAM rows in every fold after 2006.",
        },
        {
            "candidate_id": "pressure_high_uncertainty_guard",
            "priority": 6,
            "source_evidence": "0056 pressure-pair high bucket elevated error",
            "evidence_n": int(safe_float(pressure_high.get("n", 0))),
            "evidence_mae": safe_float(pressure_high.get("mae")),
            "evidence_bias": safe_float(pressure_high.get("bias")),
            "mechanism_hypothesis": "High pressure-gradient states carry residual uncertainty; specialist may need lower confidence or alternate residual mean.",
            "deployable_inputs": "pressure_pair_spread_bucket and associated station pressure spreads",
            "diagnostic_inputs_forbidden_in_model": "",
            "leakage_status": "deployable_if_fit_inside_fold",
            "next_test": "Fold-local pressure-bucket residual uncertainty and mean correction screen with prior-only shrinkage.",
            "promotion_gate": "Must improve MAE and calibrated interval coverage in pressure-high rows.",
        },
        {
            "candidate_id": "nearby_temperature_level_error_scale",
            "priority": 7,
            "source_evidence": f"0056 top abs-error correlation feature {top_abs_feature.get('feature', '')}",
            "evidence_n": int(safe_float(top_abs_feature.get("n_abs_error_corr", 0))),
            "evidence_mae": math.nan,
            "evidence_bias": safe_float(top_abs_feature.get("corr_abs_error")),
            "mechanism_hypothesis": "Absolute error varies with nearby station temperature level; this may be more useful for uncertainty scaling than mean correction.",
            "deployable_inputs": str(top_abs_feature.get("feature", "")),
            "diagnostic_inputs_forbidden_in_model": "",
            "leakage_status": "deployable_if_fit_inside_fold",
            "next_test": "Fold-local residual scale model or interval-width adjustment keyed by station temperature level.",
            "promotion_gate": "Must improve tail calibration or reduce p90 absolute error without hurting MAE.",
        },
    ]
    queue = pd.DataFrame(rows)
    queue["ready_for_training_now"] = queue["leakage_status"].eq("deployable_if_fit_inside_fold")
    return queue.sort_values("priority").reset_index(drop=True)


def build_test_protocol(queue: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in queue.itertuples(index=False):
        rows.append(
            {
                "candidate_id": item.candidate_id,
                "precondition": "Use only 0054 station features and calendar fields available by T-1 15:00 HKT.",
                "split_policy": "Chronological 2000-2005, 2006-2011, 2012-2017, 2018-2023 OOF; fit all transforms inside each fold.",
                "forbidden_data": "2024+ confirmation rows, realized target heat bucket, upper-air/HKO daily candidates lacking timestamp proof, delayed backfill rows not yet verified continuous.",
                "primary_metric": "MAE versus 0055 ridge_all_station on same rows.",
                "secondary_metrics": "RMSE, bias, p90 absolute error, share abs error >=2C, subgroup degradation.",
                "minimum_acceptance": item.promotion_gate,
                "documentation_required": "New numbered research folder with predictions, fold metrics, leakage audit, and null-result report.",
            }
        )
    return pd.DataFrame(rows)


def leakage_audit(queue: pd.DataFrame) -> pd.DataFrame:
    checks = [
        {
            "check_id": "target_heat_bucket_not_ready_for_training",
            "passed": bool(
                queue[
                    queue["diagnostic_inputs_forbidden_in_model"].astype(str).str.contains("target heat bucket|realized target heat bucket", case=False, regex=True)
                ]["leakage_status"].eq("requires_proxy_validation").all()
            ),
            "evidence": "diagnostic heat-bucket candidates require deployable proxy validation",
        },
        {
            "check_id": "ready_rows_are_fold_local_only",
            "passed": bool(
                queue[queue["ready_for_training_now"]]["next_test"].astype(str).str.contains("fold|prior", case=False, regex=True).all()
            ),
            "evidence": f"{int(queue['ready_for_training_now'].sum())} ready rows require fold-local/prior-only tests",
        },
        {
            "check_id": "no_missing_candidate_ids",
            "passed": bool(queue["candidate_id"].astype(str).str.len().gt(0).all()),
            "evidence": f"{len(queue)} design candidates",
        },
    ]
    return pd.DataFrame(checks)


def build_readme(
    *,
    summary: dict[str, Any],
    queue: pd.DataFrame,
    protocol: pd.DataFrame,
    leakage: pd.DataFrame,
    failure_rank: pd.DataFrame,
) -> str:
    return f"""# Station-Only Residual Specialist Design Queue

Generated: `{summary['generated_at_utc']}`

## Purpose

`0056` identified where the `0055` station-only winner fails. This folder converts those miss patterns into a strict design queue for future residual specialists. It does not train a model, score a new forecast, or use delayed backfill rows.

## Summary

| Item | Value |
|---|---:|
| Design candidates | {summary['design_candidates']} |
| Ready for fold-local training now | {summary['ready_for_training_now']} |
| Proxy-validation required | {summary['proxy_validation_required']} |
| Top candidate | {summary['top_candidate_id']} |

## Design Queue

{markdown_table(queue, max_rows=30)}

## Test Protocol

{markdown_table(protocol, max_rows=30)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Source Failure Evidence

{markdown_table(failure_rank.head(20), max_rows=20)}

## Interpretation

The highest-priority work is not another broad Ridge model. The queue points to targeted specialists: recent-period bias repair, winter/summer mid-heat proxy regimes, February-March transition handling, spring pressure/dew gating, and uncertainty scaling by nearby station temperature level. Rows marked `requires_proxy_validation` must not use realized target heat buckets; they first need deployable pre-cutoff station proxies.

## Files

- `artifacts/design_queue.csv`
- `artifacts/test_protocol.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/source_failure_evidence.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_station_only_residual_specialist_design_queue.py`:

- `{FOLDER_NAME}`: non-modelling design queue for station-only residual specialists.

| Metric | Value |
|---|---:|
| Design candidates | {summary['design_candidates']} |
| Ready now | {summary['ready_for_training_now']} |
| Proxy-validation required | {summary['proxy_validation_required']} |
| Top candidate | {summary['top_candidate_id']} |

Leakage contract: diagnostic target-heat findings are blocked from training until converted to deployable pre-cutoff proxies.
"""
    update_markdown_section(RESEARCH_ROOT / "README.md", heading="Station-Only Residual Specialist Design Queue", section=section)


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_station_only_residual_specialist_design_queue.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Design candidates | `{summary['design_candidates']}` | Complete |
| Ready for fold-local tests | `{summary['ready_for_training_now']}` | Queued |
| Proxy-validation required | `{summary['proxy_validation_required']}` | Guarded |
| Top candidate | `{summary['top_candidate_id']}` | Prioritized |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: `0057` turns the station-only failure analysis into a test queue. It still does not train or promote any new model.
"""
    update_markdown_section(
        path,
        heading="Station-Only Residual Specialist Design Queue",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    blocker = (
        f"41. Station-only residual design queue contains `{summary['design_candidates']}` candidates; "
        f"`{summary['ready_for_training_now']}` can be tested fold-locally now and "
        f"`{summary['proxy_validation_required']}` require deployable proxy validation."
    )
    text = path.read_text(encoding="utf-8")
    if blocker not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{blocker}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Wait for the ongoing forecast backfill to finish enough continuity checks, then run the first fold-local residual-specialist test from `0057`: `late_period_bias_repair`, using only 0054 station features/calendar and current verified RSS/press rows where applicable.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    failure_rank = read_csv_required(FAILURE_RANK_PATH)
    feature_corr = read_csv_required(FEATURE_CORR_PATH)
    read_csv_required(WORST_DAYS_PATH)
    read_csv_required(THRESHOLDS_PATH)
    queue = build_design_queue(failure_rank, feature_corr)
    protocol = build_test_protocol(queue)
    leakage = leakage_audit(queue)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0057 leakage audit failed: {failed}")
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "design_candidates": int(len(queue)),
        "ready_for_training_now": int(queue["ready_for_training_now"].sum()),
        "proxy_validation_required": int(queue["leakage_status"].eq("requires_proxy_validation").sum()),
        "top_candidate_id": str(queue.iloc[0]["candidate_id"]) if not queue.empty else "",
        "leakage_check_rows": int(len(leakage)),
        "leakage_checks_passed": int(leakage["passed"].astype(bool).sum()),
    }

    folder = output_root / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    write_csv(artifacts / "design_queue.csv", queue)
    write_csv(artifacts / "test_protocol.csv", protocol)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_csv(artifacts / "source_failure_evidence.csv", failure_rank.head(100))
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "station_only_residual_specialist_design_queue_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            queue=queue,
            protocol=protocol,
            leakage=leakage,
            failure_rank=failure_rank,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Create station-only residual specialist design queue.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
