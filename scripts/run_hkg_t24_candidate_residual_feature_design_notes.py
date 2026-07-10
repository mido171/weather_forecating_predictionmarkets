from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_beastmode_signal_discovery import (  # noqa: E402
    CONFIRMATION_START,
    RESEARCH_ROOT,
    markdown_table,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import (  # noqa: E402
    EVAL_END,
    EVAL_START,
    TRAIN_END,
    update_markdown_section,
)

FOLDER_NAME = "0052_candidate_residual_feature_design_notes"
ARTIFACTS_0046 = RESEARCH_ROOT / "0046_long_history_cross_family_interaction_atlas" / "artifacts"
ARTIFACTS_0047 = RESEARCH_ROOT / "0047_station_contribution_atlas" / "artifacts"
ARTIFACTS_0048 = RESEARCH_ROOT / "0048_gated_residual_specialist_screen" / "artifacts"
ARTIFACTS_0049 = RESEARCH_ROOT / "0049_router_gate_stack_screen" / "artifacts"
ARTIFACTS_0050 = RESEARCH_ROOT / "0050_station_lag_slope_information_atlas" / "artifacts"
ARTIFACTS_0051 = RESEARCH_ROOT / "0051_station_regime_interaction_atlas" / "artifacts"


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 100) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def finite_float(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def deployable_status_for_family(source_family: str, feature_text: str) -> str:
    text = f"{source_family} {feature_text}".lower()
    if any(token in text for token in ["official_error", "abs_error", "residual", "target_tmax", "target_anomaly"]):
        return "diagnostic_only_outcome_or_residual"
    if "calendar" in text or "day_of_year" in text:
        return "deployable_input_candidate"
    if any(token in text for token in ["station", "isd_", "latest_before_1500", "wind_", "pressure", "dew_point"]):
        return "deployable_input_candidate"
    if any(token in text for token in ["igra", "upper_air", "ua_", "thickness"]):
        return "deployable_after_timestamp_audit"
    if any(token in text for token in ["daily_", "hko_daily", "lag"]):
        return "deployable_input_candidate"
    return "needs_timestamp_contract_review"


def priority_tier(score: float) -> str:
    if not math.isfinite(score):
        return "review"
    if score >= 5.0:
        return "tier_1"
    if score >= 3.0:
        return "tier_2"
    return "tier_3"


def add_candidate(
    rows: list[dict[str, object]],
    *,
    source_analysis: str,
    candidate_type: str,
    candidate_name: str,
    source_family: str,
    station_ids: str,
    deployable_feature_text: str,
    diagnostic_metric: str,
    primary_score: float,
    official_error_score: float,
    notes: str,
) -> None:
    status = deployable_status_for_family(source_family, deployable_feature_text)
    rows.append(
        {
            "candidate_id": slug(f"{source_analysis}_{candidate_type}_{candidate_name}", limit=140),
            "source_analysis": source_analysis,
            "candidate_type": candidate_type,
            "candidate_name": candidate_name,
            "source_family": source_family,
            "station_ids": station_ids,
            "deployable_feature_text": deployable_feature_text,
            "deployable_status": status,
            "diagnostic_metric": diagnostic_metric,
            "primary_score": primary_score,
            "official_error_score": official_error_score,
            "priority_tier": priority_tier(primary_score + 0.5 * (official_error_score if math.isfinite(official_error_score) else 0.0)),
            "leakage_note": (
                "Candidate inputs must be recomputed from timestamped pre-cutoff source rows only; "
                "scores, target labels, and residuals are diagnostics only."
            ),
            "notes": notes,
        }
    )


def build_deployable_candidates() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    physical_0046 = read_csv_if_exists(ARTIFACTS_0046 / "physical_interactions.csv").head(12)
    for row in physical_0046.to_dict("records"):
        feature_a = str(row.get("feature_a", ""))
        feature_b = str(row.get("feature_b", ""))
        source_family = f"{row.get('family_a', '')}+{row.get('family_b', '')}"
        add_candidate(
            rows,
            source_analysis="0046_cross_family_interaction_atlas",
            candidate_type="cross_family_joint_regime",
            candidate_name=f"{feature_a} x {feature_b}",
            source_family=source_family,
            station_ids=str(row.get("station_ids", "")),
            deployable_feature_text=f"{feature_a}; {feature_b}",
            diagnostic_metric="eval_target_anomaly_spread_c",
            primary_score=finite_float(row.get("eval_target_anomaly_spread_c")),
            official_error_score=finite_float(row.get("official_error_spread_c")),
            notes="Pre-2000 tertile thresholds; 2000-2023 evaluation. Upper-air items need source timestamp audit before deployment.",
        )

    station_attr = read_csv_if_exists(ARTIFACTS_0047 / "station_attribute_atlas.csv").head(20)
    for row in station_attr.to_dict("records"):
        add_candidate(
            rows,
            source_analysis="0047_station_contribution_atlas",
            candidate_type="station_attribute",
            candidate_name=f"{row.get('station_id', '')} {row.get('attribute', '')}",
            source_family="station_attribute",
            station_ids=str(row.get("station_id", "")),
            deployable_feature_text=str(row.get("attribute", "")),
            diagnostic_metric="abs_corr_eval_2000_2023_target_anomaly",
            primary_score=finite_float(row.get("abs_corr_eval_2000_2023_target_anomaly")),
            official_error_score=finite_float(row.get("abs_corr_official_error")),
            notes="T-1 latest-before-1500 HKT station value or network-relative station value.",
        )

    station_pairs = read_csv_if_exists(ARTIFACTS_0047 / "pair_spread_atlas.csv").head(20)
    for row in station_pairs.to_dict("records"):
        add_candidate(
            rows,
            source_analysis="0047_station_contribution_atlas",
            candidate_type="station_pair_spread",
            candidate_name=f"{row.get('pair_expression', '')} {row.get('attribute', '')}",
            source_family="station_pair_spread",
            station_ids=f"{row.get('station_a', '')},{row.get('station_b', '')}",
            deployable_feature_text=f"{row.get('pair_expression', '')} {row.get('attribute', '')}",
            diagnostic_metric="eval_tertile_target_anomaly_spread_c",
            primary_score=finite_float(row.get("eval_tertile_target_anomaly_spread_c")),
            official_error_score=finite_float(row.get("abs_corr_official_error")),
            notes="T-1 station-pair spread; strong candidate for smooth gating rather than sparse hard buckets.",
        )

    trajectories = read_csv_if_exists(ARTIFACTS_0050 / "station_lag_slope_variant_atlas.csv").head(30)
    for row in trajectories.to_dict("records"):
        add_candidate(
            rows,
            source_analysis="0050_station_lag_slope_information_atlas",
            candidate_type="station_trajectory",
            candidate_name=f"{row.get('station_id', '')} {row.get('feature_name', '')}",
            source_family="station_trajectory",
            station_ids=str(row.get("station_id", "")),
            deployable_feature_text=str(row.get("feature_name", "")),
            diagnostic_metric="abs_corr_eval_2000_2023_target_anomaly",
            primary_score=finite_float(row.get("abs_corr_eval_2000_2023_target_anomaly")),
            official_error_score=finite_float(row.get("abs_corr_official_error")),
            notes="Station-local lag, rolling, slope, or departure feature ending at T-1 pre-cutoff row.",
        )

    interactions = read_csv_if_exists(ARTIFACTS_0051 / "interaction_scoreboard.csv").head(40)
    for row in interactions.to_dict("records"):
        add_candidate(
            rows,
            source_analysis="0051_station_regime_interaction_atlas",
            candidate_type="station_regime_interaction",
            candidate_name=f"{row.get('gate_display_name', '')} x {row.get('response_display_name', '')}",
            source_family=f"{row.get('gate_source_family', '')}+{row.get('response_source_family', '')}",
            station_ids=f"{row.get('gate_station_ids', '')};{row.get('response_station_ids', '')}",
            deployable_feature_text=f"{row.get('gate_display_name', '')}; {row.get('response_display_name', '')}",
            diagnostic_metric="eval_target_anomaly_spread_c",
            primary_score=finite_float(row.get("eval_target_anomaly_spread_c")),
            official_error_score=finite_float(row.get("official_error_spread_c")),
            notes="Pre-2000 low/mid/high thresholds with 2000-2023 station-regime separation.",
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["priority_tier", "primary_score", "official_error_score"],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)


def build_diagnostic_only_inputs() -> pd.DataFrame:
    rows = [
        {
            "diagnostic_item": "target_tmax_c",
            "why_not_deployable_input": "It is the realized label being forecast.",
            "allowed_use": "Training labels, OOF scoring, feature discovery target.",
        },
        {
            "diagnostic_item": "target_anomaly_vs_past_doy_c",
            "why_not_deployable_input": "It includes the realized target date Tmax minus past-only climatology.",
            "allowed_use": "Diagnostic outcome for correlation and cell spread analysis.",
        },
        {
            "diagnostic_item": "official_error_c / official_abs_error_c",
            "why_not_deployable_input": "Requires realized target Tmax; unavailable at forecast time.",
            "allowed_use": "Residual modelling target and post-hoc failure anatomy only.",
        },
        {
            "diagnostic_item": "MAE/RMSE/delta_vs_anchor score columns",
            "why_not_deployable_input": "They summarize realized performance on historical rows.",
            "allowed_use": "Candidate selection reports and prior-only validation logs.",
        },
        {
            "diagnostic_item": "2024-2026 confirmation labels",
            "why_not_deployable_input": "Locked confirmation period; not part of current research selection.",
            "allowed_use": "Final confirmation only after explicit command.",
        },
    ]
    return pd.DataFrame(rows)


def build_model_test_queue(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "test_id": "0052_T1_station_thermal_departure_router",
            "test_family": "station_trajectory_interaction",
            "required_inputs": "0050 current_minus_rolling_mean_14d station temperature departures + 0047 static station temperature/network-relative states",
            "anchor": "continuous official pre-cutoff forecast max",
            "minimum_data_gate": "continuous 2000-2023 scored official rows, no 2024+",
            "oof_policy": "walk-forward or blocked folds with at least 4 full OOF years; all thresholds fit before each OOF block",
            "reason": "0051 top interactions show 4.7+ C target-anomaly separation and 0.6+ C official-error separation.",
        },
        {
            "test_id": "0052_T2_pressure_pair_dew_thermal_gate",
            "test_family": "pressure_pair_plus_moisture",
            "required_inputs": "0047 pressure station-pair spreads + 0050 dew/temperature station departures",
            "anchor": "continuous official pre-cutoff forecast max",
            "minimum_data_gate": "continuous official archive and timestamp-verified station source rows",
            "oof_policy": "same OOF blocks as T1; compare against frozen 0042/0049 router benchmark",
            "reason": "Top 0051 rows include pressure-pair and dew/thermal interactions; 0046/0047 also rank pressure placement highly.",
        },
        {
            "test_id": "0052_T3_upper_air_pressure_station_joint",
            "test_family": "upper_air_station_pressure",
            "required_inputs": "0046 upper-air thickness/tendency features + station pressure plane/pair features",
            "anchor": "continuous official pre-cutoff forecast max",
            "minimum_data_gate": "upper-air issue/retrieval timestamp audit plus continuous official rows",
            "oof_policy": "timestamp-audited walk-forward; exclude any upper-air row not proven available before cutoff",
            "reason": "0046 strongest non-calendar interaction is pressure plane slope x upper-air thickness change.",
        },
        {
            "test_id": "0052_T4_router_feature_ablation",
            "test_family": "trust_router_ablation",
            "required_inputs": "0049 router benchmark plus top deployable 0051 interaction features",
            "anchor": "0049 frozen router benchmark",
            "minimum_data_gate": "continuous official rows; no target leakage features",
            "oof_policy": "nested prior-only router selection; feature additions must beat 0049 on full and late OOF windows",
            "reason": "0049 gains were small; new features should be admitted only through strict ablation.",
        },
        {
            "test_id": "0052_T5_no_model_feature_freeze",
            "test_family": "governance",
            "required_inputs": "deployable_feature_candidates.csv and diagnostic_only_inputs.csv",
            "anchor": "none",
            "minimum_data_gate": "manual review before any ML run",
            "oof_policy": "not a model test",
            "reason": "Prevents diagnostic residual/label columns from accidentally becoming model inputs.",
        },
    ]
    out = pd.DataFrame(rows)
    if not candidates.empty:
        out["top_candidate_count_available"] = int((candidates["priority_tier"].eq("tier_1")).sum())
    return out


def build_family_evidence(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    return (
        candidates.groupby(["source_analysis", "candidate_type", "deployable_status"], observed=True)
        .agg(
            rows=("candidate_id", "size"),
            best_primary_score=("primary_score", "max"),
            median_primary_score=("primary_score", "median"),
            best_official_error_score=("official_error_score", "max"),
            tier_1_count=("priority_tier", lambda series: int((series == "tier_1").sum())),
        )
        .reset_index()
        .sort_values(["tier_1_count", "best_primary_score"], ascending=[False, False])
    )


def build_readme(
    *,
    summary: dict[str, Any],
    candidates: pd.DataFrame,
    diagnostics: pd.DataFrame,
    queue: pd.DataFrame,
    family_evidence: pd.DataFrame,
) -> str:
    return f"""# Candidate Residual Feature Design Notes

Generated: `{summary['generated_at_utc']}`

## Purpose

This folder consolidates the local research findings from `0046` through `0051` into an explicit next-step design map. It does not train a model. It does not backtest Polymarket. It does not use 2024+ labels. Its job is to separate what can become a real pre-cutoff model input from what is only a diagnostic score, and to define the strict tests that should run once the official HKO forecast archive backfill becomes continuous.

## Leakage Contract

- Deployable candidate inputs must be known before the T-24 forecast cutoff.
- Target labels, target anomalies, forecast errors, MAE/RMSE, and realized residual columns are diagnostics only.
- Thresholds discovered in 0046/0051 were fit on pre-2000 rows and evaluated on 2000-2023 rows.
- The current official forecast archive is still non-contiguous; residual-model claims stay provisional until that is fixed.
- Confirmation rows beginning `{CONFIRMATION_START.date()}` remain locked.

## Scope

| Item | Value |
|---|---:|
| Candidate rows | {summary['candidate_rows']} |
| Tier 1 candidates | {summary['tier_1_candidate_rows']} |
| Diagnostic-only rows | {summary['diagnostic_only_rows']} |
| Model test queue rows | {summary['model_test_queue_rows']} |
| Evidence first date | {summary['evidence_first_date']} |
| Evidence last date | {summary['evidence_last_date']} |

## Candidate Families

{markdown_table(family_evidence, max_rows=80)}

## Top Deployable Candidate Inputs

{markdown_table(candidates.head(80), max_rows=80)}

## Diagnostic-Only Inputs

{markdown_table(diagnostics, max_rows=20)}

## Strict Future Test Queue

{markdown_table(queue, max_rows=20)}

## Interpretation

The strongest direction is now clear: station thermal departure regimes, station pressure placement, station-pair pressure spreads, and dew/moisture departures are the main local information channels to carry forward. The 0051 result is especially important because it shows that one station's recent warming departure changes how another station's absolute temperature maps to HKG target Tmax. That is exactly the kind of nonlinear, cross-station behavior that simple global regressions tend to underuse.

The next real modelling step should wait for the continuous official forecast archive. Until then, the correct work is to harden feature contracts, timestamp eligibility, and the ablation queue.

## Files

- `artifacts/deployable_feature_candidates.csv`
- `artifacts/diagnostic_only_inputs.csv`
- `artifacts/model_test_queue.csv`
- `artifacts/feature_family_evidence.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_candidate_residual_feature_design_notes.py`:

- `{FOLDER_NAME}`: deployable-vs-diagnostic feature synthesis and strict future test queue from `0046`-`0051`.

| Metric | Value |
|---|---:|
| Candidate rows | {summary['candidate_rows']} |
| Tier 1 candidates | {summary['tier_1_candidate_rows']} |
| Model test queue rows | {summary['model_test_queue_rows']} |

Leakage contract: no new model, no 2024+ labels, diagnostic residual/score columns separated from deployable pre-cutoff candidate inputs.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Candidate Residual Feature Design Notes",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_candidate_residual_feature_design_notes.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Candidate synthesis | `{summary['candidate_rows']}` candidate rows from `0046`-`0051` | Documented |
| Tier 1 rows | `{summary['tier_1_candidate_rows']}` rows | Prioritized |
| Diagnostic-only guard | `{summary['diagnostic_only_rows']}` forbidden input categories | Guarded |
| Future test queue | `{summary['model_test_queue_rows']}` strict tests | Waiting for continuous forecast archive |

Interpretation: `0052` is the handoff layer between feature discovery and future modelling. It prevents residual/label leakage and defines the exact candidate families to test after archive continuity is available.
"""
    update_markdown_section(
        path,
        heading="Candidate Residual Feature Design Notes",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    blocker = (
        f"36. Candidate residual feature design notes now separate `{summary['candidate_rows']}` candidate inputs "
        f"from diagnostic-only fields and define `{summary['model_test_queue_rows']}` strict future tests. "
        "The next modelling action remains blocked on a continuous official forecast archive."
    )
    text = path.read_text(encoding="utf-8")
    if blocker not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{blocker}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Wait for the official forecast backfill to finish or advance the timestamp/audit hardening: verify the continuous scored forecast frame, then run the `0052` queued tests in strict walk-forward order without using 2024+ confirmation rows.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    folder = output_root / FOLDER_NAME
    artifacts = folder / "artifacts"
    candidates = build_deployable_candidates()
    diagnostics = build_diagnostic_only_inputs()
    queue = build_model_test_queue(candidates)
    family_evidence = build_family_evidence(candidates)
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "candidate_rows": int(len(candidates)),
        "tier_1_candidate_rows": int((candidates["priority_tier"].eq("tier_1")).sum()) if not candidates.empty else 0,
        "diagnostic_only_rows": int(len(diagnostics)),
        "model_test_queue_rows": int(len(queue)),
        "evidence_first_date": "pre-2000 threshold training where applicable",
        "evidence_last_date": str(EVAL_END.date()),
        "training_threshold_end": str(TRAIN_END.date()),
        "evaluation_start": str(EVAL_START.date()),
        "confirmation_start": str(CONFIRMATION_START.date()),
        "uses_2024_plus_rows": False,
    }
    write_csv(artifacts / "deployable_feature_candidates.csv", candidates)
    write_csv(artifacts / "diagnostic_only_inputs.csv", diagnostics)
    write_csv(artifacts / "model_test_queue.csv", queue)
    write_csv(artifacts / "feature_family_evidence.csv", family_evidence)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "candidate_residual_feature_design_notes_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            candidates=candidates,
            diagnostics=diagnostics,
            queue=queue,
            family_evidence=family_evidence,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Build HKG T24 candidate residual feature design notes.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
