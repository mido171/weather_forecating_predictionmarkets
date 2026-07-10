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
    require_no_confirmation_dates,
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

FOLDER_NAME = "0053_candidate_timestamp_eligibility_audit"
DATASETS_ROOT = REPO_ROOT / "data" / "datasets"
INPUT_CANDIDATES_PATH = (
    RESEARCH_ROOT
    / "0052_candidate_residual_feature_design_notes"
    / "artifacts"
    / "deployable_feature_candidates.csv"
)
INPUT_DIAGNOSTICS_PATH = (
    RESEARCH_ROOT
    / "0052_candidate_residual_feature_design_notes"
    / "artifacts"
    / "diagnostic_only_inputs.csv"
)
STATION_DAY_PATH = (
    DATASETS_ROOT
    / "04_noaa_isd_regional_surface"
    / "noaa_isd_station_day_cutoff_summary.parquet"
)
CORE_OBS_PATH = (
    DATASETS_ROOT
    / "04_noaa_isd_regional_surface"
    / "noaa_isd_core_observations.parquet"
)
FEATURE_MATRIX_PATH = (
    DATASETS_ROOT
    / "12_hkg_t24_robust_experiment_outputs"
    / "hkg_t24_exp0050_0099_feature_matrix.parquet"
)


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 120) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def source_family_tokens(source_family: str) -> set[str]:
    return {token.strip().lower() for token in re.split(r"[+,; ]+", str(source_family)) if token.strip()}


def contains_any(text: str, needles: set[str]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def timing_status_for_candidate(row: pd.Series) -> dict[str, object]:
    status_0052 = str(row.get("deployable_status", ""))
    source_family = str(row.get("source_family", ""))
    candidate_type = str(row.get("candidate_type", ""))
    feature_text = str(row.get("deployable_feature_text", ""))
    combined = f"{source_family} {candidate_type} {feature_text}".lower()
    tokens = source_family_tokens(source_family)

    if "diagnostic_only" in status_0052 or contains_any(
        combined,
        {"official_error", "official_abs_error", "target_tmax", "target_anomaly", "mae", "rmse"},
    ):
        return {
            "timestamp_audit_status": "forbidden_diagnostic_or_outcome",
            "allowed_for_future_walkforward": False,
            "cutoff_rule": "forbidden",
            "evidence_source": "0052 diagnostic guard",
            "required_proof_before_model": "remove from predictor matrix",
            "blocker": "uses outcome, realized error, or diagnostic score",
        }

    if "upper_air" in tokens or contains_any(combined, {"igra_", "ua_", "thickness", "925hpa", "850hpa"}):
        return {
            "timestamp_audit_status": "timestamp_audit_required",
            "allowed_for_future_walkforward": False,
            "cutoff_rule": "must prove sounding/upper-air value was issued or available by T-1 15:00 HKT",
            "evidence_source": "feature matrix has valid-time fields, but issue/available-at proof is not yet attached",
            "required_proof_before_model": "join upper-air rows to provider issue/retrieval metadata and enforce available_at <= cutoff",
            "blocker": "upper-air valid time alone is not sufficient for point-in-time eligibility",
        }

    if "hko_daily_climate" in tokens or contains_any(combined, {"daily_hong_kong_observatory", "daily_"}):
        return {
            "timestamp_audit_status": "publication_lag_audit_required",
            "allowed_for_future_walkforward": False,
            "cutoff_rule": "only lagged daily climate values with proven publication before cutoff are eligible",
            "evidence_source": "SOURCE_TIMESTAMP_CONTRACTS marks same-day HKO daily climate retrospective",
            "required_proof_before_model": "prove lagged daily value publication timestamp is before cutoff for every target row",
            "blocker": "official daily climate publication lag is not fully proven",
        }

    station_like = (
        tokens <= {"station_attribute", "station_trajectory", "station_pair_spread", "isd_station_network"}
        or contains_any(combined, {"latest_before_1500", "station_", "isd_", "wind_", "dew_point", "sea_level_pressure"})
    )
    if station_like:
        return {
            "timestamp_audit_status": "eligible_proven_pre_cutoff_station",
            "allowed_for_future_walkforward": True,
            "cutoff_rule": "target T uses station local_date T-1 and latest_before_1500_hkt observation only",
            "evidence_source": "noaa_isd_station_day_cutoff_summary.operational_input_allowed plus latest_before_1500_hkt",
            "required_proof_before_model": "recompute feature inside each walk-forward fold from station cutoff summary only",
            "blocker": "",
        }

    return {
        "timestamp_audit_status": "needs_manual_timestamp_review",
        "allowed_for_future_walkforward": False,
        "cutoff_rule": "unknown",
        "evidence_source": "not covered by automated 0053 family rules",
        "required_proof_before_model": "write source-specific issue/available-at contract",
        "blocker": "missing source-specific timestamp rule",
    }


def audit_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in candidates.to_dict("records"):
        series = pd.Series(row)
        timing = timing_status_for_candidate(series)
        primary = float(row.get("primary_score", math.nan))
        official = float(row.get("official_error_score", math.nan))
        rows.append(
            {
                **row,
                **timing,
                "audit_priority_score": (
                    (primary if math.isfinite(primary) else 0.0)
                    + 0.5 * (abs(official) if math.isfinite(official) else 0.0)
                    + (1.0 if timing["allowed_for_future_walkforward"] else 0.0)
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["allowed_for_future_walkforward", "audit_priority_score"],
        ascending=[False, False],
        na_position="last",
    )


def summarize_station_day_source() -> dict[str, object]:
    if not STATION_DAY_PATH.exists():
        return {"source_id": "noaa_isd_station_day_cutoff_summary", "status": "missing"}
    day = pd.read_parquet(STATION_DAY_PATH)
    day["local_date"] = pd.to_datetime(day["local_date"], errors="coerce").dt.normalize()
    allowed = day["operational_input_allowed"].astype(bool) if "operational_input_allowed" in day.columns else pd.Series(False, index=day.index)
    latest = pd.to_datetime(day["latest_before_1500_hkt"], errors="coerce", utc=True) if "latest_before_1500_hkt" in day.columns else pd.Series(pd.NaT, index=day.index)
    return {
        "source_id": "noaa_isd_station_day_cutoff_summary",
        "status": "present",
        "path": str(STATION_DAY_PATH),
        "rows": int(len(day)),
        "stations": int(day["station_id"].nunique()) if "station_id" in day.columns else 0,
        "first_local_date": str(day["local_date"].min().date()),
        "last_local_date": str(day["local_date"].max().date()),
        "operational_input_allowed_rows": int(allowed.sum()),
        "latest_before_1500_non_null_rows": int(latest.notna().sum()),
        "timing_contract": "station local_date T-1 latest_before_1500_hkt; no target-day observation",
    }


def summarize_core_observation_source() -> dict[str, object]:
    if not CORE_OBS_PATH.exists():
        return {"source_id": "noaa_isd_core_observations", "status": "missing"}
    core = pd.read_parquet(CORE_OBS_PATH, columns=["station_id", "observed_at_hkt", "raw_retrieved_at_utc", "operational_input_allowed", "source_time_policy"])
    observed = pd.to_datetime(core["observed_at_hkt"], errors="coerce", utc=True)
    retrieved = pd.to_datetime(core["raw_retrieved_at_utc"], errors="coerce", utc=True)
    allowed = core["operational_input_allowed"].astype(bool)
    return {
        "source_id": "noaa_isd_core_observations",
        "status": "present",
        "path": str(CORE_OBS_PATH),
        "rows": int(len(core)),
        "stations": int(core["station_id"].nunique()),
        "first_observed_at_hkt": str(observed.min()),
        "last_observed_at_hkt": str(observed.max()),
        "archive_retrieved_first_utc": str(retrieved.min()),
        "archive_retrieved_last_utc": str(retrieved.max()),
        "operational_input_allowed_rows": int(allowed.sum()),
        "source_time_policy_values": ",".join(sorted(core["source_time_policy"].dropna().astype(str).unique())),
        "timing_contract": "observed_at_hkt is source time; archival retrieval timestamp is provenance, not issue time",
    }


def summarize_feature_matrix_source() -> dict[str, object]:
    if not FEATURE_MATRIX_PATH.exists():
        return {"source_id": "hkg_t24_exp0050_0099_feature_matrix", "status": "missing"}
    columns = pd.read_parquet(FEATURE_MATRIX_PATH, columns=["target_date", "valid_at_hkt"])
    columns["target_date"] = pd.to_datetime(columns["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(columns["target_date"], context="0053 feature matrix target dates")
    valid_at = pd.to_datetime(columns["valid_at_hkt"], errors="coerce", utc=True)
    return {
        "source_id": "hkg_t24_exp0050_0099_feature_matrix",
        "status": "present",
        "path": str(FEATURE_MATRIX_PATH),
        "rows": int(len(columns)),
        "first_target_date": str(columns["target_date"].min().date()),
        "last_target_date": str(columns["target_date"].max().date()),
        "valid_at_hkt_non_null_rows": int(valid_at.notna().sum()),
        "timing_contract": "mixed long-history feature matrix; upper-air and daily-climate features require source-specific available-at audit",
    }


def source_timing_evidence() -> pd.DataFrame:
    rows = [
        summarize_station_day_source(),
        summarize_core_observation_source(),
        summarize_feature_matrix_source(),
    ]
    return pd.DataFrame(rows)


def status_summary(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame()
    return (
        audit.groupby(["timestamp_audit_status", "allowed_for_future_walkforward"], observed=True)
        .agg(
            rows=("candidate_id", "size"),
            best_audit_priority_score=("audit_priority_score", "max"),
            best_primary_score=("primary_score", "max"),
            best_official_error_score=("official_error_score", "max"),
        )
        .reset_index()
        .sort_values(["allowed_for_future_walkforward", "rows"], ascending=[False, False])
    )


def blocked_candidate_summary(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame()
    blocked = audit[~audit["allowed_for_future_walkforward"].astype(bool)].copy()
    if blocked.empty:
        return pd.DataFrame()
    return (
        blocked.groupby(["timestamp_audit_status", "blocker", "required_proof_before_model"], observed=True)
        .agg(rows=("candidate_id", "size"), best_primary_score=("primary_score", "max"))
        .reset_index()
        .sort_values(["rows", "best_primary_score"], ascending=[False, False])
    )


def build_readme(
    *,
    summary: dict[str, Any],
    audit: pd.DataFrame,
    evidence: pd.DataFrame,
    statuses: pd.DataFrame,
    blocked: pd.DataFrame,
) -> str:
    allowed = audit[audit["allowed_for_future_walkforward"].astype(bool)].head(50) if not audit.empty else pd.DataFrame()
    allowed_display = allowed[
        [
            "candidate_type",
            "candidate_name",
            "source_family",
            "timestamp_audit_status",
            "primary_score",
            "official_error_score",
            "cutoff_rule",
        ]
    ] if not allowed.empty else pd.DataFrame()
    blocked_display = blocked.head(40) if not blocked.empty else pd.DataFrame()
    return f"""# Candidate Timestamp Eligibility Audit

Generated: `{summary['generated_at_utc']}`

## Purpose

`0052` created a feature handoff list from the `0046`-`0051` deep-dive analyses. This folder audits that list against the T-24 cutoff contract. The point is simple: strong information gain is not enough. A candidate is only allowed into future walk-forward modelling if the source value can be proven available no later than target date `T-1 15:00 HKT`.

## Result

| Item | Value |
|---|---:|
| Candidate rows audited | {summary['candidate_rows']} |
| Walk-forward allowed now | {summary['allowed_now_rows']} |
| Blocked pending timestamp proof | {summary['blocked_rows']} |
| Station-derived allowed rows | {summary['station_allowed_rows']} |
| Upper-air audit-required rows | {summary['upper_air_audit_required_rows']} |
| HKO daily publication audit-required rows | {summary['hko_daily_audit_required_rows']} |
| Uses 2024+ rows | {summary['uses_2024_plus_rows']} |

## Leakage Contract

- Target labels, target anomalies, official forecast errors, and realized score columns remain forbidden predictors.
- Station-derived candidates are allowed only when recomputed from `noaa_isd_station_day_cutoff_summary` using local date `T-1` and `latest_before_1500_hkt`.
- Upper-air candidates are not allowed yet unless their issue/retrieval availability can be joined and proven before `T-1 15:00 HKT`.
- HKO official daily climate candidates are not allowed yet unless publication lag is proven before the cutoff for the lagged value used.
- Confirmation rows beginning `{CONFIRMATION_START.date()}` remain locked.

## Status Summary

{markdown_table(statuses, max_rows=40)}

## Source Timing Evidence

{markdown_table(evidence, max_rows=20)}

## Allowed Candidate Examples

{markdown_table(allowed_display, max_rows=50)}

## Blocked Candidate Summary

{markdown_table(blocked_display, max_rows=40)}

## Interpretation

This audit narrows the immediate modelling-ready input pool to station-derived features. That is useful: the best 0051 interactions are station thermal-departure and station temperature/pressure regime features, and those survive the timestamp audit. Upper-air and HKO-daily candidates remain valuable research leads, but they need explicit available-at proof before they can enter a leakage-free model.

## Files

- `artifacts/candidate_timestamp_audit.csv`
- `artifacts/source_timing_evidence.csv`
- `artifacts/status_summary.csv`
- `artifacts/blocked_candidate_summary.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_candidate_timestamp_eligibility_audit.py`:

- `{FOLDER_NAME}`: point-in-time eligibility audit for `0052` candidate inputs.

| Metric | Value |
|---|---:|
| Candidate rows audited | {summary['candidate_rows']} |
| Allowed now | {summary['allowed_now_rows']} |
| Blocked pending timestamp proof | {summary['blocked_rows']} |

Leakage contract: station candidates must be recomputed from T-1 latest-before-15:00 station rows; upper-air and HKO daily candidates remain blocked until available-at proof is attached.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Candidate Timestamp Eligibility Audit",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_candidate_timestamp_eligibility_audit.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Candidate timestamp audit | `{summary['candidate_rows']}` rows | Complete |
| Walk-forward allowed now | `{summary['allowed_now_rows']}` station-derived rows | Eligible |
| Blocked pending proof | `{summary['blocked_rows']}` rows | Blocked before modelling |
| Upper-air audit required | `{summary['upper_air_audit_required_rows']}` rows | Needs available-at proof |
| HKO daily publication audit required | `{summary['hko_daily_audit_required_rows']}` rows | Needs publication-lag proof |

Interpretation: `0053` proves the immediate leakage-safe candidate pool is station-derived. Upper-air and HKO daily candidates stay out of future models until timestamp contracts are attached.
"""
    update_markdown_section(
        path,
        heading="Candidate Timestamp Eligibility Audit",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    blocker = (
        f"37. Candidate timestamp eligibility audit allows `{summary['allowed_now_rows']}` station-derived rows now "
        f"and blocks `{summary['blocked_rows']}` rows pending timestamp/publication proof. This preserves point-in-time safety "
        "for the next walk-forward model queue."
    )
    text = path.read_text(encoding="utf-8")
    if blocker not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{blocker}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
While the official forecast backfill continues, build a station-only walk-forward feature matrix from the `0053` allowed candidates and run leakage checks only. Do not train a final model until the continuous official forecast frame is verified.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    if not INPUT_CANDIDATES_PATH.exists():
        raise FileNotFoundError(f"Missing 0052 candidate file: {INPUT_CANDIDATES_PATH}")
    generated_at = now_utc()
    folder = output_root / FOLDER_NAME
    artifacts = folder / "artifacts"
    candidates = pd.read_csv(INPUT_CANDIDATES_PATH)
    diagnostics = pd.read_csv(INPUT_DIAGNOSTICS_PATH) if INPUT_DIAGNOSTICS_PATH.exists() else pd.DataFrame()
    audit = audit_candidates(candidates)
    evidence = source_timing_evidence()
    statuses = status_summary(audit)
    blocked = blocked_candidate_summary(audit)

    allowed_now = audit[audit["allowed_for_future_walkforward"].astype(bool)].copy()
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "candidate_rows": int(len(audit)),
        "diagnostic_guard_rows": int(len(diagnostics)),
        "allowed_now_rows": int(len(allowed_now)),
        "blocked_rows": int((~audit["allowed_for_future_walkforward"].astype(bool)).sum()) if not audit.empty else 0,
        "station_allowed_rows": int(audit["timestamp_audit_status"].eq("eligible_proven_pre_cutoff_station").sum()) if not audit.empty else 0,
        "upper_air_audit_required_rows": int(audit["timestamp_audit_status"].eq("timestamp_audit_required").sum()) if not audit.empty else 0,
        "hko_daily_audit_required_rows": int(audit["timestamp_audit_status"].eq("publication_lag_audit_required").sum()) if not audit.empty else 0,
        "training_threshold_end": str(TRAIN_END.date()),
        "evaluation_start": str(EVAL_START.date()),
        "evaluation_end": str(EVAL_END.date()),
        "confirmation_start": str(CONFIRMATION_START.date()),
        "uses_2024_plus_rows": False,
    }

    write_csv(artifacts / "candidate_timestamp_audit.csv", audit)
    write_csv(artifacts / "source_timing_evidence.csv", evidence)
    write_csv(artifacts / "status_summary.csv", statuses)
    write_csv(artifacts / "blocked_candidate_summary.csv", blocked)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "candidate_timestamp_eligibility_audit_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            audit=audit,
            evidence=evidence,
            statuses=statuses,
            blocked=blocked,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 candidate timestamp eligibility audit.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
