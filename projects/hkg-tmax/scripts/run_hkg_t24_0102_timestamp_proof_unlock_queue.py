from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from hkg_tmax.paths import ProjectPaths

PROJECT_PATHS = ProjectPaths.discover(Path(__file__))
REPO_ROOT = PROJECT_PATHS.project_root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_beastmode_signal_discovery import (
    CONFIRMATION_START,
    RESEARCH_ROOT,
    markdown_table,
    require_no_confirmation_dates,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import update_markdown_section

FOLDER_NAME = "0102_timestamp_proof_unlock_queue"
DATASETS_ROOT = PROJECT_PATHS.data_root / "datasets"
INPUT_0100_ATLAS_PATH = RESEARCH_ROOT / "0100_stable_mam_cell_feature_atlas" / "artifacts" / "feature_atlas.csv"
INPUT_0101_SUMMARY_PATH = (
    RESEARCH_ROOT / "0101_stable_mam_cell_feature_specialists" / "artifacts" / "summary.json"
)

IGRA_SOUNDING_FEATURES_PATH = (
    DATASETS_ROOT
    / "03_noaa_igra_upper_air_hkm00045004"
    / "noaa_igra_hkm00045004_sounding_features.parquet"
)
IGRA_KEY_PRESSURE_LEVELS_PATH = (
    DATASETS_ROOT
    / "03_noaa_igra_upper_air_hkm00045004"
    / "noaa_igra_hkm00045004_key_pressure_levels.parquet"
)
HKO_DAILY_CLIMATE_PATH = (
    DATASETS_ROOT / "02_hko_daily_climate_all_elements" / "hko_daily_climate_elements.parquet"
)

AUDITED_FAMILIES = ("upper_air", "hko_daily_climate", "marine_proxy")
BOOLEAN_TRUE = {"1", "true", "t", "yes", "y"}
PROVIDER_TIME_PROOF_TOKENS = (
    "available_at",
    "available_time",
    "first_available",
    "first_seen_at",
    "issued_at",
    "issue_utc",
    "issue_time",
    "published_at",
    "publication_time",
    "publish_time",
)
SOURCE_POLICY_COLUMN = "source_time_policy"


@dataclass(frozen=True)
class SourceTableSpec:
    source_family: str
    source_id: str
    path: Path
    source_time_column: str
    timing_contract: str


SOURCE_TABLES = (
    SourceTableSpec(
        source_family="upper_air",
        source_id="noaa_igra_hkm00045004_sounding_features",
        path=IGRA_SOUNDING_FEATURES_PATH,
        source_time_column="valid_at_utc",
        timing_contract=(
            "Target T features are derived from the T-1 00 UTC sounding where available, but provider "
            "release/available-at timing must still be proven before promotion."
        ),
    ),
    SourceTableSpec(
        source_family="upper_air",
        source_id="noaa_igra_hkm00045004_key_pressure_levels",
        path=IGRA_KEY_PRESSURE_LEVELS_PATH,
        source_time_column="valid_at_utc",
        timing_contract=(
            "Pressure-level rows share the same IGRA archive timing risk: valid_at is meteorological time, "
            "not issue or first-available time."
        ),
    ),
    SourceTableSpec(
        source_family="hko_daily_climate",
        source_id="hko_daily_climate_elements",
        path=HKO_DAILY_CLIMATE_PATH,
        source_time_column="local_date",
        timing_contract=(
            "Features use lagged daily climate values, but the finalized table currently lacks first-publication "
            "timestamps."
        ),
    ),
    SourceTableSpec(
        source_family="marine_proxy",
        source_id="hko_daily_climate_elements_marine_proxies",
        path=HKO_DAILY_CLIMATE_PATH,
        source_time_column="local_date",
        timing_contract=(
            "Marine proxy features are lagged North Point and Waglan Island values inside the finalized HKO "
            "daily climate table; first-publication timing is not attached."
        ),
    ),
)


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def truthy_count(series: pd.Series) -> int:
    if pd.api.types.is_bool_dtype(series):
        return int(series.fillna(False).astype(bool).sum())
    return int(series.astype(str).str.strip().str.lower().isin(BOOLEAN_TRUE).sum())


def compact_values(series: pd.Series, *, limit: int = 4) -> str:
    values = sorted({str(value) for value in series.dropna().astype(str).unique() if str(value).strip()})
    if not values:
        return ""
    clipped = values[:limit]
    suffix = "" if len(values) <= limit else f" | ... +{len(values) - limit} more"
    return " | ".join(clipped) + suffix


def matching_time_proof_columns(columns: list[str]) -> list[str]:
    matches: list[str] = []
    for column in columns:
        lowered = column.lower()
        if lowered.startswith("valid_at") or lowered.startswith("raw_retrieved_at"):
            continue
        if any(token in lowered for token in PROVIDER_TIME_PROOF_TOKENS):
            matches.append(column)
    return sorted(matches)


def source_unlock_decision(
    *,
    source_family: str,
    rows: int,
    provider_time_proof_columns: list[str],
    operational_input_allowed_true_rows: int,
    release_latency_proven_true_rows: int,
) -> dict[str, object]:
    all_rows_operational = rows > 0 and operational_input_allowed_true_rows == rows
    all_release_latency_proven = rows > 0 and release_latency_proven_true_rows == rows
    has_provider_time_proof = bool(provider_time_proof_columns)

    if source_family == "upper_air":
        if all_rows_operational and (has_provider_time_proof or all_release_latency_proven):
            return {
                "unlock_decision": True,
                "proof_status": "provider_available_at_or_release_latency_proven",
                "required_next_evidence": "",
            }
        blockers = []
        if not all_rows_operational:
            blockers.append("operational_input_allowed is not true for every row")
        if not has_provider_time_proof and not all_release_latency_proven:
            blockers.append("no provider issue/available-at column or all-row release-latency proof")
        return {
            "unlock_decision": False,
            "proof_status": "blocked_missing_upper_air_available_at_or_release_latency",
            "required_next_evidence": "; ".join(blockers),
        }

    if source_family in {"hko_daily_climate", "marine_proxy"}:
        if all_rows_operational and has_provider_time_proof:
            return {
                "unlock_decision": True,
                "proof_status": "provider_publication_time_proven",
                "required_next_evidence": "",
            }
        blockers = []
        if not all_rows_operational:
            blockers.append("operational_input_allowed is not true for every row")
        if not has_provider_time_proof:
            blockers.append("no first-publication or available-at column is attached")
        return {
            "unlock_decision": False,
            "proof_status": "blocked_missing_daily_publication_timestamp",
            "required_next_evidence": "; ".join(blockers),
        }

    return {
        "unlock_decision": False,
        "proof_status": "blocked_unknown_source_family",
        "required_next_evidence": "write a family-specific source timestamp contract",
    }


def source_table_summary(spec: SourceTableSpec) -> dict[str, object]:
    if not spec.path.exists():
        return {
            "source_family": spec.source_family,
            "source_id": spec.source_id,
            "path": str(spec.path),
            "status": "missing",
            "rows": 0,
            "source_time_column": spec.source_time_column,
            "first_source_time": "",
            "last_source_time": "",
            "provider_time_proof_columns": "",
            "operational_input_allowed_true_rows": 0,
            "release_latency_proven_true_rows": 0,
            "availability_tier_values": "",
            "source_time_policy_values": "",
            "timing_contract": spec.timing_contract,
            **source_unlock_decision(
                source_family=spec.source_family,
                rows=0,
                provider_time_proof_columns=[],
                operational_input_allowed_true_rows=0,
                release_latency_proven_true_rows=0,
            ),
        }

    frame = pd.read_parquet(spec.path)
    columns = [str(column) for column in frame.columns]
    provider_time_proof_columns = matching_time_proof_columns(columns)
    source_time = (
        pd.to_datetime(frame[spec.source_time_column], errors="coerce", utc=True)
        if spec.source_time_column in frame.columns
        else pd.Series(pd.NaT, index=frame.index)
    )
    operational_true = (
        truthy_count(frame["operational_input_allowed"])
        if "operational_input_allowed" in frame.columns
        else 0
    )
    release_true = (
        truthy_count(frame["release_latency_proven"])
        if "release_latency_proven" in frame.columns
        else 0
    )
    decision = source_unlock_decision(
        source_family=spec.source_family,
        rows=int(len(frame)),
        provider_time_proof_columns=provider_time_proof_columns,
        operational_input_allowed_true_rows=operational_true,
        release_latency_proven_true_rows=release_true,
    )
    return {
        "source_family": spec.source_family,
        "source_id": spec.source_id,
        "path": str(spec.path),
        "status": "present",
        "rows": int(len(frame)),
        "source_time_column": spec.source_time_column,
        "first_source_time": "" if source_time.dropna().empty else str(source_time.min()),
        "last_source_time": "" if source_time.dropna().empty else str(source_time.max()),
        "provider_time_proof_columns": ";".join(provider_time_proof_columns),
        "operational_input_allowed_true_rows": operational_true,
        "release_latency_proven_true_rows": release_true,
        "availability_tier_values": compact_values(frame["availability_tier"])
        if "availability_tier" in frame.columns
        else "",
        "source_time_policy_values": compact_values(frame[SOURCE_POLICY_COLUMN])
        if SOURCE_POLICY_COLUMN in frame.columns
        else "",
        "timing_contract": spec.timing_contract,
        **decision,
    }


def source_evidence_summary() -> pd.DataFrame:
    return pd.DataFrame([source_table_summary(spec) for spec in SOURCE_TABLES])


def load_0100_atlas() -> pd.DataFrame:
    if not INPUT_0100_ATLAS_PATH.exists():
        raise FileNotFoundError(f"0102 requires 0100 feature atlas first: {INPUT_0100_ATLAS_PATH}")
    atlas = pd.read_csv(INPUT_0100_ATLAS_PATH)
    required = {
        "feature",
        "family",
        "diagnostic_score",
        "timestamp_audit_status",
        "allowed_for_future_walkforward",
    }
    missing = sorted(required.difference(atlas.columns))
    if missing:
        raise ValueError(f"0100 feature atlas is missing required columns: {missing}")
    require_no_confirmation_dates(atlas["last_non_null_date"], context="0102 0100 atlas feature coverage")
    return atlas


def family_source_status(source_evidence: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family, group in source_evidence.groupby("source_family", sort=True, observed=True):
        unlockable = bool(group["unlock_decision"].astype(bool).all()) and not group.empty
        rows.append(
            {
                "family": family,
                "source_ids": ";".join(group["source_id"].astype(str)),
                "source_statuses": ";".join(sorted(group["status"].astype(str).unique())),
                "proof_statuses": ";".join(sorted(group["proof_status"].astype(str).unique())),
                "unlock_decision": unlockable,
                "source_rows": int(pd.to_numeric(group["rows"], errors="coerce").fillna(0).sum()),
                "provider_time_proof_columns": ";".join(
                    sorted(
                        {
                            value
                            for raw in group["provider_time_proof_columns"].astype(str)
                            for value in raw.split(";")
                            if value
                        }
                    )
                ),
                "operational_input_allowed_true_rows": int(
                    pd.to_numeric(group["operational_input_allowed_true_rows"], errors="coerce")
                    .fillna(0)
                    .sum()
                ),
                "release_latency_proven_true_rows": int(
                    pd.to_numeric(group["release_latency_proven_true_rows"], errors="coerce")
                    .fillna(0)
                    .sum()
                ),
                "required_next_evidence": "; ".join(
                    sorted(
                        {
                            value
                            for value in group["required_next_evidence"].dropna().astype(str)
                            if value.strip()
                        }
                    )
                ),
                "source_time_policy_values": " | ".join(
                    sorted(
                        {
                            value
                            for value in group["source_time_policy_values"].dropna().astype(str)
                            if value.strip()
                        }
                    )
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("family").reset_index(drop=True)


def feature_timing_basis(feature: str, family: str) -> str:
    if family == "upper_air":
        return (
            "Feature builder maps the value to target T from prior valid-time upper-air rows, "
            "but valid time is not first-available proof."
        )
    if family == "marine_proxy":
        return (
            "Target T feature is a lagged marine value from finalized HKO daily climate rows; "
            "publication lag is not row-proven."
        )
    if family == "hko_daily_climate":
        return (
            "Target T feature is a lagged finalized HKO daily climate value; first-publication "
            "time is not row-proven."
        )
    return f"Manual review required for {feature}."


def audit_feature_unlock_queue(atlas: pd.DataFrame, family_status: pd.DataFrame) -> pd.DataFrame:
    blocked = atlas[atlas["family"].isin(AUDITED_FAMILIES)].copy()
    blocked["diagnostic_score"] = pd.to_numeric(blocked["diagnostic_score"], errors="coerce")
    merged = blocked.merge(family_status, on="family", how="left", suffixes=("", "_source"))
    merged["post_0102_allowed_for_future_walkforward"] = merged["unlock_decision"].fillna(False).astype(bool)
    merged["post_0102_status"] = merged["post_0102_allowed_for_future_walkforward"].map(
        {True: "unlocked_for_candidate_testing", False: "still_diagnostic_only"}
    )
    merged["feature_timing_basis"] = [
        feature_timing_basis(str(row["feature"]), str(row["family"])) for row in merged.to_dict("records")
    ]
    merged["proof_gap"] = merged["required_next_evidence"].fillna("missing source family status")
    return (
        merged.sort_values(["post_0102_allowed_for_future_walkforward", "diagnostic_score", "feature"],
                           ascending=[False, False, True])
        .reset_index(drop=True)
    )


def build_family_unlock_summary(feature_audit: pd.DataFrame, source_evidence: pd.DataFrame) -> pd.DataFrame:
    feature_summary = (
        feature_audit.groupby("family", observed=True)
        .agg(
            feature_count=("feature", "count"),
            unlockable_feature_count=("post_0102_allowed_for_future_walkforward", "sum"),
            top_feature=("feature", "first"),
            top_diagnostic_score=("diagnostic_score", "max"),
            proof_statuses=("proof_statuses", "first"),
            required_next_evidence=("required_next_evidence", "first"),
        )
        .reset_index()
    )
    table_summary = (
        source_evidence.groupby("source_family", observed=True)
        .agg(
            source_ids=("source_id", lambda values: ";".join(values.astype(str))),
            source_rows=("rows", "sum"),
            provider_time_proof_columns=("provider_time_proof_columns", lambda values: ";".join(sorted(set(values)))),
            operational_input_allowed_true_rows=("operational_input_allowed_true_rows", "sum"),
            release_latency_proven_true_rows=("release_latency_proven_true_rows", "sum"),
        )
        .reset_index()
        .rename(columns={"source_family": "family"})
    )
    return (
        feature_summary.merge(table_summary, on="family", how="left")
        .sort_values(["unlockable_feature_count", "top_diagnostic_score"], ascending=[False, False])
        .reset_index(drop=True)
    )


def load_0101_summary() -> dict[str, object]:
    if not INPUT_0101_SUMMARY_PATH.exists():
        return {}
    return json.loads(INPUT_0101_SUMMARY_PATH.read_text(encoding="utf-8"))


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    atlas = load_0100_atlas()
    source_evidence = source_evidence_summary()
    family_status = family_source_status(source_evidence)
    feature_audit = audit_feature_unlock_queue(atlas, family_status)
    family_summary = build_family_unlock_summary(feature_audit, source_evidence)
    unlockable_count = int(feature_audit["post_0102_allowed_for_future_walkforward"].astype(bool).sum())
    source_last_times = pd.to_datetime(source_evidence["last_source_time"], errors="coerce", utc=True)
    metadata_includes_2024_plus = bool(source_last_times.dropna().ge(CONFIRMATION_START.tz_localize("UTC")).any())
    input_0101 = load_0101_summary()
    summary = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "input_0101_best_candidate": input_0101.get("best_candidate", ""),
        "input_0101_best_mae": input_0101.get("best_mae", math.nan),
        "input_0101_best_rmse": input_0101.get("best_rmse", math.nan),
        "source_tables_audited": int(len(source_evidence)),
        "families_audited": int(feature_audit["family"].nunique()),
        "features_audited": int(len(feature_audit)),
        "unlockable_feature_count": unlockable_count,
        "still_blocked_feature_count": int(len(feature_audit) - unlockable_count),
        "rerun_0101_with_newly_eligible_families": bool(unlockable_count > 0),
        "confirmation_start": CONFIRMATION_START.date().isoformat(),
        "uses_2024_plus_target_rows": False,
        "source_metadata_coverage_includes_2024_plus": metadata_includes_2024_plus,
        "status": "timestamp_proof_unlock_queue_complete_no_unlocks"
        if unlockable_count == 0
        else "timestamp_proof_unlock_queue_complete_with_unlocks",
        "next_recommended_task": (
            "Run 0103_current_rss_continuation_without_blocked_sources: continue leakage-free analysis using "
            "the current scoreable RSS/press archive plus already future-allowed station and target-memory "
            "features while the forecast backfill continues; keep upper-air, daily climate, and marine proxies "
            "diagnostic-only until provider timestamp proof arrives."
        ),
    }
    return source_evidence, feature_audit, family_summary, summary


def build_readme(
    *,
    source_evidence: pd.DataFrame,
    feature_audit: pd.DataFrame,
    family_summary: pd.DataFrame,
    summary: dict[str, object],
) -> str:
    source_cols = [
        "source_family",
        "source_id",
        "rows",
        "first_source_time",
        "last_source_time",
        "provider_time_proof_columns",
        "operational_input_allowed_true_rows",
        "release_latency_proven_true_rows",
        "proof_status",
        "required_next_evidence",
    ]
    feature_cols = [
        "feature",
        "family",
        "diagnostic_score",
        "timestamp_audit_status",
        "post_0102_status",
        "proof_statuses",
        "source_ids",
        "proof_gap",
    ]
    return f"""# 0102 Timestamp Proof Unlock Queue

Generated: `{summary['generated_at_utc']}`

## Purpose

`0100` found that several upper-air and HKO daily/marine features explain residual behavior very well. `0101` did the correct safe thing and excluded them from candidate testing because their provider issue, available-at, or first-publication timestamps were not attached.

`0102` audits the local normalized source tables to answer one narrow question: can any of those high-scoring diagnostic-only families be unlocked now without violating point-in-time safety?

## Result

| Field | Value |
|---|---|
| Status | `{summary['status']}` |
| Source tables audited | `{summary['source_tables_audited']}` |
| Families audited | `{summary['families_audited']}` |
| Features audited | `{summary['features_audited']}` |
| Unlockable features | `{summary['unlockable_feature_count']}` |
| Still blocked features | `{summary['still_blocked_feature_count']}` |
| Rerun 0101 with newly eligible families | `{summary['rerun_0101_with_newly_eligible_families']}` |
| 2024+ target rows used | `{summary['uses_2024_plus_target_rows']}` |
| Source metadata coverage includes 2024+ | `{summary['source_metadata_coverage_includes_2024_plus']}` |

## Interpretation

No blocked family is promoted by this audit if `unlockable_feature_count` is zero. The raw weather values may be old enough in meteorological time, but the local archive still does not prove that every row was available before the operational forecast cutoff. That distinction matters: a valid-time field describes when the atmosphere was observed, not when the model would have been allowed to know the data.

## Source Evidence

{markdown_table(source_evidence[source_cols], max_rows=20)}

## Family Unlock Summary

{markdown_table(family_summary, max_rows=20)}

## Highest-Priority Blocked Feature Queue

{markdown_table(feature_audit[feature_cols].head(35), max_rows=35)}

## Leakage Controls

This audit does not score predictions, train a model, or inspect 2024+ target outcomes. It reads source metadata through the currently available archive only to check whether timestamp-proof columns exist. All `0100` feature coverage rows used to build the queue remain before `{summary['confirmation_start']}`.

## Required Evidence Before Unlocking

Upper-air rows need provider-grade `issued_at`, `available_at`, or all-row release-latency proof before the `T-1 15:00 HKT` cutoff. HKO daily climate and marine proxy rows need first-publication or available-at proof for the lagged daily values. Archive retrieval timestamps alone do not satisfy this requirement.

## Exact Next Recommended Task

{summary['next_recommended_task']}
"""


def update_milestones(
    summary: dict[str, object],
    source_evidence: pd.DataFrame,
    feature_audit: pd.DataFrame,
    family_summary: pd.DataFrame,
) -> None:
    feature_cols = [
        "feature",
        "family",
        "diagnostic_score",
        "post_0102_status",
        "proof_statuses",
        "proof_gap",
    ]
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0102_timestamp_proof_unlock_queue.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Current evidence | Status |
|---|---|---|
| Input 0101 best | `{summary['input_0101_best_candidate']}` | Current pre-2024 research champion |
| Input 0101 MAE/RMSE | `{summary['input_0101_best_mae']}` / `{summary['input_0101_best_rmse']}` | Pre-2024 only |
| Source tables audited | `{summary['source_tables_audited']}` | IGRA + HKO daily climate |
| Blocked features audited | `{summary['features_audited']}` | Upper-air + HKO daily/marine |
| Unlockable features | `{summary['unlockable_feature_count']}` | Timestamp-proof gate |
| Rerun 0101 with new families | `{summary['rerun_0101_with_newly_eligible_families']}` | Gate result |
| Leakage | `0` 2024+ target rows | PASS |

Source evidence:

{markdown_table(source_evidence[["source_family", "source_id", "rows", "proof_status", "required_next_evidence"]], max_rows=20)}

Family summary:

{markdown_table(family_summary, max_rows=20)}

Top blocked queue:

{markdown_table(feature_audit[feature_cols].head(15), max_rows=15)}
"""
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="0102 Timestamp Proof Unlock Queue",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Exact Next Recommended Codex Task",
        section=str(summary["next_recommended_task"]),
    )


def run() -> dict[str, object]:
    source_evidence, feature_audit, family_summary, summary = build_outputs()
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    write_csv(artifacts / "source_evidence_summary.csv", source_evidence)
    write_csv(artifacts / "feature_unlock_audit.csv", feature_audit)
    write_csv(
        artifacts / "blocked_feature_queue.csv",
        feature_audit[~feature_audit["post_0102_allowed_for_future_walkforward"].astype(bool)].copy(),
    )
    write_csv(artifacts / "family_unlock_summary.csv", family_summary)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "timestamp_proof_unlock_queue_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            source_evidence=source_evidence,
            feature_audit=feature_audit,
            family_summary=family_summary,
            summary=summary,
        ),
    )
    update_milestones(summary, source_evidence, feature_audit, family_summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-summary", action="store_true", help="Print JSON summary after writing artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run()
    if args.print_summary:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
