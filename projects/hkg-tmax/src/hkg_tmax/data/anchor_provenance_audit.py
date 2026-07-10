"""Forecast-anchor provenance audit for HKG Tmax early cutoffs."""

from __future__ import annotations

import json
from datetime import time
from typing import Any

import numpy as np
import pandas as pd

from hkg_tmax.data.forecast_anchor import CUTOFF_PROFILES, CutoffProfile, cutoff_timestamps, load_sql_frame


AUDIT_CUTOFFS = {
    "tminus1_1500": CutoffProfile("tminus1_1500", time(15, 0)),
    "tminus1_1630": CutoffProfile("tminus1_1630", time(16, 30)),
    "tminus1_1800": CutoffProfile("tminus1_1800", time(18, 0)),
    "tminus1_2100": CutoffProfile("tminus1_2100", time(21, 0)),
    "tminus1_2359": CutoffProfile("tminus1_2359", time(23, 59)),
    **{profile.name: profile for profile in CUTOFF_PROFILES},
}

ALL_FORECAST_CANDIDATES_SQL = """
SELECT
  bulletin_id,
  source,
  source_url,
  product_type,
  title,
  index_date,
  snapshot_at_hkt,
  snapshot_at_utc,
  issue_at_hkt,
  issue_at_utc,
  target_date::date AS target_date,
  target_issue_lead_days,
  forecast_min_c::double precision AS forecast_min_c,
  forecast_max_c::double precision AS forecast_max_c,
  row_quality_status,
  temperature_text,
  parse_status,
  parse_notes,
  full_text,
  raw_sha256,
  ingested_at_utc
FROM public.hko_historical_forecasts_2000_2026
WHERE target_date BETWEEN %(start_date)s AND %(sealed_end_date)s
ORDER BY target_date, issue_at_utc, source_url
"""


def load_all_forecast_candidates(connection: Any, params: dict[str, Any]) -> pd.DataFrame:
    forecasts = load_sql_frame(connection, ALL_FORECAST_CANDIDATES_SQL, params)
    if forecasts.empty:
        return forecasts
    forecasts["target_date"] = pd.to_datetime(forecasts["target_date"], errors="coerce").dt.normalize()
    forecasts["issue_at_utc"] = pd.to_datetime(forecasts["issue_at_utc"], errors="coerce", utc=True)
    forecasts["issue_at_hkt"] = pd.to_datetime(forecasts["issue_at_hkt"], errors="coerce")
    forecasts["forecast_min_c"] = pd.to_numeric(forecasts["forecast_min_c"], errors="coerce")
    forecasts["forecast_max_c"] = pd.to_numeric(forecasts["forecast_max_c"], errors="coerce")
    return forecasts.dropna(subset=["target_date"]).reset_index(drop=True)


def _profile(profile: str | CutoffProfile) -> CutoffProfile:
    if isinstance(profile, CutoffProfile):
        return profile
    if profile not in AUDIT_CUTOFFS:
        raise KeyError(f"Unknown cutoff profile for provenance audit: {profile}")
    return AUDIT_CUTOFFS[profile]


def _counts(series: pd.Series) -> str:
    if series.empty:
        return "{}"
    return json.dumps(series.fillna("MISSING").astype(str).value_counts().to_dict(), sort_keys=True)


def _strict_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.get("source", "").astype(str).eq("info_gov")
        & frame.get("product_type", "").astype(str).eq("local")
        & frame.get("row_quality_status", "").astype(str).eq("usable_local_minmax")
        & pd.to_numeric(frame.get("target_issue_lead_days", np.nan), errors="coerce").eq(1)
        & pd.to_numeric(frame.get("forecast_max_c", np.nan), errors="coerce").notna()
        & pd.to_numeric(frame.get("forecast_min_c", np.nan), errors="coerce").notna()
        & pd.to_datetime(frame.get("issue_at_utc", pd.NaT), errors="coerce", utc=True).notna()
    )


def _reason_no_anchor(group: pd.DataFrame, before_cutoff: pd.DataFrame, strict_before: pd.DataFrame, strict_after: pd.DataFrame) -> str:
    if not strict_before.empty:
        return "selected"
    if group.empty:
        return "no_forecast_rows_for_target"
    if before_cutoff.empty and not strict_after.empty:
        return "next_strict_eligible_after_cutoff"
    if before_cutoff.empty:
        return "no_forecast_rows_before_cutoff"
    local = before_cutoff[before_cutoff.get("source", "").astype(str).eq("info_gov") & before_cutoff.get("product_type", "").astype(str).eq("local")]
    if local.empty:
        return "no_info_gov_local_rows_before_cutoff"
    lead1 = local[pd.to_numeric(local.get("target_issue_lead_days", np.nan), errors="coerce").eq(1)]
    if lead1.empty:
        return "no_info_gov_local_lead1_rows_before_cutoff"
    usable = lead1[lead1.get("row_quality_status", "").astype(str).eq("usable_local_minmax")]
    if usable.empty:
        return "no_usable_local_minmax_before_cutoff"
    minmax = usable[
        pd.to_numeric(usable.get("forecast_max_c", np.nan), errors="coerce").notna()
        & pd.to_numeric(usable.get("forecast_min_c", np.nan), errors="coerce").notna()
    ]
    if minmax.empty:
        return "usable_local_rows_missing_minmax_before_cutoff"
    return "unknown_strict_selector_gap"


def build_anchor_provenance_audit(
    targets: pd.DataFrame,
    forecasts_all: pd.DataFrame,
    cutoff_profiles: list[CutoffProfile] | list[str],
) -> pd.DataFrame:
    if targets.empty:
        return pd.DataFrame()
    target_frame = targets.copy()
    target_frame["target_date"] = pd.to_datetime(target_frame["target_date"], errors="coerce").dt.normalize()
    forecast_frame = forecasts_all.copy()
    if forecast_frame.empty:
        forecast_frame = pd.DataFrame(columns=["target_date", "issue_at_utc", "issue_at_hkt"])
    forecast_frame["target_date"] = pd.to_datetime(forecast_frame["target_date"], errors="coerce").dt.normalize()
    forecast_frame["issue_at_utc"] = pd.to_datetime(forecast_frame["issue_at_utc"], errors="coerce", utc=True)
    rows: list[dict[str, Any]] = []
    grouped = {date: group.sort_values("issue_at_utc") for date, group in forecast_frame.groupby("target_date", dropna=False)}
    for target in target_frame.itertuples(index=False):
        target_date = pd.Timestamp(target.target_date).normalize()
        group = grouped.get(target_date, forecast_frame.iloc[0:0])
        strict_all = group[_strict_mask(group)] if not group.empty else group
        for item in cutoff_profiles:
            profile = _profile(item)
            cutoff_hkt, cutoff_utc = cutoff_timestamps(target_date, profile)
            before = group[group["issue_at_utc"].le(cutoff_utc)] if not group.empty else group
            after = group[group["issue_at_utc"].gt(cutoff_utc)] if not group.empty else group
            strict_before = strict_all[strict_all["issue_at_utc"].le(cutoff_utc)] if not strict_all.empty else strict_all
            strict_after = strict_all[strict_all["issue_at_utc"].gt(cutoff_utc)] if not strict_all.empty else strict_all
            info_gov = before[before.get("source", "").astype(str).eq("info_gov")] if not before.empty else before
            info_gov_local = info_gov[info_gov.get("product_type", "").astype(str).eq("local")] if not info_gov.empty else info_gov
            lead = pd.to_numeric(before.get("target_issue_lead_days", np.nan), errors="coerce") if not before.empty else pd.Series(dtype=float)
            usable_local_minmax = info_gov_local[
                info_gov_local.get("row_quality_status", "").astype(str).eq("usable_local_minmax")
                & pd.to_numeric(info_gov_local.get("forecast_max_c", np.nan), errors="coerce").notna()
                & pd.to_numeric(info_gov_local.get("forecast_min_c", np.nan), errors="coerce").notna()
            ] if not info_gov_local.empty else info_gov_local
            nearest_after = strict_after["issue_at_utc"].min() if not strict_after.empty else pd.NaT
            minutes_to_next = (
                float((nearest_after - cutoff_utc).total_seconds() / 60.0)
                if pd.notna(nearest_after)
                else np.nan
            )
            rows.append(
                {
                    "target_date": target_date,
                    "cutoff_profile": profile.name,
                    "cutoff_at_hkt": cutoff_hkt,
                    "strict_selected_anchor_status": "selected" if not strict_before.empty else "no_eligible_anchor",
                    "strict_candidate_count": int(len(strict_before)),
                    "all_info_gov_local_count": int(len(info_gov_local)),
                    "all_info_gov_any_quality_count": int(len(info_gov)),
                    "all_lead0_count": int(lead.eq(0).sum()) if not before.empty else 0,
                    "all_lead1_count": int(lead.eq(1).sum()) if not before.empty else 0,
                    "usable_local_minmax_count": int(len(usable_local_minmax)),
                    "parse_status_counts": _counts(before.get("parse_status", pd.Series(dtype=str))) if not before.empty else "{}",
                    "row_quality_status_counts": _counts(before.get("row_quality_status", pd.Series(dtype=str))) if not before.empty else "{}",
                    "min_issue_at_hkt": before["issue_at_hkt"].min() if not before.empty and "issue_at_hkt" in before else pd.NaT,
                    "max_issue_at_hkt_before_cutoff": before["issue_at_hkt"].max() if not before.empty and "issue_at_hkt" in before else pd.NaT,
                    "nearest_issue_after_cutoff_hkt": after["issue_at_hkt"].min() if not after.empty and "issue_at_hkt" in after else pd.NaT,
                    "minutes_to_next_eligible_after_cutoff": minutes_to_next,
                    "reason_no_anchor": _reason_no_anchor(group, before, strict_before, strict_after),
                }
            )
    return pd.DataFrame(rows).sort_values(["cutoff_profile", "target_date"]).reset_index(drop=True)


def summarize_anchor_provenance(audit: pd.DataFrame) -> dict[str, Any]:
    if audit.empty:
        return {"status": "empty", "cutoffs": {}}
    frame = audit.copy()
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce")
    cutoffs: dict[str, Any] = {}
    for cutoff, group in frame.groupby("cutoff_profile"):
        modern = group[group["target_date"].dt.year >= 2011]
        cutoffs[str(cutoff)] = {
            "rows": int(len(group)),
            "selected_rows": int(group["strict_selected_anchor_status"].eq("selected").sum()),
            "coverage_pct": float(group["strict_selected_anchor_status"].eq("selected").mean() * 100.0) if len(group) else 0.0,
            "coverage_pct_2011_onward": float(modern["strict_selected_anchor_status"].eq("selected").mean() * 100.0)
            if len(modern)
            else 0.0,
            "top_missing_reasons": group[group["strict_selected_anchor_status"].ne("selected")]["reason_no_anchor"]
            .value_counts()
            .head(10)
            .to_dict(),
            "eligible_for_future_modeling": bool(
                len(modern) > 0 and modern["strict_selected_anchor_status"].eq("selected").mean() >= 0.80
            ),
        }
    return {
        "status": "pass",
        "cutoffs": cutoffs,
        "primary_interpretation": "Earlier cutoffs may enter ML only when coverage from 2011 onward is at least 80 percent and source identity remains apples-to-apples.",
    }
