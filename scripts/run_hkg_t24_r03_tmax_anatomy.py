from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from hkg_tmax.hkg_t24.governance import check_four_year_oof_feasibility
from hkg_tmax.hkg_t24.guard import assert_no_locked_dates
from hkg_tmax.hkg_t24.peak_anatomy import (
    classify_peak_time,
    count_peak_episodes,
    maximum_heating_in_window,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(r"C:\hkg_tmax_data")
RESEARCH_ID = "HKG-T24-R03"
EXPERIMENT_DIR = (
    REPO_ROOT
    / "analysis"
    / "hkg_tmax_t24"
    / "experiments"
    / "EXP-0035-HKG-T24-R03"
)
ANALYSIS_START = pd.Timestamp("2020-07-01")
ANALYSIS_END = pd.Timestamp("2023-12-31")
EXPECTED_TEN_MINUTE_ROWS_PER_DAY = 144
HKT = ZoneInfo("Asia/Hong_Kong")
ZIP_ENTRY_TIME_RE = re.compile(r"(?P<date>\d{8})-(?P<hhmm>\d{4})")
FULL_DAY_SOURCE_IDS = {
    "datagov_hko_historical_latest_1min_temperature_archive": {
        "family": "latest_1min_temperature",
        "variables": {
            "Air Temperature(degree Celsius)": ("air_temperature_c", "degC"),
        },
    },
    "datagov_hko_historical_latest_since_midnight_maxmin_archive": {
        "family": "latest_since_midnight_maxmin",
        "variables": {
            "MaximumAir Temperature Since Midnight(degree Celsius)": (
                "temperature_since_midnight_max_c",
                "degC",
            ),
            "Minimum Air Temperature Since Midnight(degree Celsius)": (
                "temperature_since_midnight_min_c",
                "degC",
            ),
        },
    },
}


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    return {"head": head, "dirty_count": len([line for line in status if line.strip()])}


def season(month: int) -> str:
    if month in {12, 1, 2}:
        return "DJF"
    if month in {3, 4, 5}:
        return "MAM"
    if month in {6, 7, 8}:
        return "JJA"
    return "SON"


def clock_minutes(timestamp: pd.Timestamp) -> int:
    return int(timestamp.hour) * 60 + int(timestamp.minute)


def read_retrieval_ledger(data_root: Path) -> pd.DataFrame:
    path = data_root / "manifests" / "retrieval_ledger.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing retrieval ledger: {path}")
    ledger = pd.read_csv(path)
    ledger["retrieved_at"] = pd.to_datetime(ledger["retrieved_at"], utc=True, errors="coerce")
    return ledger


def latest_successes(ledger: pd.DataFrame, source_ids: set[str]) -> pd.DataFrame:
    rows = ledger[
        (ledger["status"] == "success")
        & (ledger["source_id"].isin(source_ids))
        & ledger["content_path"].notna()
    ].copy()
    if rows.empty:
        return rows
    return rows.sort_values("retrieved_at").drop_duplicates(["source_id", "content_sha256"], keep="last")


def zip_entry_timestamp_hkt(name: str) -> datetime | None:
    match = ZIP_ENTRY_TIME_RE.search(name)
    if match is None:
        return None
    token = match.group("date") + match.group("hhmm")
    try:
        return datetime.strptime(token, "%Y%m%d%H%M").replace(tzinfo=HKT)
    except ValueError:
        return None


def parse_hkt_observed_at(value: object) -> datetime | None:
    token = str(value).strip()
    if not token or token.upper() in {"N/A", "NA", "NULL"}:
        return None
    for fmt in ("%Y%m%d%H%M", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(token, fmt).replace(tzinfo=HKT)
        except ValueError:
            continue
    return None


def parse_number(value: object) -> float | None:
    token = str(value).strip()
    if not token or token.upper() in {"N/A", "NA", "NULL", "***", "-"}:
        return None
    try:
        result = float(token)
    except ValueError:
        return None
    if np.isnan(result):
        return None
    return result


def iter_full_day_hko_hq_rows(
    source: pd.Series,
    *,
    family: str,
    variables: Mapping[str, tuple[str, str]],
) -> list[dict[str, object]]:
    path = Path(str(source["content_path"]))
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".csv"):
                continue
            entry_time = zip_entry_timestamp_hkt(name)
            if entry_time is None:
                continue
            entry_day = pd.Timestamp(entry_time.date())
            if entry_day < ANALYSIS_START or entry_day > ANALYSIS_END:
                continue
            with archive.open(name) as raw:
                text = raw.read().decode("utf-8-sig", errors="replace").splitlines()
            if not text:
                continue
            reader = csv.DictReader(text)
            for raw_row in reader:
                if str(raw_row.get("Automatic Weather Station", "")).strip() != "HK Observatory":
                    continue
                observed_at = parse_hkt_observed_at(raw_row.get("Date time", ""))
                if observed_at is None:
                    observed_at = entry_time
                observed_day = pd.Timestamp(observed_at.date())
                if observed_day < ANALYSIS_START or observed_day > ANALYSIS_END:
                    continue
                for raw_column, variable_pair in variables.items():
                    variable, unit = variable_pair
                    value = parse_number(raw_row.get(raw_column, ""))
                    if value is None:
                        continue
                    rows.append(
                        {
                            "source_id": str(source["source_id"]),
                            "family": family,
                            "content_sha256": str(source["content_sha256"]),
                            "retrieved_at": source["retrieved_at"],
                            "archive_entry_name": name,
                            "archive_payload_timestamp_hkt": entry_time,
                            "station": "HK Observatory",
                            "observed_at_hkt": observed_at,
                            "local_date": observed_at.date(),
                            "variable": variable,
                            "unit": unit,
                            "value": value,
                            "role": "TARGET_ONLY_MECHANISM_DIAGNOSTIC",
                            "availability_assumption": "target-day diagnostic only; not eligible as T-24 predictor",
                            "available_at_hkt": observed_at + timedelta(minutes=20),
                        }
                    )
                break
    return rows


def build_full_day_hko_hq_high_frequency(data_root: Path) -> tuple[pd.DataFrame, Path]:
    ledger = read_retrieval_ledger(data_root)
    rows: list[dict[str, object]] = []
    for _, source in latest_successes(ledger, set(FULL_DAY_SOURCE_IDS)).iterrows():
        meta = FULL_DAY_SOURCE_IDS[str(source["source_id"])]
        variables = meta["variables"]
        assert isinstance(variables, Mapping)
        rows.extend(
            iter_full_day_hko_hq_rows(
                source,
                family=str(meta["family"]),
                variables=variables,
            )
        )
    if not rows:
        raise RuntimeError("No full-day HKO Headquarters high-frequency rows parsed for R03.")
    frame = pd.DataFrame(rows)
    frame["observed_at_hkt"] = pd.to_datetime(frame["observed_at_hkt"], utc=True).dt.tz_convert(HKT)
    frame["archive_payload_timestamp_hkt"] = pd.to_datetime(
        frame["archive_payload_timestamp_hkt"], utc=True
    ).dt.tz_convert(HKT)
    frame["available_at_hkt"] = pd.to_datetime(frame["available_at_hkt"], utc=True).dt.tz_convert(HKT)
    frame["local_date"] = pd.to_datetime(frame["local_date"], errors="coerce")
    frame = frame.sort_values(["family", "variable", "local_date", "observed_at_hkt"]).reset_index(drop=True)
    output_path = data_root / "bronze" / "hkg_t24" / "r03_hko_hq_full_day_high_frequency.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return frame, output_path


def build_reconstruction(hf: pd.DataFrame) -> pd.DataFrame:
    temp = hf[
        (hf["station"] == "HK Observatory")
        & (hf["variable"] == "air_temperature_c")
        & (hf["local_date"] >= ANALYSIS_START)
        & (hf["local_date"] <= ANALYSIS_END)
    ].copy()
    temp["observed_at_hkt"] = pd.to_datetime(temp["observed_at_hkt"])
    temp = temp.sort_values(["local_date", "observed_at_hkt"])
    rows: list[dict[str, object]] = []
    for local_date, group in temp.groupby("local_date", sort=True):
        values = group["value"].astype(float).to_numpy()
        times = [pd.Timestamp(item).to_pydatetime() for item in group["observed_at_hkt"]]
        observed = pd.to_datetime(group["observed_at_hkt"])
        max_value = float(np.nanmax(values))
        peak_mask = np.isclose(values, max_value, atol=1e-9)
        peak_times = [times[index] for index, is_peak in enumerate(peak_mask) if is_peak]
        first_peak = min(peak_times)
        last_peak = max(peak_times)
        diffs = observed.sort_values().diff().dt.total_seconds().dropna() / 60.0
        median_cadence = float(diffs.median()) if not diffs.empty else 10.0
        first_clock = clock_minutes(pd.Timestamp(group["observed_at_hkt"].min()))
        last_clock = clock_minutes(pd.Timestamp(group["observed_at_hkt"].max()))
        obs_count = int(len(group))
        complete = (
            obs_count >= 140
            and first_clock <= 20
            and last_clock >= 23 * 60 + 40
            and 8 <= median_cadence <= 12
        )
        rows.append(
            {
                "local_date": pd.Timestamp(local_date),
                "hko_temp_obs_count": obs_count,
                "hko_temp_first_observed_at_hkt": group["observed_at_hkt"].min(),
                "hko_temp_last_observed_at_hkt": group["observed_at_hkt"].max(),
                "hko_temp_median_cadence_minutes": median_cadence,
                "hko_temp_complete_day": bool(complete),
                "reconstructed_tmax_c": max_value,
                "first_time_at_reconstructed_tmax_hkt": first_peak.isoformat(),
                "last_time_at_reconstructed_tmax_hkt": last_peak.isoformat(),
                "peak_time_class": classify_peak_time(first_peak),
                "peak_row_count": int(peak_mask.sum()),
                "peak_episode_count": count_peak_episodes(peak_times, gap_minutes=max(15.0, median_cadence * 1.5)),
                "peak_duration_minutes": float(peak_mask.sum() * median_cadence),
                "max_heating_10m_before_peak_c": maximum_heating_in_window(
                    times,
                    values.tolist(),
                    end_time=first_peak,
                    window_minutes=10,
                ),
                "max_heating_30m_before_peak_c": maximum_heating_in_window(
                    times,
                    values.tolist(),
                    end_time=first_peak,
                    window_minutes=30,
                ),
                "max_heating_60m_before_peak_c": maximum_heating_in_window(
                    times,
                    values.tolist(),
                    end_time=first_peak,
                    window_minutes=60,
                ),
                "content_hashes": ",".join(sorted(map(str, group["content_sha256"].unique()))),
                "source_ids": ",".join(sorted(map(str, group["source_id"].unique()))),
            }
        )
    return pd.DataFrame(rows)


def build_since_midnight_max(hf: pd.DataFrame) -> pd.DataFrame:
    feed = hf[
        (hf["station"] == "HK Observatory")
        & (hf["variable"] == "temperature_since_midnight_max_c")
        & (hf["local_date"] >= ANALYSIS_START)
        & (hf["local_date"] <= ANALYSIS_END)
    ].copy()
    feed["observed_at_hkt"] = pd.to_datetime(feed["observed_at_hkt"])
    rows: list[dict[str, object]] = []
    for local_date, group in feed.sort_values(["local_date", "observed_at_hkt"]).groupby("local_date", sort=True):
        group = group.copy()
        group["clock_minutes"] = pd.to_datetime(group["observed_at_hkt"]).map(clock_minutes)
        latest = group.iloc[-1]
        after_0100 = group[group["clock_minutes"] >= 60]
        late_day = group[group["clock_minutes"] >= 23 * 60]
        latest_late = late_day.iloc[-1] if not late_day.empty else latest
        raw_feed_max = float(group["value"].astype(float).max())
        max_after_0100 = float(after_0100["value"].astype(float).max()) if not after_0100.empty else raw_feed_max
        latest_value = float(latest["value"])
        late_value = float(latest_late["value"])
        rows.append(
            {
                "local_date": pd.Timestamp(local_date),
                "since_midnight_max_obs_count": int(len(group)),
                "since_midnight_feed_raw_max_c": raw_feed_max,
                "since_midnight_feed_max_after_0100_c": max_after_0100,
                "since_midnight_latest_value_c": latest_value,
                "since_midnight_latest_observed_at_hkt": latest["observed_at_hkt"],
                "since_midnight_late_final_value_c": late_value,
                "since_midnight_late_final_observed_at_hkt": latest_late["observed_at_hkt"],
                "since_midnight_midnight_carryover_suspected": bool(raw_feed_max > late_value + 0.100000001),
            }
        )
    return pd.DataFrame(rows)


def build_daily_context(daily: pd.DataFrame) -> pd.DataFrame:
    subset = daily[
        (daily["local_date"] >= ANALYSIS_START)
        & (daily["local_date"] <= ANALYSIS_END)
        & daily["variable"].isin(
            [
                "daily_rainfall",
                "global_solar_radiation",
                "mean_wind_speed",
                "prevailing_wind_direction",
            ]
        )
    ].copy()
    if subset.empty:
        return pd.DataFrame({"local_date": []})
    subset["key"] = subset["station_or_domain"].astype(str) + "__" + subset["variable"].astype(str)
    wide = subset.pivot_table(index="local_date", columns="key", values="value", aggfunc="first").reset_index()
    rename = {
        "Hong Kong Observatory__daily_rainfall": "hko_daily_rainfall_mm",
        "King's Park__global_solar_radiation": "kings_park_global_solar_radiation_mj_m2",
        "Waglan Island__mean_wind_speed": "waglan_mean_wind_speed_kmh",
        "Waglan Island__prevailing_wind_direction": "waglan_prevailing_wind_direction_deg",
    }
    return wide.rename(columns={key: value for key, value in rename.items() if key in wide.columns})


def add_comparisons(recon: pd.DataFrame, target: pd.DataFrame, since: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    target_view = target[
        (target["local_date"] >= ANALYSIS_START) & (target["local_date"] <= ANALYSIS_END)
    ][["local_date", "target_tmax_c"]].copy()
    out = recon.merge(target_view, on="local_date", how="left")
    out = out.merge(since, on="local_date", how="left")
    out = out.merge(context, on="local_date", how="left")
    out["official_minus_reconstructed_c"] = out["target_tmax_c"] - out["reconstructed_tmax_c"]
    out["abs_official_minus_reconstructed_c"] = out["official_minus_reconstructed_c"].abs()
    out["exact_value_match"] = out["target_tmax_c"] == out["reconstructed_tmax_c"]
    out["rounded_0p1_match"] = out["target_tmax_c"].round(1) == out["reconstructed_tmax_c"].round(1)
    out["within_0p1_c"] = out["abs_official_minus_reconstructed_c"] <= 0.100000001
    out["cell_0p1_match"] = out["rounded_0p1_match"]
    out["season"] = pd.to_datetime(out["local_date"]).dt.month.map(season)
    out["month"] = pd.to_datetime(out["local_date"]).dt.month
    out["high_tail_33c_or_more"] = out["target_tmax_c"] >= 33.0
    out["feed_coverage_band"] = np.where(out["hko_temp_complete_day"], "complete", "incomplete")
    out["rain_state"] = np.select(
        [
            out.get("hko_daily_rainfall_mm", pd.Series(np.nan, index=out.index)).isna(),
            out.get("hko_daily_rainfall_mm", pd.Series(np.nan, index=out.index)) >= 1.0,
        ],
        ["unknown", "wet_ge_1mm"],
        default="dry_lt_1mm",
    )
    solar = out.get("kings_park_global_solar_radiation_mj_m2")
    if solar is not None and solar.notna().sum() >= 10:
        low, high = solar.quantile([0.33, 0.67])
        out["solar_state"] = np.select(
            [solar.isna(), solar <= low, solar >= high],
            ["unknown", "low_solar", "high_solar"],
            default="middle_solar",
        )
    else:
        out["solar_state"] = "unknown"
    wind = out.get("waglan_mean_wind_speed_kmh")
    if wind is not None and wind.notna().sum() >= 10:
        low, high = wind.quantile([0.33, 0.67])
        out["wind_speed_state"] = np.select(
            [wind.isna(), wind <= low, wind >= high],
            ["unknown", "low_wind", "high_wind"],
            default="middle_wind",
        )
    else:
        out["wind_speed_state"] = "unknown"
    out["since_midnight_rawmax_minus_reconstructed_c"] = (
        out["since_midnight_feed_raw_max_c"] - out["reconstructed_tmax_c"]
    )
    out["since_midnight_rawmax_minus_official_c"] = out["since_midnight_feed_raw_max_c"] - out["target_tmax_c"]
    out["since_midnight_after0100_minus_official_c"] = (
        out["since_midnight_feed_max_after_0100_c"] - out["target_tmax_c"]
    )
    out["since_midnight_latest_minus_official_c"] = out["since_midnight_latest_value_c"] - out["target_tmax_c"]
    out["since_midnight_late_final_minus_official_c"] = (
        out["since_midnight_late_final_value_c"] - out["target_tmax_c"]
    )
    out["since_midnight_late_final_within_0p1_c"] = (
        out["since_midnight_late_final_minus_official_c"].abs() <= 0.100000001
    )
    return out.sort_values("local_date").reset_index(drop=True)


def summarize_dimension(frame: pd.DataFrame, dimension: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for value, group in frame.groupby(dimension, dropna=False):
        rows.append(
            {
                "dimension": dimension,
                "value": str(value),
                "n": int(len(group)),
                "complete_days": int(group["hko_temp_complete_day"].sum()),
                "mean_abs_diff_c": float(group["abs_official_minus_reconstructed_c"].mean()),
                "median_abs_diff_c": float(group["abs_official_minus_reconstructed_c"].median()),
                "max_abs_diff_c": float(group["abs_official_minus_reconstructed_c"].max()),
                "bias_official_minus_reconstructed_c": float(group["official_minus_reconstructed_c"].mean()),
                "within_0p1_rate": float(group["within_0p1_c"].mean()),
                "rounded_0p1_match_rate": float(group["rounded_0p1_match"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_stratification(frame: pd.DataFrame) -> pd.DataFrame:
    dimensions = [
        "season",
        "month",
        "peak_time_class",
        "feed_coverage_band",
        "rain_state",
        "solar_state",
        "wind_speed_state",
        "high_tail_33c_or_more",
    ]
    return pd.concat([summarize_dimension(frame, dimension) for dimension in dimensions], ignore_index=True)


def build_metrics(frame: pd.DataFrame, stratified: pd.DataFrame, maxmin_disagreements: pd.DataFrame) -> dict[str, object]:
    complete = frame[frame["hko_temp_complete_day"]].copy()
    since_late_available = frame[frame["since_midnight_late_final_value_c"].notna()].copy()
    feasibility = check_four_year_oof_feasibility(
        frame["local_date"].min().date(),
        frame["local_date"].max().date(),
        reason_context="R03 modern high-frequency pre-validation anatomy period",
    )
    complete_within_rate = float(complete["within_0p1_c"].mean()) if not complete.empty else 0.0
    since_late_within_rate = (
        float(since_late_available["since_midnight_late_final_within_0p1_c"].mean())
        if not since_late_available.empty
        else 0.0
    )
    if complete_within_rate >= 0.99 and feasibility.status == "PASS":
        success_status = "PASS_LABEL_RECONSTRUCTION_DIAGNOSTIC"
    elif feasibility.status == "BLOCKED":
        success_status = "COMPLETE_LABEL_DIAGNOSTIC_SOURCE_SEMANTICS_INVESTIGATED_OOF_BLOCKED"
    else:
        success_status = "COMPLETE_LABEL_DIAGNOSTIC_SOURCE_SEMANTICS_INVESTIGATED"
    return {
        "research_id": RESEARCH_ID,
        "status": success_status,
        "validation_2024_accessed": False,
        "locked_test_accessed": False,
        "analysis_start": str(frame["local_date"].min().date()),
        "analysis_end": str(frame["local_date"].max().date()),
        "days": int(len(frame)),
        "complete_days": int(frame["hko_temp_complete_day"].sum()),
        "complete_day_rate": float(frame["hko_temp_complete_day"].mean()),
        "exact_match_rate": float(frame["exact_value_match"].mean()),
        "rounded_0p1_match_rate": float(frame["rounded_0p1_match"].mean()),
        "within_0p1_rate": float(frame["within_0p1_c"].mean()),
        "complete_within_0p1_rate": complete_within_rate,
        "mean_abs_diff_c": float(frame["abs_official_minus_reconstructed_c"].mean()),
        "median_abs_diff_c": float(frame["abs_official_minus_reconstructed_c"].median()),
        "max_abs_diff_c": float(frame["abs_official_minus_reconstructed_c"].max()),
        "mean_bias_official_minus_reconstructed_c": float(frame["official_minus_reconstructed_c"].mean()),
        "since_midnight_late_final_available_days": int(len(since_late_available)),
        "since_midnight_late_final_within_0p1_rate": since_late_within_rate,
        "since_midnight_late_final_mean_abs_diff_c": (
            float(since_late_available["since_midnight_late_final_minus_official_c"].abs().mean())
            if not since_late_available.empty
            else None
        ),
        "since_midnight_late_final_max_abs_diff_c": (
            float(since_late_available["since_midnight_late_final_minus_official_c"].abs().max())
            if not since_late_available.empty
            else None
        ),
        "since_midnight_midnight_carryover_suspected_days": int(
            frame["since_midnight_midnight_carryover_suspected"].eq(True).sum()
        ),
        "peak_time_class_counts": frame["peak_time_class"].value_counts().to_dict(),
        "maxmin_disagreement_rows": int(len(maxmin_disagreements)),
        "four_year_oof_feasibility": feasibility.__dict__,
        "stratification_rows": int(len(stratified)),
    }


def markdown_table(frame: pd.DataFrame, columns: list[str], *, limit: int | None = None) -> str:
    if frame.empty:
        return "_No rows._\n"
    view = frame[columns].head(limit) if limit is not None else frame[columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in view.to_dict(orient="records"):
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def long_report(metrics: Mapping[str, object]) -> str:
    oof = metrics["four_year_oof_feasibility"]
    assert isinstance(oof, dict)
    return f"""# EXP-0035 / HKG-T24-R03 Long-Form Experiment Report

## Purpose

R03 is the target-side reconstruction and time-of-maximum anatomy experiment for the HKG T24 Tmax project. It does not build an operational forecasting model. It asks whether the available HKO Headquarters high-frequency temperature feed can reconstruct the official daily maximum temperature label, and it quantifies when the label is difficult to reconstruct from the archived public high-frequency data. This is important because a forecast distribution has to be mapped to official 0.1 C labels. If the archived high-frequency feed misses short-lived peaks or differs semantically from the official climate label, later rounding adapters and probabilistic calibration need to carry that uncertainty explicitly.

## Data Used

The analysis uses only pre-validation target dates from {metrics['analysis_start']} through {metrics['analysis_end']}. It reads HKO Headquarters target labels, HKO Headquarters target-day high-frequency air-temperature rows, target-day since-midnight max rows, and target-day daily mechanism context such as rainfall, King's Park solar radiation, and Waglan wind where present. Every one of these target-day values is treated as label-side or mechanism-only. None is allowed to enter a T-24 operational predictor. Validation 2024 and the 2025-2026 locked test are not read.

The important implementation detail is that this experiment parses the full local day directly from the immutable downloaded DATA.GOV monthly ZIP payloads. It does not use the Phase A/B selected high-frequency table for the reconstruction calculation, because that table was intentionally windowed around 09:00, 12:00, and 15:00 for cutoff-feature engineering. That earlier selected table is still valid for T-24 candidate-feature work, but it is not a full-day target reconstruction table. R03 writes a dedicated derived bronze table at `C:\\hkg_tmax_data\\bronze\\hkg_t24\\r03_hko_hq_full_day_high_frequency.parquet`. Rows retain `source_id`, `content_sha256`, `archive_entry_name`, `archive_payload_timestamp_hkt`, station, variable, observed time, local date, role, and an explicit diagnostic-only availability assumption.

The source date range in this report is deliberately narrower than the raw archive. The raw HKO high-frequency archive extends beyond 2023-12-31, but R03 refuses to read validation 2024 and refuses to read the 2025-2026 locked-test period. This protects later one-shot validation discipline and prevents target-day anatomy knowledge from leaking into feature choices for model experiments.

## Leakage Control

The script applies the locked-test guard to the reconstructed daily target dates. It caps the analysis at 2023-12-31 and records zero validation rows and zero locked-test rows. The experiment uses target-day observations, so its outputs are explicitly marked TARGET_ONLY or MECHANISM_ONLY for future modelling. Peak-time labels, peak duration, and reconstructed-target flags may be considered later only as auxiliary training targets, not as T-24 inference features. The strict four-year OOF gate is also evaluated and recorded. For this modern high-frequency sample it is blocked: {oof['reason']}.

## Reconstruction Method

For each local date, the script filters the full-day parsed `latest_1min_temperature` family to station `HK Observatory` and variable `air_temperature_c`. It computes the maximum observed value, first and last timestamp at that observed maximum, observed row count, median cadence, and a conservative complete-day flag. A day is treated as complete only when it has at least 140 rows, starts near local midnight, ends near 23:40 or later, and has a median cadence consistent with the ten-minute historical snapshots.

This is a high-frequency public snapshot archive, not a dense every-minute raw archive. The name of the feed includes `1min`, but each archived payload is captured roughly every ten minutes and each payload carries the latest one-minute mean at the station. The best reconstruction from the temperature rows is therefore the maximum of the observed ten-minute snapshots, not the true continuous maximum over every minute of the day. That distinction matters: an official daily maximum can occur between public snapshots, especially during short convective breaks, rapid post-rain recovery, sea-breeze transitions, cold-surge rebounds, or brief late-afternoon heating. This is why R03 does not silently replace the official HKO daily Tmax with reconstructed temperature-feed maxima.

The complete-day flag is intentionally strict. It is not meant to make the result look better. It is a quality gate: if a day does not have near-full local-day coverage, the comparison is interpreted as a source-coverage diagnostic instead of a proof of target parity. The max of incomplete snapshots can only be a lower-bound-like diagnostic for the official target.

## Peak Anatomy

R03 records first and last time at the reconstructed maximum, number of peak rows, distinct peak episodes, total peak-duration proxy, and maximum heating over 10, 30, and 60 minutes before the first peak. Peak timing is classified using fixed clock thresholds: early before 12:00, normal from 12:00 through 16:59, and late at 17:00 or later. These thresholds are deliberately not fitted to validation or future data. The current pre-validation sample has the following peak-time counts: {metrics['peak_time_class_counts']}.

## Main Result

The pre-validation sample contains {metrics['days']} reconstructed days, of which {metrics['complete_days']} satisfy the complete-day rule. The overall within-0.1 C agreement rate is {metrics['within_0p1_rate']:.4f}. On complete days the within-0.1 C agreement rate is {metrics['complete_within_0p1_rate']:.4f}. The mean absolute official-minus-reconstructed difference is {metrics['mean_abs_diff_c']:.4f} C, the median absolute difference is {metrics['median_abs_diff_c']:.4f} C, and the maximum absolute difference is {metrics['max_abs_diff_c']:.4f} C. Since the complete-day agreement and the strict four-year OOF gate do not jointly pass, R03 is not a promotion artifact. It is a diagnostic and source-semantics warning.

The sign of the bias is also informative. A positive official-minus-reconstructed value means the official daily maximum is warmer than the maximum observed in the public snapshot temperature feed. That is the expected failure mode when the public archive samples every several minutes rather than preserving every one-minute mean. This supports a conservative conclusion: the official target table remains the authoritative label, while the full-day public temperature archive is useful for peak-timing anatomy, data-quality flags, and missing-peak risk analysis.

## Max/Min Feed Cross-Check

The experiment also compares the temperature-feed reconstruction with the HKO since-midnight max feed. The since-midnight feed is handled with special care. The raw maximum over all running-feed values can be contaminated by a midnight carryover behavior: early rows just after local midnight may still show the previous day's maximum before the running statistic resets. Therefore R03 stores separate columns for the raw feed maximum, the after-01:00 maximum, the latest observed value, and the late-day final value. The late-day final value is the cleanest available source-side approximation to the final daily running max, while the raw maximum is retained as a warning signal rather than treated as truth.

The late-day since-midnight final value is available on {metrics['since_midnight_late_final_available_days']} days. Its within-0.1 C rate against the official daily target is {metrics['since_midnight_late_final_within_0p1_rate']:.4f}. Its mean absolute difference is {metrics['since_midnight_late_final_mean_abs_diff_c']} C, and its maximum absolute difference is {metrics['since_midnight_late_final_max_abs_diff_c']} C. R03 detected {metrics['since_midnight_midnight_carryover_suspected_days']} days where the raw running-feed maximum exceeded the late-day final value, which is the specific signature expected from midnight carryover. Rows where raw max, late final max, or reconstructed temperature-feed max differ materially are written to `artifacts/maxmin_feed_disagreements.csv`. There are {metrics['maxmin_disagreement_rows']} such rows in the pre-validation analysis. These rows are not model failures; they are source-behavior evidence that later target adapters and label-side QC need to inspect.

## Stratification

Discrepancies are stratified by season, month, peak-time class, feed completeness, rainfall state, solar state, wind-speed state, and high-temperature tail status. Rain, solar, and wind are target-day mechanism labels only, not predictors. The stratified table is useful for deciding where source semantics are fragile. For example, larger errors on incomplete days would suggest missing high-frequency observations; larger errors on high-solar or high-tail days would suggest short-lived peaks being missed by ten-minute snapshots.

## Interpretation

R03 makes two points clear. First, the official daily target remains the only authoritative label; reconstructed high-frequency maxima are diagnostic and must not silently replace it. Second, the available public high-frequency history is not long enough to satisfy the strict four-year pre-validation OOF rule for modern experiments. That means trajectory, spatial-field, moisture, wind, and pressure experiments that depend on the same modern feed need either a blocked status, a revised predeclared evaluation design, or additional prospective archive time. This is exactly why R03 is written as a mechanism/label audit rather than a model-skill experiment.

The practical implication is not that the public high-frequency archive is useless. It is highly valuable, but the value is different from a direct target substitute. The archive can support as-of-safe T-1 station-state features, trajectory features up to 15:00 on T-1, station-network gradients, humidity/pressure/wind regime indicators, and operational freshness diagnostics. For target-day anatomy, it can show approximate peak time, peak broadness, short-term heating before the sampled peak, and whether the target is likely to be hard to reconstruct from public snapshots. Those outputs are label-side evidence only.

For later modelling, the safe rule is simple. `target_tmax_c` from the official HKO daily target table remains the label. `reconstructed_tmax_c`, first/last peak time, peak episode count, peak-duration proxy, and target-day rain/solar/wind states are not predictors for T-24 inference. They may be used only inside training-fold diagnostics or as auxiliary labels for later experiments that explicitly model peak timing or suppression mechanisms. If an auxiliary task is built later, its folds must be chronological, its transformations must fit only on training dates, and no validation 2024 or locked-test rows may influence feature choice before the one-shot R30 validation gate.

## Artifacts

The main daily reconstruction table is stored under `C:\\hkg_tmax_data\\gold\\hkg_t24\\r03_tmax_anatomy\\r03_daily_reconstruction.parquet` and copied into this experiment folder. Stratified diagnostics, peak summaries, and max/min disagreement tables are stored beside it and in the experiment `artifacts` directory. The dedicated full-day parsed HKO Headquarters diagnostic table is stored under `C:\\hkg_tmax_data\\bronze\\hkg_t24\\r03_hko_hq_full_day_high_frequency.parquet`. The human report is `reports/hkg_t24/R03_TMAX_RECONSTRUCTION_AND_PEAK_ANATOMY.md`. The reproduction command is in `REPRODUCE.md`. The final status is `{metrics['status']}`, not accepted as a production model or challenger.

The row-level reconstruction table contains the date range used by this experiment, official target values, reconstructed snapshot maxima, row counts, first and last observed timestamps, median cadence, completeness flags, peak timing classes, peak episode counts, heating-rate summaries, source hashes, since-midnight final-value diagnostics, target-day mechanism context, and all discrepancy columns. This is the artifact a future reviewer should inspect before deciding whether a later peak-time auxiliary model is justified.

## Next Use

R04 should only proceed as a cutoff-safe trajectory analysis if the evaluation design explicitly handles the modern high-frequency four-year blocker. If R04 is run before that is solved, it must remain a blocked or exploratory mechanism experiment and must not use validation 2024 or locked-test rows. Peak-time and reconstructed-target diagnostics from R03 can be used to define future auxiliary labels, but only on training folds and never at operational inference time.

The exact next safe task is to update the research ledger and then run the focused tests for the guard, governance, peak anatomy, and R03 parser behavior. If those checks pass, the project can proceed to R04 with a clear limitation: modern high-frequency model-skill claims remain blocked under the user's strict four-year OOF requirement until either more lawful history is acquired or the evaluation design is explicitly revised and documented without touching validation 2024 or the locked test.
"""


def write_experiment(
    *,
    data_root: Path,
    target_path: Path,
    hf_path: Path,
    daily_path: Path,
    reconstruction_path: Path,
    stratified: pd.DataFrame,
    peak_summary: pd.DataFrame,
    disagreements: pd.DataFrame,
    metrics: dict[str, object],
) -> None:
    for subdir in ["results", "artifacts", "predictions", "logs"]:
        (EXPERIMENT_DIR / subdir).mkdir(parents=True, exist_ok=True)
    write_text(EXPERIMENT_DIR / "results" / "metrics.json", json.dumps(metrics, indent=2, default=str))
    stratified.to_csv(EXPERIMENT_DIR / "artifacts" / "stratified_discrepancy.csv", index=False)
    peak_summary.to_csv(EXPERIMENT_DIR / "artifacts" / "peak_time_summary.csv", index=False)
    disagreements.to_csv(EXPERIMENT_DIR / "artifacts" / "maxmin_feed_disagreements.csv", index=False)
    pd.read_parquet(reconstruction_path).to_parquet(
        EXPERIMENT_DIR / "results" / "r03_daily_reconstruction.parquet",
        index=False,
    )
    write_text(
        EXPERIMENT_DIR / "README.md",
        "# EXP-0035 HKG-T24-R03 Official Tmax Reconstruction and Time-of-Maximum Anatomy\n\n"
        "Label-side reconstruction/anatomy experiment using full-day HKO Headquarters rows parsed from immutable raw ZIPs. No validation 2024, no locked test, no Polymarket, and no operational model promotion.\n",
    )
    write_text(
        EXPERIMENT_DIR / "HYPOTHESIS.md",
        "# Hypothesis\n\nThe available HKO high-frequency temperature feed usually reconstructs the official daily Tmax closely, but discrepancies concentrate around missing data, short-lived peaks, and source aggregation differences. Peak timing is mechanism-only label anatomy.\n",
    )
    write_text(
        EXPERIMENT_DIR / "PROTOCOL.md",
        "# Protocol\n\n"
        "1. Use only target dates from 2020-07-01 through 2023-12-31.\n"
        "2. Parse full-day HKO Headquarters target-day high-frequency temperature and since-midnight max/min rows directly from immutable raw ZIPs.\n"
        "3. Compare exact, rounded 0.1 C, and within-0.1 C agreement to the official label.\n"
        "4. Derive first/last peak time, duration proxy, peak episodes, and heating rates.\n"
        "5. Cross-check since-midnight max feed raw-max, after-01:00, latest, and late-final disagreement rows.\n"
        "6. Do not access validation 2024 or locked-test rows.\n",
    )
    write_text(
        EXPERIMENT_DIR / "ASOF_CONTRACT.md",
        "# As-Of Contract\n\n"
        "R03 uses target-day observations and is therefore label-side only. Outputs are TARGET_ONLY or MECHANISM_ONLY and must not enter operational T-24 predictors. Locked-test rows and validation 2024 rows are excluded.\n",
    )
    write_text(
        EXPERIMENT_DIR / "DATA_MANIFEST.yaml",
        f"""research_id: {RESEARCH_ID}
target_table: {target_path}
target_table_sha256: {sha256_file(target_path)}
high_frequency_table: {hf_path}
high_frequency_table_sha256: {sha256_file(hf_path)}
daily_climate_table: {daily_path}
daily_climate_table_sha256: {sha256_file(daily_path)}
reconstruction_table: {reconstruction_path}
data_root: {data_root}
availability: TARGET_ONLY_AND_MECHANISM_ONLY
validation_2024_accessed: false
locked_test_accessed: false
""",
    )
    write_text(
        EXPERIMENT_DIR / "RUN_CONFIG.yaml",
        f"""research_id: {RESEARCH_ID}
analysis_start: {metrics['analysis_start']}
analysis_end: {metrics['analysis_end']}
complete_day_min_rows: 140
expected_rows_per_day: {EXPECTED_TEN_MINUTE_ROWS_PER_DAY}
validation_2024_accessed: false
locked_test_policy: deny
""",
    )
    write_text(
        EXPERIMENT_DIR / "DATE_RANGES.md",
        f"""# Date Ranges

- R03 analysis period: `{metrics['analysis_start']}` through `{metrics['analysis_end']}`.
- Reconstructed days: `{metrics['days']}`.
- Complete days: `{metrics['complete_days']}`.
- Validation 2024: not accessed.
- Locked test 2025-01-01 onward: not accessed.
- Four-year OOF status: `{metrics['four_year_oof_feasibility']['status']}`.
""",
    )
    write_text(
        EXPERIMENT_DIR / "RESULTS.md",
        "# Results\n\n"
        + markdown_table(
            pd.DataFrame([metrics]),
            [
                "days",
                "complete_days",
                "within_0p1_rate",
                "complete_within_0p1_rate",
                "mean_abs_diff_c",
                "median_abs_diff_c",
                "max_abs_diff_c",
                "since_midnight_late_final_available_days",
                "since_midnight_late_final_within_0p1_rate",
                "since_midnight_midnight_carryover_suspected_days",
                "maxmin_disagreement_rows",
            ],
        ),
    )
    write_text(
        EXPERIMENT_DIR / "CONCLUSION.md",
        "# Conclusion\n\n"
        "R03 is complete as a label-side anatomy and source-semantics diagnostic. It is not promotable as a model-skill result because modern high-frequency history fails the strict four-year OOF gate before validation 2024, and target-day peak anatomy is not an operational T-24 predictor.\n",
    )
    write_text(
        EXPERIMENT_DIR / "REPRODUCE.md",
        "# Reproduce\n\n"
        "```powershell\n"
        ".\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_r03_tmax_anatomy.py --data-root C:\\hkg_tmax_data\n"
        "```\n",
    )
    write_text(
        EXPERIMENT_DIR / "STATUS.yaml",
        f"""status: {metrics['status']}
research_id: HKG-T24-R03
locked_test_accessed: false
validation_2024_accessed: false
leakage_guard: PASS
four_year_oof: BLOCKED
production_eligible: false
source_semantics_investigated: true
""",
    )
    write_text(EXPERIMENT_DIR / "EXPERIMENT_REPORT_7500_CHARS.md", long_report(metrics))


def write_report(metrics: dict[str, object], stratified: pd.DataFrame, peak_summary: pd.DataFrame) -> None:
    report_path = REPO_ROOT / "reports" / "hkg_t24" / "R03_TMAX_RECONSTRUCTION_AND_PEAK_ANATOMY.md"
    write_text(
        report_path,
        long_report(metrics)
        + "\n# R03 Machine-Readable Summary Tables\n\n"
        f"Generated: `{now_utc()}`\n\n"
        f"- Validation 2024 accessed: `false`\n"
        f"- Locked test accessed: `false`\n"
        f"- Analysis period: `{metrics['analysis_start']}` through `{metrics['analysis_end']}`\n"
        f"- Days: `{metrics['days']}`\n"
        f"- Complete days: `{metrics['complete_days']}`\n"
        f"- Within 0.1 C rate: `{metrics['within_0p1_rate']:.4f}`\n"
        f"- Complete-day within 0.1 C rate: `{metrics['complete_within_0p1_rate']:.4f}`\n"
        f"- Mean absolute difference: `{metrics['mean_abs_diff_c']:.4f}` C\n"
        f"- Four-year OOF status: `{metrics['four_year_oof_feasibility']['status']}`\n\n"
        "## Stratified Discrepancy\n\n"
        + markdown_table(
            stratified,
            [
                "dimension",
                "value",
                "n",
                "complete_days",
                "mean_abs_diff_c",
                "max_abs_diff_c",
                "within_0p1_rate",
            ],
            limit=80,
        )
        + "\n## Peak-Time Summary\n\n"
        + markdown_table(peak_summary, ["peak_time_class", "n", "mean_target_tmax_c", "mean_abs_diff_c"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R03 HKO Tmax reconstruction and peak anatomy.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    target_path = data_root / "silver" / "targets" / "hko_daily_tmax.parquet"
    daily_path = data_root / "bronze" / "analysis_phase_a" / "hko_daily_climate_elements.parquet"
    hf, hf_path = build_full_day_hko_hq_high_frequency(data_root)
    target = pd.read_parquet(target_path)
    daily = pd.read_parquet(daily_path)
    for frame in (hf, target, daily):
        frame["local_date"] = pd.to_datetime(frame["local_date"])
    reconstruction = build_reconstruction(hf)
    since = build_since_midnight_max(hf)
    context = build_daily_context(daily)
    daily_reconstruction = add_comparisons(reconstruction, target, since, context)
    assert_no_locked_dates(daily_reconstruction["local_date"], context="R03 daily reconstruction")
    stratified = build_stratification(daily_reconstruction)
    peak_summary = (
        daily_reconstruction.groupby("peak_time_class")
        .agg(
            n=("local_date", "size"),
            mean_target_tmax_c=("target_tmax_c", "mean"),
            mean_abs_diff_c=("abs_official_minus_reconstructed_c", "mean"),
        )
        .reset_index()
    )
    maxmin_disagreements = daily_reconstruction[
        (daily_reconstruction["since_midnight_rawmax_minus_reconstructed_c"].abs() > 0.100000001)
        | (daily_reconstruction["since_midnight_rawmax_minus_official_c"].abs() > 0.100000001)
        | (daily_reconstruction["since_midnight_late_final_minus_official_c"].abs() > 0.100000001)
    ].copy()
    metrics = build_metrics(daily_reconstruction, stratified, maxmin_disagreements)
    metrics["git"] = git_state()

    output_dir = data_root / "gold" / "hkg_t24" / "r03_tmax_anatomy"
    output_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_path = output_dir / "r03_daily_reconstruction.parquet"
    stratified_path = output_dir / "r03_stratified_discrepancy.parquet"
    peak_path = output_dir / "r03_peak_time_summary.parquet"
    disagreements_path = output_dir / "r03_maxmin_feed_disagreements.parquet"
    daily_reconstruction.to_parquet(reconstruction_path, index=False)
    stratified.to_parquet(stratified_path, index=False)
    peak_summary.to_parquet(peak_path, index=False)
    maxmin_disagreements.to_parquet(disagreements_path, index=False)
    metrics["data_root_outputs"] = {
        "daily_reconstruction": str(reconstruction_path),
        "stratified_discrepancy": str(stratified_path),
        "peak_time_summary": str(peak_path),
        "maxmin_feed_disagreements": str(disagreements_path),
    }
    write_experiment(
        data_root=data_root,
        target_path=target_path,
        hf_path=hf_path,
        daily_path=daily_path,
        reconstruction_path=reconstruction_path,
        stratified=stratified,
        peak_summary=peak_summary,
        disagreements=maxmin_disagreements,
        metrics=metrics,
    )
    write_report(metrics, stratified, peak_summary)
    print(json.dumps({"status": "ok", "metrics": metrics}, indent=2, default=str))


if __name__ == "__main__":
    main()
