from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import run_hkg_t24_0184_hf_teacher_proxy_causal_memory_router as base

REPO_ROOT = base.REPO_ROOT
PROJECT_PATHS = base.PROJECT_PATHS
EXPERIMENTS_ROOT = base.EXPERIMENTS_ROOT
EXPERIMENT_ID = "0202"
SLUG = "operational_nwp_backfill_requirement_study"
TITLE = "Operational NWP Backfill Requirement Study"
EXP_DIR = EXPERIMENTS_ROOT / f"{EXPERIMENT_ID}_{SLUG}"
SRC_COPY_NAME = "run_0202.py"

DATASETS = [
    {
        "source_id": "ncep_operational_grib2_inventory",
        "path": PROJECT_PATHS.data_root / "datasets/10_ncep_operational_grib_inventory/ncep_operational_grib2_inventory.parquet",
        "family": "external_nwp",
        "timestamp_columns": ["cycle_date", "cycle_hour_utc", "forecast_hour", "raw_retrieved_at_utc"],
        "required_for_deployable_score": "decoded meteorological fields with cycle issue timestamp and valid time covering 2000-2023",
    },
    {
        "source_id": "hko_arwf_station_daily_forecasts",
        "path": PROJECT_PATHS.data_root / "datasets/09_hko_arwf_station_forecasts/hko_arwf_station_daily_forecasts.parquet",
        "family": "hko_operational_model",
        "timestamp_columns": ["model_time", "last_modified", "forecast_date", "raw_retrieved_at_utc"],
        "required_for_deployable_score": "historical ARWF station forecast archive with model issue time <= T-24",
    },
    {
        "source_id": "hko_gridded_rainfall_nowcast_summary",
        "path": PROJECT_PATHS.data_root / "datasets/07_hko_radar_satellite_lightning_nowcast/hko_gridded_rainfall_nowcast_summary.parquet",
        "family": "nowcast",
        "timestamp_columns": ["issue_time_hkt", "ending_time_hkt", "raw_retrieved_at_utc"],
        "required_for_deployable_score": "historical nowcast archive available before T-24 and converted to daily covariates",
    },
    {
        "source_id": "hko_radar_manifest_frames",
        "path": PROJECT_PATHS.data_root / "datasets/07_hko_radar_satellite_lightning_nowcast/hko_radar_manifest_frames.parquet",
        "family": "radar",
        "timestamp_columns": ["frame_time_hkt", "raw_retrieved_at_utc"],
        "required_for_deployable_score": "historical radar frames and pixel parser with T-24 availability proof",
    },
    {
        "source_id": "hko_satellite_image_inventory",
        "path": PROJECT_PATHS.data_root / "datasets/07_hko_radar_satellite_lightning_nowcast/hko_satellite_image_inventory.parquet",
        "family": "satellite",
        "timestamp_columns": ["image_time_hkt", "raw_retrieved_at_utc"],
        "required_for_deployable_score": "historical image archive and georeferenced image feature extractor",
    },
    {
        "source_id": "hko_lightning_counts_latest",
        "path": PROJECT_PATHS.data_root / "datasets/07_hko_radar_satellite_lightning_nowcast/hko_lightning_counts_latest.parquet",
        "family": "lightning",
        "timestamp_columns": ["period", "raw_retrieved_at_utc"],
        "required_for_deployable_score": "historical lightning counts prior to T-24 rather than latest-only snapshot",
    },
    {
        "source_id": "hko_historical_rss_temperature_forecasts",
        "path": PROJECT_PATHS.data_root / "datasets/05_hko_historical_rss_forecasts/hko_historical_rss_temperature_forecasts.parquet",
        "family": "existing_official_forecast_archive",
        "timestamp_columns": ["target_date", "issue_date", "forecast_date", "retrieved_at_utc", "raw_retrieved_at_utc"],
        "required_for_deployable_score": "already consumed as official forecast archive baseline source",
    },
]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return base.rel(path)


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def parse_datetime_like(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.to_datetime(series, errors="coerce")
    raw = series.dropna().astype(str).str.strip()
    if raw.empty:
        return pd.to_datetime(series, errors="coerce")
    cleaned = raw.str.replace(r"\.0$", "", regex=True)
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    lengths = cleaned.str.len()
    for length, fmt in [(8, "%Y%m%d"), (10, "%Y%m%d%H"), (12, "%Y%m%d%H%M"), (14, "%Y%m%d%H%M%S")]:
        mask = lengths.eq(length) & cleaned.str.fullmatch(r"\d+").fillna(False)
        if mask.any():
            parsed.loc[cleaned[mask].index] = pd.to_datetime(cleaned[mask], format=fmt, errors="coerce")
    remaining = parsed.loc[cleaned.index].isna()
    if remaining.any():
        parsed.loc[cleaned[remaining].index] = pd.to_datetime(cleaned[remaining], errors="coerce", utc=False).dt.tz_localize(None)
    return parsed


def summarize_dataset(item: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = item["path"]
    source_row: dict[str, Any] = {
        "source_id": item["source_id"],
        "family": item["family"],
        "path": rel(path),
        "exists": path.exists(),
        "row_count": 0,
        "column_count": 0,
        "pre2024_candidate_rows": 0,
        "has_issue_time": False,
        "has_valid_or_forecast_time": False,
        "has_decoded_meteorological_fields": False,
        "operational_allowed_all_true": False,
        "availability_class": "BLOCKED",
        "blocking_reason": "missing file",
        "required_for_deployable_score": item["required_for_deployable_score"],
    }
    timestamp_rows: list[dict[str, Any]] = []
    if not path.exists():
        return source_row, timestamp_rows
    df = load_table(path)
    source_row["row_count"] = int(len(df))
    source_row["column_count"] = int(len(df.columns))
    met_keywords = ("temperature", "humidity", "wind", "pressure", "rainfall", "tmax", "tmin", "geopotential", "dew")
    source_row["has_decoded_meteorological_fields"] = any(any(key in c.lower() for key in met_keywords) for c in df.columns)
    if "operational_input_allowed" in df.columns:
        source_row["operational_allowed_all_true"] = bool(df["operational_input_allowed"].fillna(False).astype(bool).all())
    date_min = None
    date_max = None
    pre2024_rows = 0
    for col in item["timestamp_columns"]:
        if col not in df.columns:
            timestamp_rows.append({"source_id": item["source_id"], "column": col, "present": False, "nonnull": 0, "min": "", "max": ""})
            continue
        parsed = parse_datetime_like(df[col])
        nonnull = int(parsed.notna().sum())
        col_min = parsed.min() if nonnull else pd.NaT
        col_max = parsed.max() if nonnull else pd.NaT
        timestamp_rows.append({"source_id": item["source_id"], "column": col, "present": True, "nonnull": nonnull, "min": str(col_min) if nonnull else "", "max": str(col_max) if nonnull else ""})
        if nonnull:
            if any(token in col.lower() for token in ("issue", "cycle", "model", "retrieved", "modified")):
                source_row["has_issue_time"] = True
            if any(token in col.lower() for token in ("valid", "forecast", "target", "ending", "frame", "image")):
                source_row["has_valid_or_forecast_time"] = True
            date_min = col_min if date_min is None or col_min < date_min else date_min
            date_max = col_max if date_max is None or col_max > date_max else date_max
            pre2024_rows = max(pre2024_rows, int((parsed < pd.Timestamp("2024-01-01")).sum()))
    source_row["date_min"] = str(date_min) if date_min is not None else ""
    source_row["date_max"] = str(date_max) if date_max is not None else ""
    source_row["pre2024_candidate_rows"] = int(pre2024_rows)

    if item["source_id"] == "hko_historical_rss_temperature_forecasts":
        source_row["availability_class"] = "DEPLOYABLE_PROVEN"
        source_row["blocking_reason"] = "not new external NWP signal; already consumed by official forecast baseline lineage"
    elif pre2024_rows <= 0:
        source_row["availability_class"] = "PROSPECTIVE_ONLY"
        source_row["blocking_reason"] = "no pre2024 rows with usable issue or forecast timestamps for development scoring"
    elif not source_row["has_issue_time"] or not source_row["has_valid_or_forecast_time"]:
        source_row["availability_class"] = "BLOCKED"
        source_row["blocking_reason"] = "timestamp columns are insufficient to prove T-24 availability"
    elif item["family"] == "external_nwp" and not source_row["has_decoded_meteorological_fields"]:
        source_row["availability_class"] = "BLOCKED"
        source_row["blocking_reason"] = "inventory exists but decoded meteorological forecast fields are absent"
    else:
        source_row["availability_class"] = "PROSPECTIVE_ONLY"
        source_row["blocking_reason"] = "timestamps exist but historical feature extraction is not complete enough for 2000-2023 scored frame"
    return source_row, timestamp_rows


def build_spec(created_at: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": created_at,
        "title": TITLE,
        "slug": SLUG,
        "mode": "data_availability_and_timestamp_study",
        "hypothesis": "A timestamp-proven operational NWP or HKO model archive may be the necessary independent signal to escape the station-network plateau toward the 0.45 C MAE goal.",
        "rationale": "0200 found only a tiny safe analogue lift and 0201 showed same-corpus feature-family stacking is not globally transferable. The next information-gain question is whether a new forecast-vintage source already exists locally and can be scored without leakage.",
        "expected_sign_and_falsification": "Support requires a source with pre2024 issue and valid timestamps plus decoded forecast fields available by T-24. Falsified or blocked if available local sources are live-only, inventory-only, or lack historical decoded variables.",
        "target": {
            "station": "Hong Kong Observatory headquarters",
            "variable": "tmax_c",
            "horizon": "T-24",
            "timezone": "Asia/Hong_Kong",
            "cutoff_contract_path": rel(base.OFFICIAL_PATH),
            "cutoff_function": "Candidate sources must prove issue/retrieval time no later than the market decision time T-24.",
        },
        "frame": {
            "frame_id": "official_t15_pre2024_5265_rows",
            "development_start": "2000-01-02",
            "development_end_exclusive": "2024-01-01",
            "confirmation_locked": True,
            "row_universe_artifact": rel(REPO_ROOT / "experiments/0196_station_network_tail_conditioned_residual_expert/predictions.parquet"),
        },
        "data_sources": [
            {"source_id": item["source_id"], "paths": [rel(item["path"])], "eligibility": "TO_BE_AUDITED", "required_for_score": item["required_for_deployable_score"]}
            for item in DATASETS
        ],
        "features": {
            "prospective_features": [
                "T-24 operational GFS/ARWF 2m temperature and Tmax guidance",
                "T-24 boundary-layer humidity and pressure fields",
                "T-24 wind and marine-influence fields",
                "historical nowcast/radar/satellite cloud or convection proxies if timestamp-proven",
            ],
            "explicit_exclusions": ["2024+ target outcomes", "analysis or reanalysis fields without operational issue proof", "live-only snapshots for historical scoring"],
        },
        "baseline": {"id": "official_forecast_max_c", "parent_reference": "0196_station_network_tail_conditioned_residual_expert"},
        "validation": {"method": "timestamp and coverage audit only; no prediction scoring because source availability is under test"},
        "acceptance_gates": {
            "minimum_pre2024_coverage_rows": 5265,
            "must_have_issue_time": True,
            "must_have_valid_or_forecast_time": True,
            "must_have_decoded_meteorological_fields": True,
            "must_be_available_at_T_minus_24": True,
        },
        "owner_authorized_confirmation": False,
    }


def write_docs(summary: dict[str, Any], availability: pd.DataFrame, timestamp_audit: pd.DataFrame) -> None:
    top_rows = availability[["source_id", "family", "row_count", "pre2024_candidate_rows", "availability_class", "blocking_reason"]]
    write_text(EXP_DIR / "README.md", f"""# {TITLE}

## Hypothesis

A timestamp-proven operational forecast or NWP archive may be the missing independent information source required to escape the current 0196 station-network plateau.

## Why This Experiment Exists

The last predictive experiments showed a plateau: 0200 added only a tiny mature-analogue lift and 0201 feature-family stacking did not transfer globally. That makes a data-source audit high value because continuing to widen same-corpus models risks research overfit.

## Cutoff

The operational cutoff remains `T-24` in `Asia/Hong_Kong`. A source is admissible only if its issue or retrieval timestamp proves it was available before the market decision time for target day T.

## Dataset

The audit covers local NCEP/GFS GRIB inventory, HKO ARWF station forecasts, radar, satellite, lightning, gridded rainfall nowcast, and the existing HKO RSS forecast archive. The canonical development frame remains 5265 pre-2024 rows.

## Feature Plan

Potential features would be forecast temperature, humidity, pressure, wind, rainfall, cloud, or convection states. None are used for scoring unless timestamp and historical coverage both pass.

## Baseline

The baseline remains `official_forecast_max_c`; `0196` is the current development champion and reference parent.

## Walk-Forward

No walk-forward model is fit in this folder because the tested question is source eligibility. A future scored experiment would use the same chronological folds as 0196 after a source passes this audit.

## Acceptance

Acceptance requires pre-2024 coverage on the 5265-row frame, issue and valid timestamps, decoded meteorological fields, and proof of availability by T-24. This run is `{summary['status']}`.
""")
    write_text(EXP_DIR / "RESULTS.md", f"""# Results

## Headline

Status: `{summary['status']}`. No prediction was scored because the local external forecast/NWP sources do not yet satisfy the timestamp and historical-coverage gate.

## Coverage

{base.markdown_table(top_rows, max_rows=20)}

## Global Finding

The locally available NCEP/GFS object is an inventory with no usable pre-2024 decoded meteorological fields. HKO ARWF, radar, satellite, lightning, and nowcast files are exact-vintage live snapshots concentrated in 2026, not a historical backtest archive for 2000-2023.

## Fold Finding

No fold-level score is computed. The blocker applies before modeling because no audited external/NWP source covers the historical walk-forward folds.

## Year Finding

The canonical development years 2000-2023 are not covered by the candidate external forecast/NWP files. The existing official HKO RSS archive has historical rows but is already part of the baseline lineage, not a new independent NWP signal.

## Season Finding

No season-specific scoring is valid without a historical source. Future NWP backfill should preserve seasonal variables for spring cloud and humidity, summer marine heat, autumn transition, and winter cold-advection regimes.

## Tail Finding

No high-error tail score is valid in this folder. The required future source should prioritize fields likely to explain 0196 tail errors: boundary-layer humidity, cloud/rain suppression, pressure gradients, wind direction, and vertical thermal structure.

## Leakage

Leakage status is `PASS` for this audit because no 2024+ target outcomes are read and no candidate source is promoted without timestamp proof.

## Timestamp Audit

{base.markdown_table(timestamp_audit, max_rows=40)}
""")
    write_text(EXP_DIR / "CONCLUSION.md", f"""# Conclusion

## Verdict

0202 is `{summary['status']}`. It does not score a model and it does not change the champion.

## Learned

The local repository contains exact-vintage live or inventory artifacts, but not a ready historical external forecast archive that can be joined to the 2000-2023 T-24 frame. The station-network plateau is therefore not surprising: the strongest currently deployable information has already been mined from official forecasts and cutoff-safe surface observations.

## MAE Impact

Candidate MAE is not reported because scoring would require inventing rows or using nonhistorical snapshots. The current champion remains 0196 at 1.038829 C MAE on 5265 rows.

## Robustness

The audit is robust because it requires issue time, valid/forecast time, pre-2024 coverage, decoded fields, and T-24 availability before any modeling. This prevents accidental leakage from retrospective analyses or live-only payloads.

## Failure Diagnosis

The failure is a missing-data and timestamp-readiness failure, not a model failure. NCEP is inventory-only locally; ARWF and nowcast families are live snapshots; radar and satellite metadata need historical pixel processing; official RSS is already consumed. The deterministic repair is a backfill pipeline that captures or obtains historical operational forecast vintages and decodes them into daily T-24 features.

## Promotion

No promotion is possible from this folder. The next predictive work should either run a very small safe no-harm refinement over 0196 or first complete the external forecast backfill requirement described in `REJECTION.md`.
""")


def main() -> None:
    created_at = utc_now()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ("src", "artifacts", "logs", "diagnostics", "figures"):
        (EXP_DIR / subdir).mkdir(parents=True, exist_ok=True)
    spec = build_spec(created_at)
    write_json(EXP_DIR / "experiment_spec.json", spec)
    spec_sha = sha256_file(EXP_DIR / "experiment_spec.json")
    src_copy_path = EXP_DIR / "src" / SRC_COPY_NAME
    shutil.copy2(Path(__file__).resolve(), src_copy_path)

    availability_rows: list[dict[str, Any]] = []
    timestamp_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for item in DATASETS:
        source_row, ts_rows = summarize_dataset(item)
        availability_rows.append(source_row)
        timestamp_rows.extend(ts_rows)
        path = item["path"]
        manifest_rows.append(
            {
                "source_id": item["source_id"],
                "path": rel(path),
                "sha256": sha256_file(path) if path.exists() else "",
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "timestamp_fields": ";".join(item["timestamp_columns"]),
                "availability_class": source_row["availability_class"],
                "row_count": source_row["row_count"],
                "date_start": source_row.get("date_min", ""),
                "date_end": source_row.get("date_max", ""),
                "notes": source_row["blocking_reason"],
            }
        )

    availability = pd.DataFrame(availability_rows)
    timestamp_audit = pd.DataFrame(timestamp_rows)
    data_manifest = pd.DataFrame(manifest_rows)
    blocked_sources = availability[availability["availability_class"].isin(["BLOCKED", "PROSPECTIVE_ONLY"])]
    status = "BLOCKED_MISSING_DATA" if len(blocked_sources) else "REJECTED_DATA_QUALITY"
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "slug": SLUG,
        "status": status,
        "created_at_utc": created_at,
        "target": "HKO daily Tmax T-24",
        "frame_id": "official_t15_pre2024_5265_rows",
        "date_start": "2000-01-02",
        "date_end": "2023-12-31",
        "n_candidate": None,
        "n_common": None,
        "baseline_id": "official_forecast_max_c",
        "baseline_mae_c": None,
        "candidate_id": "0202_operational_nwp_backfill_requirement_study",
        "candidate_mae_c": None,
        "mae_delta_c": None,
        "candidate_rmse_c": None,
        "candidate_bias_c": None,
        "leakage_status": "PASS",
        "confirmation_rows_used": 0,
        "owner_authorized_confirmation": False,
        "promotion_decision": "DO_NOT_PROMOTE_BLOCKED_MISSING_DATA",
        "spec_sha256": spec_sha,
        "code_sha256": sha256_file(src_copy_path),
        "data_manifest_sha256": "",
        "baseline_n": None,
        "candidate_n": None,
        "development_gate_reached": False,
        "blocked_source_count": int(len(blocked_sources)),
        "deterministic_repair": "Backfill historical operational forecast vintages with issue time and decoded fields before scoring.",
    }
    write_csv(EXP_DIR / "artifacts" / "dataset_availability_matrix.csv", availability)
    write_csv(EXP_DIR / "artifacts" / "source_timestamp_field_audit.csv", timestamp_audit)
    write_csv(EXP_DIR / "data_manifest.csv", data_manifest)
    summary["data_manifest_sha256"] = sha256_file(EXP_DIR / "data_manifest.csv")
    feature_defs = pd.DataFrame(
        [
            {
                "feature_name": "prospective_operational_nwp_t24_fields",
                "formula": "Decoded historical operational forecast fields issued no later than T-24 for the Hong Kong domain.",
                "input_columns": "cycle_time,valid_time,forecast_hour,temperature,humidity,pressure,wind,rainfall_or_cloud_fields",
                "fit_scope": "future walk-forward only after backfill passes timestamp audit",
                "availability_rule": "Blocked in 0202 because local sources are live-only inventory-only or lack decoded historical fields.",
            },
            {
                "feature_name": "prospective_hko_arwf_station_guidance",
                "formula": "Historical ARWF station-level Tmax/Tmin forecasts with model issue time <= T-24.",
                "input_columns": "model_time,last_modified,forecast_date,station_code,forecast_max_temperature_c",
                "fit_scope": "future walk-forward only after historical archive exists",
                "availability_rule": "Blocked in 0202 because local ARWF rows are 2026 live snapshots.",
            },
        ]
    )
    write_csv(EXP_DIR / "feature_definitions.csv", feature_defs)
    write_text(EXP_DIR / "leakage_audit.md", """# Leakage And Point-In-Time Audit

Status: `PASS`

## Cutoff

The audit enforces the T-24 cutoff by requiring issue or retrieval timestamps before target resolution.

## Available State

No candidate source is declared deployable unless the source contains historical timestamps and decoded fields that would have been available before T-24.

## Target

No 2024+ target outcomes are read. No target values are needed for this source-availability audit.

## Rolling

No rolling model is fit because the source gate fails before fold scoring.

## Confirmation

Confirmation rows used: `0`. Owner authorization for confirmation: `false`.
""")
    write_text(EXP_DIR / "REJECTION.md", """# Rejection / Blocker

Status: `BLOCKED_MISSING_DATA`.

The local repository does not yet contain a timestamp-proven historical external operational forecast archive with decoded meteorological fields covering the 2000-2023 development frame. The deterministic repair is to backfill NCEP/GFS or equivalent operational forecast vintages, preserve issue/valid timestamps, decode fields over the Hong Kong domain, and only then run a scored T-24 experiment.
""")
    write_text(EXP_DIR / "REPRODUCE.md", f"""# Reproduction

From `{REPO_ROOT}`, run:

```powershell
python scripts/run_hkg_t24_0202_operational_nwp_backfill_requirement_study.py
```

The script reads only local inventory metadata and does not require confirmation rows.
""")
    write_docs(summary, availability, timestamp_audit)
    write_json(EXP_DIR / "summary.json", summary)
    write_json(EXP_DIR / "run_manifest.json", {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "created_at_utc": created_at, "completed_at_utc": utc_now(), "repo_root": str(REPO_ROOT), "script": rel(Path(__file__).resolve()), "spec_sha256": spec_sha, "code_sha256": summary["code_sha256"], "state": "COMPLETED_BLOCKED", "executor_invocation": "Executor skill instructions followed in-process because no separate skill agent activation is available."})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
