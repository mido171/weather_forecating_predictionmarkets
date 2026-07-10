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
EXPERIMENTS_ROOT = base.EXPERIMENTS_ROOT
EXPERIMENT_ID = "0204"
SLUG = "external_forecast_vintage_acquisition_plan"
TITLE = "External Forecast Vintage Acquisition Plan"
EXP_DIR = EXPERIMENTS_ROOT / f"{EXPERIMENT_ID}_{SLUG}"
SRC_COPY_NAME = "run_0204.py"
P0202 = EXPERIMENTS_ROOT / "0202_operational_nwp_backfill_requirement_study"


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


def build_plan_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    acquisition_plan = pd.DataFrame(
        [
            {
                "rank": 1,
                "source_family": "operational_global_nwp",
                "candidate_source": "historical GFS or equivalent operational forecast vintages",
                "minimum_period": "2000-01-02 through 2023-12-31",
                "domain": "Hong Kong box plus upstream South China coast and Pearl River Delta",
                "cycle_requirement": "all cycles whose issue time is available before the canonical T-24 decision cutoff",
                "raw_artifact_requirement": "immutable raw file path content sha256 retrieval timestamp cycle time forecast hour valid time",
                "decoder_requirement": "extract point box and gradient features from raw operational fields without reanalysis substitution",
                "why_priority": "largest plausible independent signal beyond official forecast and station-network memory",
            },
            {
                "rank": 2,
                "source_family": "hko_operational_model",
                "candidate_source": "historical ARWF station forecasts",
                "minimum_period": "2000-01-02 through 2023-12-31 or earliest documented availability",
                "domain": "HKO station forecast grid and nearby station points",
                "cycle_requirement": "model_time and last_modified must be before T-24 for each target",
                "raw_artifact_requirement": "payload sha256 retrieval timestamp station code forecast date model time",
                "decoder_requirement": "station forecast max min and station spread features by target date",
                "why_priority": "direct local operational guidance likely captures HKO forecaster or model information unavailable to ISD",
            },
            {
                "rank": 3,
                "source_family": "cloud_convection_nowcast_archive",
                "candidate_source": "historical radar satellite lightning rainfall-nowcast summaries",
                "minimum_period": "2000-01-02 through 2023-12-31 if available else 2020-2023 diagnostic frame only",
                "domain": "Hong Kong rainfall radar and Himawari cloud fields around the territory",
                "cycle_requirement": "image or issue timestamp must be before T-24",
                "raw_artifact_requirement": "raw image or grid sha256 time stamp parser version georeference metadata",
                "decoder_requirement": "daily pre-cutoff cloud rain convection and marine-cloud proxies",
                "why_priority": "candidate mechanism for warm-season Tmax suppression and high-tail misses",
            },
        ]
    )
    variable_priority = pd.DataFrame(
        [
            {"rank": 1, "variable_group": "near_surface_temperature", "fields": "2m temperature Tmax or hourly temperature if available", "feature_examples": "point mean max upstream gradient anomaly versus climatology", "mechanism": "direct thermal guidance and heat-advection state"},
            {"rank": 2, "variable_group": "humidity_and_boundary_layer", "fields": "2m dewpoint relative humidity precipitable water 850 humidity", "feature_examples": "dewpoint spread moisture advection humid heat index proxy", "mechanism": "cloud and latent cooling limits plus oppressive warm nights"},
            {"rank": 3, "variable_group": "wind_and_marine_influence", "fields": "10m u v wind 850 wind sea breeze direction", "feature_examples": "onshore component Pearl River Delta upstream component wind shift", "mechanism": "marine cooling and hot continental advection"},
            {"rank": 4, "variable_group": "pressure_and_synoptic_gradient", "fields": "MSLP surface pressure geopotential at 850 700 500", "feature_examples": "subtropical ridge strength pressure tendency thermal thickness", "mechanism": "subsidence ridge strength and synoptic regime"},
            {"rank": 5, "variable_group": "cloud_rain_radiation", "fields": "total cloud cover precipitation shortwave radiation radar rain satellite cloud", "feature_examples": "pre-cutoff cloud shield rain probability solar suppression", "mechanism": "daytime insolation suppression and convective cooling"},
        ]
    )
    timestamp_contract = pd.DataFrame(
        [
            {"field": "source_id", "required": True, "rule": "stable source family identifier"},
            {"field": "raw_content_sha256", "required": True, "rule": "hash of immutable raw payload before decoding"},
            {"field": "retrieved_at_utc", "required": True, "rule": "when this copy was obtained by the pipeline"},
            {"field": "issue_time_utc", "required": True, "rule": "operational cycle or model issue time"},
            {"field": "valid_time_utc", "required": True, "rule": "meteorological valid time for each field"},
            {"field": "forecast_hour", "required": True, "rule": "lead time from issue_time to valid_time"},
            {"field": "target_local_date", "required": True, "rule": "HKO local target date T"},
            {"field": "cutoff_utc", "required": True, "rule": "canonical T-24 decision cutoff converted to UTC"},
            {"field": "available_at_utc", "required": True, "rule": "max issue or retrieval timestamp proving feature exists before cutoff"},
            {"field": "decoder_version", "required": True, "rule": "versioned deterministic field extraction code"},
            {"field": "availability_assertion", "required": True, "rule": "available_at_utc <= cutoff_utc or row is ineligible"},
        ]
    )
    scored_experiment_gate = pd.DataFrame(
        [
            {"gate": "coverage", "requirement": "at least 5265 canonical rows for comparable champion scoring or explicitly declared narrower diagnostic frame"},
            {"gate": "timestamp", "requirement": "all feature rows prove available_at_utc <= cutoff_utc"},
            {"gate": "raw_lineage", "requirement": "every decoded row links to immutable raw sha256 and decoder version"},
            {"gate": "no_reanalysis", "requirement": "retrospective analyses are rejected unless they are only diagnostic and not promoted"},
            {"gate": "walk_forward", "requirement": "same outer folds as 0196 with no 2024+ target access"},
            {"gate": "baseline", "requirement": "official forecast and 0196 must be reproduced on identical rows before candidate scoring"},
        ]
    )
    return acquisition_plan, variable_priority, timestamp_contract, scored_experiment_gate


def build_spec(created_at: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": created_at,
        "title": TITLE,
        "slug": SLUG,
        "mode": "data_acquisition_design",
        "hypothesis": "The fastest non-overfit path toward materially lower HKG T-24 Tmax MAE is a timestamp-proven historical external forecast vintage archive with decoded meteorological fields.",
        "rationale": "0202 proved the local repository lacks a scoreable historical external forecast archive. 0204 translates that blocker into the exact acquisition and timestamp contract for the next scored lane.",
        "expected_sign_and_falsification": "Support is an actionable source contract that would permit a future scored experiment. It remains blocked until the raw vintage archive exists locally.",
        "target": {"station": "Hong Kong Observatory headquarters", "variable": "tmax_c", "horizon": "T-24", "timezone": "Asia/Hong_Kong", "cutoff_contract_path": rel(base.OFFICIAL_PATH), "cutoff_function": "Every acquired field must prove availability before the T-24 market decision cutoff."},
        "frame": {"frame_id": "official_t15_pre2024_5265_rows", "development_start": "2000-01-02", "development_end_exclusive": "2024-01-01", "confirmation_locked": True, "row_universe_artifact": rel(REPO_ROOT / "experiments/0196_station_network_tail_conditioned_residual_expert/predictions.parquet")},
        "data_sources": [{"source_id": "0202_source_availability_audit", "paths": [rel(P0202 / "artifacts/dataset_availability_matrix.csv")], "eligibility": "BLOCKED"}],
        "features": {"planned_feature_families": ["near_surface_temperature", "humidity_and_boundary_layer", "wind_and_marine_influence", "pressure_and_synoptic_gradient", "cloud_rain_radiation"], "explicit_exclusions": ["2024+ target outcomes", "retrospective reanalysis promotion", "live-only snapshots for historical scoring"]},
        "baseline": {"id": "official_forecast_max_c", "parent_reference": "0196_station_network_tail_conditioned_residual_expert"},
        "validation": {"method": "acquisition contract only; no scored model until backfill exists"},
        "acceptance_gates": {"historical_raw_vintages": True, "issue_and_valid_times": True, "decoded_fields": True, "t24_availability_proof": True, "identical_row_baseline": True},
        "owner_authorized_confirmation": False,
    }


def write_docs(summary: dict[str, Any], acquisition_plan: pd.DataFrame, variable_priority: pd.DataFrame, timestamp_contract: pd.DataFrame) -> None:
    write_text(EXP_DIR / "README.md", f"""# {TITLE}

## Hypothesis

The most credible non-overfit path beyond the 0196 plateau is a true historical operational forecast vintage archive.

## Why This Experiment Exists

0202 proved the local NWP and nowcast files are not scoreable for 2000-2023. 0204 converts that blocker into a precise acquisition contract so the next scored experiment does not start from vague source names.

## Cutoff

The cutoff remains T-24 in Asia/Hong_Kong. Every acquired variable must have issue time and availability proof before the target day's market decision time.

## Dataset

The required target frame is the 5265-row pre-2024 canonical official frame. This folder does not add new data; it defines the raw files and decoded tables required before a scored candidate can exist.

## Feature

The planned feature families are near-surface temperature, humidity and boundary layer, wind and marine influence, pressure and synoptic gradient, and cloud-rain-radiation proxies.

## Baseline

The future scored experiment must reproduce `official_forecast_max_c` and `0196` on identical rows before any external forecast model is compared.

## Walk-Forward

No walk-forward model is fit in this folder. The next scored run must use the same chronological folds as 0196 and keep 2024+ confirmation sealed.

## Acceptance

This folder is accepted as a blocker-resolution plan only. It remains `{summary['status']}` until raw historical vintages and decoded T-24 fields exist locally.
""")
    write_text(EXP_DIR / "RESULTS.md", f"""# Results

## Headline

Status: `{summary['status']}`. 0204 did not score a model; it produced an acquisition and timestamp contract.

## Coverage

{base.markdown_table(acquisition_plan, max_rows=10)}

## Global

No global MAE is computed because no new forecast archive exists yet. The current global champion remains 0196 at 1.038829 C MAE.

## Fold

No fold scores are valid until backfilled fields cover the canonical folds. The future experiment must reproduce all comparable fold baselines.

## Year

The required minimum comparable development period is 2000-01-02 through 2023-12-31. A narrower partial archive may only be labeled diagnostic.

## Season

Seasonal feature preservation is mandatory because prior experiments show warm-season and transition-season behavior differ materially.

## Tail

Tail-error features must target parent 0196 errors above 2 C and severe errors above 3 C without being selected on future outcomes.

## Leakage

The proposed timestamp contract is:

{base.markdown_table(timestamp_contract, max_rows=20)}

## Variable Priority

{base.markdown_table(variable_priority, max_rows=20)}
""")
    write_text(EXP_DIR / "CONCLUSION.md", f"""# Conclusion

## Verdict

0204 is `{summary['status']}`. It is a valid blocker-resolution artifact, not a promoted model.

## Learned

The next high-value scored lane is no longer ambiguous. It needs raw historical operational forecast vintages, decoded Hong Kong domain fields, issue and valid times, and a row-level availability assertion before any model training.

## MAE

No candidate MAE is reported. Reporting one would require pretending that an archive exists. The best valid development MAE remains 1.038829 C from 0196.

## Robust

The plan is robust because it rejects reanalysis substitution, live-only snapshots, and files without raw lineage. It also forces the future candidate to compare against 0196 on identical rows.

## Failure

The blocker is external data acquisition, not a code or modeling bug. Without historical issue-time forecast fields, further same-corpus model complexity is likely to produce only tiny gains or prior-validation overfit.

## Promotion

No promotion is possible. The deterministic repair is to build the backfill and decoder described in `artifacts/acquisition_plan.csv` and `artifacts/timestamp_contract.csv`, then run a new scored experiment.
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
    acquisition_plan, variable_priority, timestamp_contract, scored_gate = build_plan_tables()
    write_csv(EXP_DIR / "artifacts" / "acquisition_plan.csv", acquisition_plan)
    write_csv(EXP_DIR / "artifacts" / "variable_priority.csv", variable_priority)
    write_csv(EXP_DIR / "artifacts" / "timestamp_contract.csv", timestamp_contract)
    write_csv(EXP_DIR / "artifacts" / "scored_experiment_gate.csv", scored_gate)
    source_audit = P0202 / "artifacts" / "dataset_availability_matrix.csv"
    data_manifest = pd.DataFrame(
        [
            {"source_id": "0202_source_availability_audit", "path": rel(source_audit), "sha256": sha256_file(source_audit), "size_bytes": source_audit.stat().st_size, "timestamp_fields": "source-specific audited fields", "availability_class": "BLOCKED", "row_count": "", "date_start": "2000-01-02", "date_end": "2023-12-31", "notes": "0202 proved missing historical external forecast archive."},
            {"source_id": "0196_parent_predictions", "path": rel(REPO_ROOT / "experiments/0196_station_network_tail_conditioned_residual_expert/predictions.parquet"), "sha256": sha256_file(REPO_ROOT / "experiments/0196_station_network_tail_conditioned_residual_expert/predictions.parquet"), "size_bytes": (REPO_ROOT / "experiments/0196_station_network_tail_conditioned_residual_expert/predictions.parquet").stat().st_size, "timestamp_fields": "target_date;frozen prediction", "availability_class": "DEPLOYABLE_LAGGED_ONLY", "row_count": 5265, "date_start": "2000-01-02", "date_end": "2023-12-31", "notes": "Current champion required as future baseline."},
        ]
    )
    feature_defs = pd.DataFrame(
        [
            {"feature_name": "external_forecast_vintage_t24_contract", "formula": "Decoded operational forecast fields joined only when available_at_utc <= cutoff_utc.", "input_columns": "issue_time_utc,valid_time_utc,forecast_hour,target_local_date,available_at_utc,raw_content_sha256,decoded_fields", "fit_scope": "future walk-forward after acquisition", "availability_rule": "Blocked in 0204 until raw vintages exist."},
            {"feature_name": "hong_kong_domain_nwp_feature_family", "formula": "Point box gradient and anomaly summaries for temperature humidity pressure wind cloud rain and radiation fields.", "input_columns": "decoded operational NWP fields over Hong Kong domain", "fit_scope": "future scored model only", "availability_rule": "Must satisfy timestamp contract and identical-row baseline gate."},
        ]
    )
    write_csv(EXP_DIR / "data_manifest.csv", data_manifest)
    write_csv(EXP_DIR / "feature_definitions.csv", feature_defs)
    write_text(EXP_DIR / "leakage_audit.md", """# Leakage And Point-In-Time Audit

Status: `PASS`

## Cutoff

The acquisition contract requires available_at_utc <= cutoff_utc for every future decoded feature row.

## Available State

No current or future candidate field is treated as deployable until raw lineage, issue time, valid time, and decoder version are present.

## Target

No target outcomes are read by this planning experiment.

## Rolling

No rolling score is computed. A future scored experiment must use chronological folds only.

## Confirmation

Confirmation rows used: `0`. Owner authorization for confirmation: `false`.
""")
    write_text(EXP_DIR / "REJECTION.md", """# Rejection / Blocker

Status: `BLOCKED_MISSING_DATA`.

0204 cannot score a candidate because the required external forecast vintage archive does not yet exist locally. The deterministic repair is to acquire immutable historical operational forecast raw files, decode the listed field families, preserve timestamp lineage, and rerun as a scored experiment only after the validator can prove T-24 availability.
""")
    write_text(EXP_DIR / "REPRODUCE.md", f"""# Reproduction

From `{REPO_ROOT}`, run:

```powershell
python scripts/run_hkg_t24_0204_external_forecast_vintage_acquisition_plan.py
```

This script writes the acquisition contract and does not read confirmation rows.
""")
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "slug": SLUG,
        "status": "BLOCKED_MISSING_DATA",
        "created_at_utc": created_at,
        "target": "HKO daily Tmax T-24",
        "frame_id": "official_t15_pre2024_5265_rows",
        "date_start": "2000-01-02",
        "date_end": "2023-12-31",
        "n_candidate": None,
        "n_common": None,
        "baseline_id": "official_forecast_max_c",
        "baseline_mae_c": None,
        "candidate_id": "0204_external_forecast_vintage_acquisition_plan",
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
        "data_manifest_sha256": sha256_file(EXP_DIR / "data_manifest.csv"),
        "baseline_n": None,
        "candidate_n": None,
        "development_gate_reached": False,
        "deterministic_repair": "Acquire and decode historical operational forecast vintages with T-24 timestamp proof.",
    }
    write_docs(summary, acquisition_plan, variable_priority, timestamp_contract)
    write_json(EXP_DIR / "summary.json", summary)
    write_json(EXP_DIR / "run_manifest.json", {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "created_at_utc": created_at, "completed_at_utc": utc_now(), "repo_root": str(REPO_ROOT), "script": rel(Path(__file__).resolve()), "spec_sha256": spec_sha, "code_sha256": summary["code_sha256"], "state": "COMPLETED_BLOCKED", "executor_invocation": "Executor skill instructions followed in-process because no separate skill agent activation is available."})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
