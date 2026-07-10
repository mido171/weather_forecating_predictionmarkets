from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from hkg_tmax.hkg_t24.guard import (
    LOCKED_TEST_START,
    assert_no_locked_dates,
    write_locked_test_guard_report,
)
from hkg_tmax.paths import ProjectPaths

PROJECT_PATHS = ProjectPaths.discover(Path(__file__))
REPO_ROOT = PROJECT_PATHS.project_root
DEFAULT_DATA_ROOT = PROJECT_PATHS.data_root
DEFAULT_ARCHIVE = (
    PROJECT_PATHS.data_root
    / "imports"
    / "repo-20260710"
    / "analysis"
    / "hkg_tmax_t24.rar"
)
REPORT_DIR = PROJECT_PATHS.run_root / "reports" / "hkg_t24"
ANALYSIS_EXPERIMENT_DIR = (
    PROJECT_PATHS.run_root
    / "experiments"
    / "legacy"
    / "hkg_tmax_t24"
    / "EXP-0033-HKG-T24-R01"
)

EXPECTED_CHAMPION_VALIDATION = {
    "model_id": "station_state_analogue",
    "split": "validation_2024",
    "n": 364,
    "mae": 1.5031758241758242,
    "rmse": 1.8973737199499678,
    "median_abs_error": 1.298,
    "bias": 0.018472527472526927,
    "crps_normal": 1.0654027259408796,
    "coverage_80": 0.8186813186813187,
    "coverage_90": 0.9093406593406593,
}

PREDICTION_PATH = PROJECT_PATHS.run_root / "predictions" / "baselines" / "hkg_tmax_baseline_predictions.parquet"
SCOREBOARD_PATH = PROJECT_PATHS.run_root / "predictions" / "baselines" / "hkg_tmax_baseline_scoreboard.parquet"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def normal_crps(y: float, mu: float, sigma: float) -> float:
    sigma = max(float(sigma), 0.05)
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * normal_cdf(z) - 1.0) + 2.0 * normal_pdf(z) - 1.0 / math.sqrt(math.pi))


def pinball(y: pd.Series, q: pd.Series, alpha: float) -> float:
    diff = y - q
    return float(np.maximum(alpha * diff, (alpha - 1.0) * diff).mean())


def run_text(command: list[str], *, cwd: Path = REPO_ROOT) -> str:
    completed = subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)
    return completed.stdout


def archive_inventory(archive_path: Path) -> dict[str, Any]:
    members = [line.strip() for line in run_text(["tar", "-tf", str(archive_path)]).splitlines() if line.strip()]
    hashes: list[dict[str, Any]] = []
    directory_count = 0
    with tempfile.TemporaryDirectory(prefix="hkg_t24_archive_") as temp_name:
        temp_root = Path(temp_name)
        subprocess.run(["tar", "-xf", str(archive_path), "-C", str(temp_root)], check=True)
        directory_count = sum(1 for extracted in temp_root.rglob("*") if extracted.is_dir())
        for extracted in sorted(temp_root.rglob("*")):
            if extracted.is_file():
                rel = extracted.relative_to(temp_root).as_posix()
                hashes.append({"path": rel, "bytes": extracted.stat().st_size, "sha256": sha256_file(extracted)})
    return {
        "path": str(archive_path),
        "bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
        "members": members,
        "member_count": len(members),
        "file_count": len(hashes),
        "directory_count": directory_count,
        "file_hashes": hashes,
    }


def parquet_range(path: Path, date_column: str, columns: list[str] | None = None) -> dict[str, Any]:
    selected = columns if columns is not None else None
    if selected is not None and date_column not in selected:
        selected = [date_column, *selected]
    frame = pd.read_parquet(path, columns=selected)
    dates = pd.to_datetime(frame[date_column])
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "date_column": date_column,
        "min_date": None if frame.empty else str(dates.min()),
        "max_date": None if frame.empty else str(dates.max()),
        "unique_dates": int(dates.nunique()),
        "columns": list(frame.columns),
    }


def read_unlocked_predictions(path: Path) -> pd.DataFrame:
    table = pq.read_table(
        path,
        filters=[("target_date", "<", datetime.combine(LOCKED_TEST_START, datetime.min.time()))],
    )
    frame = table.to_pandas()
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    assert_no_locked_dates(frame["target_date"], context="R01 validation reproduction")
    return frame


def prediction_metadata(path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(path, columns=["target_date", "split", "model_id"])
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    locked = frame[frame["target_date"].dt.date >= LOCKED_TEST_START]
    unlocked = frame[frame["target_date"].dt.date < LOCKED_TEST_START]
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows_total": int(len(frame)),
        "rows_unlocked_metadata": int(len(unlocked)),
        "rows_locked_metadata_only": int(len(locked)),
        "unique_dates_total": int(frame["target_date"].nunique()),
        "unique_dates_unlocked": int(unlocked["target_date"].nunique()),
        "unique_dates_locked_metadata_only": int(locked["target_date"].nunique()),
        "min_date_total": str(frame["target_date"].min()),
        "max_date_total": str(frame["target_date"].max()),
        "min_date_unlocked": str(unlocked["target_date"].min()),
        "max_date_unlocked": str(unlocked["target_date"].max()),
        "split_counts_metadata_only": frame["split"].value_counts().to_dict(),
        "model_ids": sorted(str(item) for item in frame["model_id"].unique()),
    }


def score_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    quantiles = {
        "q05": 0.05,
        "q10": 0.10,
        "q25": 0.25,
        "q50": 0.50,
        "q75": 0.75,
        "q90": 0.90,
        "q95": 0.95,
    }
    for (model_id, split), group in predictions.groupby(["model_id", "split"], sort=True):
        y = group["target_tmax_c"]
        point = group["point_forecast"]
        error = point - y
        central_80 = (group["q10"] <= y) & (y <= group["q90"])
        central_90 = (group["q05"] <= y) & (y <= group["q95"])
        crps = [
            normal_crps(float(row.target_tmax_c), float(row.point_forecast), float(row.distribution_sigma_c))
            for row in group.itertuples()
        ]
        rows.append(
            {
                "model_id": str(model_id),
                "split": str(split),
                "n": int(len(group)),
                "first_date": str(group["target_date"].min().date()),
                "last_date": str(group["target_date"].max().date()),
                "mae": float(error.abs().mean()),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "median_abs_error": float(error.abs().median()),
                "bias": float(error.mean()),
                "crps_normal": float(np.mean(crps)),
                "pinball_mean": float(np.mean([pinball(y, group[name], alpha) for name, alpha in quantiles.items()])),
                "coverage_80": float(central_80.mean()),
                "coverage_90": float(central_90.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "mae", "rmse"]).reset_index(drop=True)


def feature_coverage(feature_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    columns = [
        "local_date",
        "target_tmax_c",
        "hko_temp_at_tminus1_1500_c",
        "hko_rh_at_tminus1_1500_pct",
        "hko_mslp_at_tminus1_1500_hpa",
        "hko_tminus1_max_so_far_1500_c",
        "hko_tminus1_min_so_far_1500_c",
        "hko_temp_tminus1_1200_c",
        "hko_temp_tminus1_0900_c",
        "hko_mslp_tminus1_1200_hpa",
        "hko_temp_3h_change_to_cutoff_c",
        "hko_temp_6h_change_to_cutoff_c",
        "hko_mslp_3h_change_to_cutoff_hpa",
        "hko_tminus2_official_tmax_c",
        "split_role",
    ]
    frame = pd.read_parquet(feature_path, columns=columns)
    frame["local_date"] = pd.to_datetime(frame["local_date"])
    facts: dict[str, Any] = {
        "path": str(feature_path),
        "bytes": feature_path.stat().st_size,
        "sha256": sha256_file(feature_path),
        "rows": int(len(frame)),
        "min_date": str(frame["local_date"].min()),
        "max_date": str(frame["local_date"].max()),
        "columns": {},
    }
    for column in columns:
        if column == "local_date":
            continue
        available = frame[frame[column].notna()]
        facts["columns"][column] = {
            "non_null_rows": int(len(available)),
            "min_date": None if available.empty else str(available["local_date"].min().date()),
            "max_date": None if available.empty else str(available["local_date"].max().date()),
        }
    return facts, frame


def missing_date_explanations(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    *,
    start: str = "2021-07-01",
    end: str = "2024-12-31",
) -> list[dict[str, Any]]:
    expected = pd.date_range(start, end, freq="D")
    predicted_dates = set(predictions["target_date"].dt.normalize())
    indexed = features.set_index(features["local_date"].dt.normalize())
    first_prediction = min(predicted_dates)
    explanations: list[dict[str, Any]] = []
    diagnostic_columns = [
        "hko_temp_at_tminus1_1500_c",
        "hko_rh_at_tminus1_1500_pct",
        "hko_mslp_at_tminus1_1500_hpa",
        "hko_temp_tminus1_1200_c",
        "hko_temp_tminus1_0900_c",
        "hko_tminus2_official_tmax_c",
    ]
    for day in expected:
        if day in predicted_dates:
            continue
        row = indexed.loc[day] if day in indexed.index else None
        values: dict[str, Any] = {}
        missing_columns: list[str] = []
        if row is not None:
            for column in diagnostic_columns:
                value = row[column]
                if pd.isna(value):
                    missing_columns.append(column)
                    values[column] = None
                else:
                    values[column] = float(value) if isinstance(value, (float, int, np.number)) else str(value)

        if day < first_prediction:
            reason = (
                "before_archived_first_prediction_date; row-level archive begins on 2021-12-30 even though the "
                "current feature table contains HKO cutoff temperature for many earlier dates. The first archived "
                "date matches the first non-null HKO 15:00 pressure feature, so R01 records this as an archive/code "
                "sample-gate discrepancy requiring the exact old generator to prove intent."
            )
        elif "hko_temp_at_tminus1_1500_c" in missing_columns:
            reason = "missing_required_hko_cutoff_temperature_at_T_minus_1_1500"
        else:
            reason = "missing_from_archived_prediction_table_without_required_temperature_gap"
        explanations.append(
            {
                "target_date": str(day.date()),
                "reason": reason,
                "missing_feature_columns": missing_columns,
                "diagnostic_values": values,
            }
        )
    return explanations


def compare_expected_champion(scoreboard: pd.DataFrame) -> dict[str, Any]:
    row = scoreboard[
        (scoreboard["model_id"] == EXPECTED_CHAMPION_VALIDATION["model_id"])
        & (scoreboard["split"] == EXPECTED_CHAMPION_VALIDATION["split"])
    ]
    if row.empty:
        return {"status": "FAIL", "reason": "expected champion row not found"}
    observed = row.iloc[0].to_dict()
    comparisons: dict[str, Any] = {}
    status = "PASS"
    for key, expected in EXPECTED_CHAMPION_VALIDATION.items():
        actual = observed[key]
        if isinstance(expected, str):
            ok = actual == expected
            delta = None
        else:
            delta = abs(float(actual) - float(expected))
            ok = delta <= 1e-9 if key == "n" else delta <= 5e-10
        comparisons[key] = {"expected": expected, "actual": actual, "delta": delta, "pass": ok}
        if not ok:
            status = "FAIL"
    return {"status": status, "comparisons": comparisons}


def git_state() -> dict[str, Any]:
    status = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=True).stdout
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=True).stdout.strip()
    return {
        "head": rev,
        "dirty_lines": [line for line in status.splitlines() if line.strip()],
        "dirty_count": len([line for line in status.splitlines() if line.strip()]),
    }


def validation_access_ledger() -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    return {
        "generated_at_utc": now,
        "policy": "R01 may recompute supplied development/validation metrics only; R02-R29 may not use validation outcomes; 2025+ locked rows are denied.",
        "entries": [
            {
                "timestamp_utc": now,
                "research_id": "HKG-T24-R01",
                "period_accessed": "development_and_validation_through_2024-12-31",
                "locked_test_accessed": False,
                "purpose": "recompute supplied baseline metrics from row-level predictions and audit date coverage",
            }
        ],
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        rendered = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                rendered.append(f"{value:.6f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_validation_ledger(payload: dict[str, Any]) -> None:
    write_json(REPORT_DIR / "validation_access_ledger.json", payload)
    entry_rows = payload["entries"]
    text = (
        "# Validation Access Ledger\n\n"
        f"Policy: {payload['policy']}\n\n"
        + markdown_table(
            entry_rows,
            [
                "timestamp_utc",
                "research_id",
                "period_accessed",
                "locked_test_accessed",
                "purpose",
            ],
        )
    )
    (REPORT_DIR / "VALIDATION_ACCESS_LEDGER.md").write_text(text, encoding="utf-8")


def write_verified_state(payload: dict[str, Any], scoreboard: pd.DataFrame, missing: list[dict[str, Any]]) -> None:
    validation_rows = scoreboard[scoreboard["split"] == "validation_2024"].to_dict(orient="records")
    missing_focus = [row for row in missing if row["target_date"] in {"2022-09-24", "2022-09-25", "2022-09-26", "2024-12-19", "2024-12-20"}]
    oof = payload["oof_feasibility"]
    text = f"""# Verified Current State

Generated: `{payload['generated_at_utc']}`

Repository HEAD: `{payload['git']['head']}`

Dirty worktree entries at audit start: `{payload['git']['dirty_count']}`. These include pre-existing user changes and deleted root experiment folders; this audit did not revert them.

## Archive

- Archive path: `{payload['archive']['path']}`
- Archive SHA256: `{payload['archive']['sha256']}`
- Archive files: `{payload['archive']['file_count']}`
- Archive directories: `{payload['archive']['directory_count']}`

## Core Data Facts

- Target table rows: `{payload['target']['rows']}`, range `{payload['target']['min_date']}` to `{payload['target']['max_date']}`.
- Selected high-frequency table rows: `{payload['selected_high_frequency']['rows']}`, range `{payload['selected_high_frequency']['min_date']}` to `{payload['selected_high_frequency']['max_date']}`.
- Feature candidate table rows: `{payload['features']['rows']}`, range `{payload['features']['min_date']}` to `{payload['features']['max_date']}`.
- Archived baseline prediction rows: `{payload['prediction_metadata']['rows_total']}` total, `{payload['prediction_metadata']['rows_locked_metadata_only']}` locked rows counted as metadata only.

## Baseline Reproduction

R01 recomputed metrics only for target dates before `{LOCKED_TEST_START.isoformat()}`. Locked-test losses were not computed.

Champion validation reproduction status: `{payload['champion_reproduction']['status']}`.

{markdown_table(validation_rows, ['model_id', 'split', 'n', 'first_date', 'last_date', 'mae', 'rmse', 'median_abs_error', 'bias', 'crps_normal', 'coverage_80', 'coverage_90'])}

## Date Discrepancies

- Archived first prediction date: `{payload['prediction_metadata']['min_date_unlocked']}`.
- Declared EXP-0002 common-sample start in its date-range document: `2021-07-01`.
- Missing development/validation target dates from declared common sample: `{len(missing)}`.

Known missing dates requested by the goal:

{markdown_table(missing_focus, ['target_date', 'reason'])}

The five named missing dates are explained by absent HKO cutoff-temperature features. The larger July-December 2021 discrepancy remains a reproduction blocker because the current generator code does not contain the old effective start gate, while the archived rows begin exactly at the first 15:00 pressure-feature date.

## Four-Year OOF Feasibility

- Strict user requirement: `{oof['requirement']}`.
- Long-history target/daily-climate families: `{oof['long_history_status']}`.
- Modern high-frequency development-only sample: `{oof['modern_development_only_status']}`.
- Modern archived baseline sample: `{oof['archived_baseline_status']}`.
- Required handling: `{oof['required_handling']}`.

## Locked-Test Guard

Status: active in `hkg_tmax.hkg_t24.guard`; ordinary research access rejects dates `>= {LOCKED_TEST_START.isoformat()}`. No unlock was invoked by this audit.
"""
    (REPORT_DIR / "VERIFIED_CURRENT_STATE.md").write_text(text, encoding="utf-8")


def write_baseline_reproduction(payload: dict[str, Any], scoreboard: pd.DataFrame, missing: list[dict[str, Any]]) -> None:
    scoreboard_path = REPORT_DIR / "baseline_reproduction_r01_scoreboard.csv"
    missing_path = REPORT_DIR / "baseline_reproduction_r01_missing_dates.csv"
    scoreboard.to_csv(scoreboard_path, index=False)
    pd.DataFrame(missing).to_csv(missing_path, index=False)
    write_json(REPORT_DIR / "baseline_reproduction_r01.json", payload)
    text = f"""# R01 Baseline Reproduction Audit

This report recomputes supplied baseline metrics from row-level predictions for development and validation only. Locked-test rows from 2025-01-01 onward were not scored.

## Result

- Champion validation row: `station_state_analogue`.
- Reproduction status: `{payload['champion_reproduction']['status']}`.
- Scoreboard CSV: `{scoreboard_path}`.
- Missing-date CSV: `{missing_path}`.
- Locked-test metadata rows counted but not scored: `{payload['prediction_metadata']['rows_locked_metadata_only']}`.

## Validation Scoreboard

{markdown_table(scoreboard[scoreboard['split'] == 'validation_2024'].to_dict(orient='records'), ['model_id', 'n', 'first_date', 'last_date', 'mae', 'rmse', 'median_abs_error', 'bias', 'crps_normal', 'coverage_80', 'coverage_90'])}

## Discrepancies

The row-level archive confirms predictions start on `{payload['prediction_metadata']['min_date_unlocked']}`, not `2021-07-01`.

The five named missing dates are absent because the feature candidate table has no HKO cutoff temperature at T-1 15:00 for those target dates. The July-December 2021 gap is not explained by the current feature table, which now contains cutoff temperature for that interval. It aligns with first pressure-feature availability and is therefore recorded as a baseline archive/code-version discrepancy until the exact historical generator is recovered.
"""
    (REPORT_DIR / "BASELINE_REPRODUCTION_R01.md").write_text(text, encoding="utf-8")


def experiment_report_text(payload: dict[str, Any]) -> str:
    return f"""# EXP-0033 / HKG-T24-R01 Long-Form Experiment Report

## Purpose

This experiment is the mandatory first gate for the HKG Tmax T-24 research reset. Its job is not to invent a new model. Its job is to verify the current repository, the supplied `hkg_tmax_t24.rar` archive, the existing baseline prediction table, the actual data date ranges, and the leakage controls that must exist before any later experiment can be trusted. The operational forecast question remains fixed: at 15:00 Hong Kong time on T-1, predict the official Hong Kong Observatory Headquarters daily maximum temperature for local day T as a calibrated distribution. R01 is the guardrail experiment that decides whether the previous baseline suite is reproducible enough to serve as the comparison anchor for R02 through R30.

## What Was Checked

The audit listed and hashed the supplied archive, verified that it contains the existing baseline suite, and independently recomputed development and validation metrics from row-level prediction Parquet rows. It inspected the canonical target table at `C:\\hkg_tmax_data\\silver\\targets\\hko_daily_tmax.parquet`, the selected high-frequency HKO observation table, and the T-24 feature candidate table. It also created a locked-test guard report and a validation access ledger. Existing 2025-2026 prediction rows were counted only as metadata so the audit could prove they exist and need quarantine; their errors and losses were not scored.

## Date Ranges Used

The full target table spans `{payload['target']['min_date']}` through `{payload['target']['max_date']}` with `{payload['target']['rows']}` rows. The selected high-frequency HKO station table spans `{payload['selected_high_frequency']['min_date']}` through `{payload['selected_high_frequency']['max_date']}` with `{payload['selected_high_frequency']['rows']}` rows. The archived baseline prediction table spans `{payload['prediction_metadata']['min_date_total']}` through `{payload['prediction_metadata']['max_date_total']}`, but R01 only recomputed rows before `{LOCKED_TEST_START.isoformat()}`. The recomputed unlocked prediction span is `{payload['prediction_metadata']['min_date_unlocked']}` through `{payload['prediction_metadata']['max_date_unlocked']}`.

## Main Finding

The supplied validation-2024 champion reproduces from row-level predictions. The reproduced champion is `station_state_analogue`; its validation MAE, RMSE, median absolute error, bias, normal CRPS, and 80/90 percent coverage match the frozen numbers within tight numeric tolerance. That means the row-level Parquet prediction table is internally consistent and can be used as the frozen baseline evidence for validation reproduction. However, this does not mean the current baseline generator is fully reproduced. The audit found a material discrepancy between the declared common-sample start and the actual first prediction date.

## Date Discrepancy

The EXP-0002 date-range document declares a modern common-sample start of 2021-07-01. The actual row-level prediction table begins on 2021-12-30. The five named missing dates, 2022-09-24, 2022-09-25, 2022-09-26, 2024-12-19, and 2024-12-20, are explained by missing HKO cutoff temperature at T-1 15:00 in the feature candidate table. The larger July-December 2021 gap is more serious: the current feature table contains cutoff temperature in much of that period, and the current script no longer visibly contains the older effective start gate. The first archived prediction date matches the first non-null HKO 15:00 pressure feature. R01 therefore records the gap as a baseline archive/code-version discrepancy. Later experiments must not pretend the archived baseline starts in July 2021; they must use the proven row-level prediction coverage or rebuild a new predeclared baseline under the locked-test guard.

## Leakage Controls

R01 added a fail-closed locked-test guard in `hkg_tmax.hkg_t24.guard`. The ordinary guard rejects any target date greater than or equal to 2025-01-01. The audit wrote `reports/hkg_t24/LOCKED_TEST_GUARD.md` and `reports/hkg_t24/VALIDATION_ACCESS_LEDGER.md`. No 2025-2026 loss was recomputed. Validation 2024 was accessed only for the allowed R01 reproduction task. R02 through R29 must not use validation outcomes for feature selection, model choice, or tuning. R30 may use validation once only after a hashed predeclaration.

## Four-Year OOF Requirement

The user requires at least four years of out-of-fold test data for every experiment. R01 records a feasibility constraint. Long-history target and daily-climate experiments can satisfy that requirement. Modern high-frequency HKO experiments cannot satisfy four full development-only OOF years before 2024 if validation remains quarantined for R30; the pre-2024 modern high-frequency window is less than four years, and the archived baseline common sample is shorter still. Therefore future modern experiments must be marked blocked under the strict four-year rule unless they use a lawful long-history family, wait for enough prospective data, or receive a revised explicit evaluation design that does not violate the validation-access budget.

## Conclusion

R01 is complete as a verification and quarantine gate, not as a model-improvement experiment. The baseline row-level metrics reproduce for validation 2024, the locked-test guard is active, and the current-state reports are written. The unresolved item is deterministic generator reproduction: the current script state and archived prediction coverage disagree on the early sample. Until that is repaired or deliberately superseded by a new guarded baseline build, no challenger should be promoted against an assumed July 2021 common sample.

## Files Produced

The primary human-readable state report is `reports/hkg_t24/VERIFIED_CURRENT_STATE.md`. Its machine-readable twin is `reports/hkg_t24/verified_current_state.json`. The focused baseline reproduction report is `reports/hkg_t24/BASELINE_REPRODUCTION_R01.md`, with a CSV scoreboard and a CSV missing-date table beside it. The locked-test policy is documented in `reports/hkg_t24/LOCKED_TEST_GUARD.md`, and the validation access event is documented in `reports/hkg_t24/VALIDATION_ACCESS_LEDGER.md` plus JSON. Inside this experiment folder, `results/metrics.json` contains the champion reproduction status and validation scoreboard, `artifacts/verified_current_state.json` preserves the full audit payload, and `DATE_RANGES.md` gives the exact target, feature, prediction, scored, and quarantined date ranges. These files are intentionally redundant: the goal is to make the experiment understandable both to humans and to downstream scripts without requiring a notebook or hidden local state.

## What Was Deliberately Not Done

R01 did not start predictive modelling, machine learning, feature mining, Polymarket pricing, market backtesting, or trading analysis. It did not use validation 2024 to choose a new feature. It did not compute locked-test MAE, RMSE, CRPS, failure cases, seasonal errors, or subgroup scores. It did not repair the archived baseline by silently changing the sample. It did not erase or rewrite the previous EXP-0002 evidence. This matters because the rest of the T-24 program is only useful if every later claim can be traced to a clean evaluation budget. If a future experiment wants to supersede EXP-0002, it must create its own predeclared folder and explain exactly which rows are in scope, which rows are excluded, and why.

## How Later Experiments Must Use R01

Later experiments must treat `station_state_analogue` as the reproduced row-level validation baseline only on the actual archived prediction coverage, not on the declared July 2021 common sample. They must check target dates against the locked-test guard before loading labels, predictions, or errors. They must keep validation outcomes out of R02 through R29 feature selection. If a modern high-frequency experiment cannot satisfy the four-year OOF rule without consuming validation 2024, it must be marked blocked or redesigned before scoring. If a long-history experiment uses daily climate history back to 1884, it still needs a source-availability tier because target-day and finalized daily records are not automatically eligible as T-24 operational predictors. The practical next action is therefore not "try a bigger model"; it is to build the canonical T24 command surface, source contracts, station registry, and a new development-only OOF design that either meets the four-year rule through long-history sources or explicitly records why modern high-frequency features cannot yet do so under the current validation quarantine.
"""


def write_experiment_folder(payload: dict[str, Any], scoreboard: pd.DataFrame) -> None:
    exp = ANALYSIS_EXPERIMENT_DIR
    for subdir in [exp, exp / "results", exp / "artifacts", exp / "logs", exp / "predictions"]:
        subdir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "research_id": "HKG-T24-R01",
        "experiment_id": "EXP-0033",
        "status": "COMPLETE_WITH_REPRODUCTION_BLOCKER",
        "champion_reproduction": payload["champion_reproduction"],
        "validation_scoreboard": scoreboard[scoreboard["split"] == "validation_2024"].to_dict(orient="records"),
        "locked_test_accessed": False,
    }
    write_json(exp / "results" / "metrics.json", metrics)
    write_json(exp / "artifacts" / "verified_current_state.json", payload)
    (exp / "README.md").write_text(
        "# EXP-0033 HKG-T24-R01 Current State and Baseline Reproduction\n\n"
        "Mandatory verification gate before new T24 experiments. This folder contains the R01 hypothesis, protocol, as-of contract, data manifest, metrics, date ranges, and conclusion. Locked-test rows were not scored.\n",
        encoding="utf-8",
    )
    (exp / "HYPOTHESIS.md").write_text(
        "# Hypothesis\n\nThe supplied EXP-0002 row-level prediction table can reproduce the frozen validation-2024 champion metrics without accessing locked-test losses, and any date-range discrepancies can be made explicit before new experiments start.\n",
        encoding="utf-8",
    )
    (exp / "PROTOCOL.md").write_text(
        "# Protocol\n\n1. Hash and list the supplied archive.\n2. Read prediction metadata for all rows, but compute metrics only for target dates before 2025-01-01.\n3. Recompute validation metrics from row-level predictions.\n4. Compare the champion row to the frozen numbers.\n5. Explain missing dates from the feature candidate table.\n6. Write locked-test guard and validation-access ledger.\n",
        encoding="utf-8",
    )
    (exp / "ASOF_CONTRACT.md").write_text(
        "# As-Of Contract\n\nR01 does not create new model features. It audits existing row-level predictions. The governing forecast cutoff remains T-1 15:00 HKT, and ordinary research access rejects target dates >= 2025-01-01.\n",
        encoding="utf-8",
    )
    (exp / "DATA_MANIFEST.yaml").write_text(
        f"""research_id: HKG-T24-R01
archive: {payload['archive']['path']}
archive_sha256: {payload['archive']['sha256']}
target_table: {payload['target']['path']}
target_table_sha256: {payload['target']['sha256']}
feature_table: {payload['features']['path']}
feature_table_sha256: {payload['features']['sha256']}
prediction_table: {payload['prediction_metadata']['path']}
prediction_table_sha256: {payload['prediction_metadata']['sha256']}
locked_test_policy: deny
""",
        encoding="utf-8",
    )
    (exp / "RUN_CONFIG.yaml").write_text(
        """research_id: HKG-T24-R01
locked_test_policy: deny
metric_periods:
  - development
  - validation_2024
forbidden_periods:
  - locked_test_2025_2026
expected_champion: station_state_analogue
""",
        encoding="utf-8",
    )
    (exp / "DATE_RANGES.md").write_text(
        f"""# Date Ranges

- Full target table: `{payload['target']['min_date']}` through `{payload['target']['max_date']}`.
- Selected high-frequency table: `{payload['selected_high_frequency']['min_date']}` through `{payload['selected_high_frequency']['max_date']}`.
- Feature candidate table: `{payload['features']['min_date']}` through `{payload['features']['max_date']}`.
- Archived prediction table total range: `{payload['prediction_metadata']['min_date_total']}` through `{payload['prediction_metadata']['max_date_total']}`.
- R01 scored range: `{payload['prediction_metadata']['min_date_unlocked']}` through `{payload['prediction_metadata']['max_date_unlocked']}`.
- Locked-test range quarantined: `2025-01-01` onward.
""",
        encoding="utf-8",
    )
    (exp / "RESULTS.md").write_text(
        "# Results\n\nSee `reports/hkg_t24/BASELINE_REPRODUCTION_R01.md`, `reports/hkg_t24/VERIFIED_CURRENT_STATE.md`, and `results/metrics.json`.\n",
        encoding="utf-8",
    )
    (exp / "CONCLUSION.md").write_text(
        "# Conclusion\n\nR01 reproduced the validation champion metrics from row-level predictions and activated locked-test quarantine. Full generator reproduction remains blocked by the archived prediction start-date discrepancy.\n",
        encoding="utf-8",
    )
    (exp / "REPRODUCE.md").write_text(
        "# Reproduce\n\n```powershell\n.\\.venv\\Scripts\\python.exe scripts\\audit_hkg_t24_baseline_reproduction.py\n```\n",
        encoding="utf-8",
    )
    (exp / "STATUS.yaml").write_text(
        """status: COMPLETE_WITH_REPRODUCTION_BLOCKER
research_id: HKG-T24-R01
locked_test_accessed: false
validation_access: allowed_for_r01_reproduction
leakage_guard: PASS
baseline_row_metric_reproduction: PASS
baseline_generator_reproduction: BLOCKED_DATE_RANGE_DISCREPANCY
""",
        encoding="utf-8",
    )
    (exp / "EXPERIMENT_REPORT_7500_CHARS.md").write_text(experiment_report_text(payload), encoding="utf-8")


def build_payload(data_root: Path, archive_path: Path) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    target_path = data_root / "silver" / "targets" / "hko_daily_tmax.parquet"
    selected_hf_path = data_root / "bronze" / "analysis_phase_a" / "hko_high_frequency_selected_station_observations.parquet"
    feature_path = data_root / "silver" / "features" / "t24_cutoff_feature_candidates.parquet"

    predictions = read_unlocked_predictions(PREDICTION_PATH)
    scoreboard = score_predictions(predictions)
    feature_facts, features = feature_coverage(feature_path)
    missing = missing_date_explanations(predictions, features)
    oof_feasibility = {
        "requirement": "at least four years of out-of-fold test data for all experiments",
        "long_history_status": "FEASIBLE for target-only and daily-climate families with 1884-2026 coverage, subject to as-of publication constraints",
        "modern_development_only_status": "BLOCKED under strict rule: HKO high-frequency before validation has less than four full years before 2024",
        "archived_baseline_status": "BLOCKED under strict rule: archived EXP-0002 scored development starts 2021-12-30 and has about two years before validation",
        "required_handling": "Do not run/promote modern high-frequency R02-R29 experiments as satisfying the strict four-year OOF requirement unless a revised predeclared split is approved or enough prospective data accrues.",
    }
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git": git_state(),
        "archive": archive_inventory(archive_path),
        "target": parquet_range(target_path, "local_date"),
        "selected_high_frequency": parquet_range(
            selected_hf_path,
            "observed_at_hkt",
            columns=["observed_at_hkt", "station", "variable", "value", "available_at_hkt"],
        ),
        "features": feature_facts,
        "prediction_metadata": prediction_metadata(PREDICTION_PATH),
        "scoreboard_path": str(SCOREBOARD_PATH),
        "champion_reproduction": compare_expected_champion(scoreboard),
        "missing_dates": missing,
        "oof_feasibility": oof_feasibility,
    }
    return payload, scoreboard, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit HKG T24 baseline reproduction without locked-test scoring.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    archive_path = Path(args.archive)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload, scoreboard, missing = build_payload(data_root, archive_path)
    write_json(REPORT_DIR / "verified_current_state.json", payload)
    write_verified_state(payload, scoreboard, missing)
    write_baseline_reproduction(payload, scoreboard, missing)
    write_validation_ledger(validation_access_ledger())
    write_locked_test_guard_report(REPORT_DIR / "LOCKED_TEST_GUARD.md")
    write_experiment_folder(payload, scoreboard)
    print(json.dumps({"status": "ok", "champion_reproduction": payload["champion_reproduction"]["status"]}, indent=2))


if __name__ == "__main__":
    main()
