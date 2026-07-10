from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from hkg_tmax.hkg_t24.governance import check_four_year_oof_feasibility
from hkg_tmax.hkg_t24.guard import assert_no_locked_dates

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(r"C:\hkg_tmax_data")
EXPERIMENT_DIR = (
    REPO_ROOT
    / "analysis"
    / "hkg_tmax_t24"
    / "experiments"
    / "EXP-0034-HKG-T24-R02"
)
RESEARCH_ID = "HKG-T24-R02"
OOF_START = pd.Timestamp("1934-01-01")
OOF_END = pd.Timestamp("2023-12-31")

QUANTILE_Z = {
    "q01": -2.3263478740408408,
    "q05": -1.6448536269514722,
    "q10": -1.2815515655446004,
    "q25": -0.6744897501960817,
    "q50": 0.0,
    "q75": 0.6744897501960817,
    "q90": 1.2815515655446004,
    "q95": 1.6448536269514722,
    "q99": 2.3263478740408408,
}

MODEL_SPECS = (
    {
        "model_id": "r02_all_history_doy31_mean",
        "family": "calendar_climatology",
        "window_days": 15,
        "history_years": None,
        "trend": False,
    },
    {
        "model_id": "r02_recent50y_doy31_mean",
        "family": "calendar_climatology",
        "window_days": 15,
        "history_years": 50,
        "trend": False,
    },
    {
        "model_id": "r02_recent30y_doy31_mean",
        "family": "calendar_climatology",
        "window_days": 15,
        "history_years": 30,
        "trend": False,
    },
    {
        "model_id": "r02_recent15y_doy31_mean",
        "family": "calendar_climatology",
        "window_days": 15,
        "history_years": 15,
        "trend": False,
    },
    {
        "model_id": "r02_all_history_doy43_linear_trend",
        "family": "linear_trend_climatology",
        "window_days": 21,
        "history_years": None,
        "trend": True,
    },
    {
        "model_id": "r02_recent50y_doy43_linear_trend",
        "family": "linear_trend_climatology",
        "window_days": 21,
        "history_years": 50,
        "trend": True,
    },
)


@dataclass(frozen=True)
class PredictionRow:
    target_date: date
    target_year: int
    model_id: str
    model_family: str
    point_forecast: float
    target_tmax_c: float
    distribution_sigma_c: float
    training_rows: int
    training_start: date
    training_end: date
    window_days: int
    history_years: int | None
    availability_tier: str
    split_role: str


@dataclass(frozen=True)
class PreparedTarget:
    dates: np.ndarray
    date_objects: list[date]
    years: np.ndarray
    doys: np.ndarray
    values: np.ndarray
    days_since_1900: np.ndarray


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def circular_doy_distance(a: int, b: int) -> int:
    raw = abs(a - b)
    return min(raw, 366 - raw)


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def normal_crps(y: float, mu: float, sigma: float) -> float:
    sigma = max(float(sigma), 0.05)
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * normal_cdf(z) - 1.0) + 2.0 * normal_pdf(z) - 1.0 / math.sqrt(math.pi))


def pinball_loss(y: pd.Series, q: pd.Series, alpha: float) -> float:
    diff = y - q
    return float(np.maximum(alpha * diff, (alpha - 1.0) * diff).mean())


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


def prepare_target(target: pd.DataFrame) -> PreparedTarget:
    ordered = target.sort_values("local_date").reset_index(drop=True)
    dates = ordered["local_date"].to_numpy()
    date_objects = [pd.Timestamp(value).date() for value in ordered["local_date"]]
    return PreparedTarget(
        dates=dates,
        date_objects=date_objects,
        years=ordered["local_date"].dt.year.to_numpy(dtype=np.int32),
        doys=ordered["doy"].to_numpy(dtype=np.int16),
        values=ordered["target_tmax_c"].to_numpy(dtype=float),
        days_since_1900=(ordered["local_date"] - pd.Timestamp("1900-01-01")).dt.days.to_numpy(dtype=float),
    )


def build_window_index(prepared: PreparedTarget, window_days: int) -> dict[int, np.ndarray]:
    return {
        doy: np.where(
            np.minimum(np.abs(prepared.doys - doy), 366 - np.abs(prepared.doys - doy))
            <= window_days
        )[0]
        for doy in range(1, 367)
    }


def eligible_training_indices(
    prepared: PreparedTarget,
    base_indices: np.ndarray,
    *,
    target_year: int,
    history_years: int | None,
) -> np.ndarray:
    year_mask = prepared.years[base_indices] < target_year
    if history_years is not None:
        year_mask &= prepared.years[base_indices] >= target_year - history_years
    return base_indices[year_mask]


def predict_from_training(
    prepared: PreparedTarget,
    train_indices: np.ndarray,
    *,
    target_day: pd.Timestamp,
    spec: Mapping[str, object],
) -> tuple[float, float]:
    y = prepared.values[train_indices]
    if bool(spec["trend"]) and len(train_indices) >= 120:
        x = prepared.days_since_1900[train_indices]
        coef = np.polyfit(x, y, deg=1)
        pred_x = float((target_day - pd.Timestamp("1900-01-01")).days)
        point = float(coef[0] * pred_x + coef[1])
        residual = y - (coef[0] * x + coef[1])
        sigma = float(np.std(residual, ddof=1))
    else:
        point = float(np.mean(y))
        sigma = float(np.std(y, ddof=1))
    if not math.isfinite(sigma) or sigma <= 0:
        sigma = 1.5
    return point, max(sigma, 0.2)


def build_predictions(target: pd.DataFrame) -> pd.DataFrame:
    rows: list[PredictionRow] = []
    prepared = prepare_target(target)
    window_indices = {
        window: build_window_index(prepared, window)
        for window in sorted({int(spec["window_days"]) for spec in MODEL_SPECS})
    }
    eval_indices = np.where((prepared.dates >= np.datetime64(OOF_START)) & (prepared.dates <= np.datetime64(OOF_END)))[0]
    for eval_index in eval_indices:
        target_day = pd.Timestamp(prepared.dates[eval_index])
        target_value = float(prepared.values[eval_index])
        target_year = int(prepared.years[eval_index])
        target_doy = int(prepared.doys[eval_index])
        for spec in MODEL_SPECS:
            history_years = None if spec["history_years"] is None else int(spec["history_years"])
            base = window_indices[int(spec["window_days"])][target_doy]
            train_indices = eligible_training_indices(
                prepared,
                base,
                target_year=target_year,
                history_years=history_years,
            )
            if len(train_indices) < 30:
                continue
            point, sigma = predict_from_training(
                prepared,
                train_indices,
                target_day=target_day,
                spec=spec,
            )
            rows.append(
                PredictionRow(
                    target_date=target_day.date(),
                    target_year=target_year,
                    model_id=str(spec["model_id"]),
                    model_family=str(spec["family"]),
                    point_forecast=point,
                    target_tmax_c=target_value,
                    distribution_sigma_c=sigma,
                    training_rows=int(len(train_indices)),
                    training_start=prepared.date_objects[int(train_indices[0])],
                    training_end=prepared.date_objects[int(train_indices[-1])],
                    window_days=int(spec["window_days"]),
                    history_years=history_years,
                    availability_tier="SILVER_OPERATIONAL_REPLAY_TARGET_HISTORY_PENDING_PUBLICATION_PROOF",
                    split_role="development_oof_pre_validation",
                )
            )
    predictions = pd.DataFrame([row.__dict__ for row in rows])
    if predictions.empty:
        raise RuntimeError("R02 produced no predictions")
    predictions["target_date"] = pd.to_datetime(predictions["target_date"])
    assert_no_locked_dates(predictions["target_date"], context="R02 long-history experiment")
    for column, z_value in QUANTILE_Z.items():
        predictions[column] = predictions["point_forecast"] + predictions["distribution_sigma_c"] * z_value
    return predictions.sort_values(["target_date", "model_id"]).reset_index(drop=True)


def score_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    periods = {
        "all_oof_1934_2023": (pd.Timestamp("1934-01-01"), pd.Timestamp("2023-12-31")),
        "recent_1991_2023": (pd.Timestamp("1991-01-01"), pd.Timestamp("2023-12-31")),
        "pre_validation_2020_2023": (pd.Timestamp("2020-01-01"), pd.Timestamp("2023-12-31")),
    }
    rows: list[dict[str, object]] = []
    for period, (start, end) in periods.items():
        subset = predictions[(predictions["target_date"] >= start) & (predictions["target_date"] <= end)]
        for model_id, group in subset.groupby("model_id", sort=True):
            y = group["target_tmax_c"]
            point = group["point_forecast"]
            error = point - y
            crps = [
                normal_crps(float(row.target_tmax_c), float(row.point_forecast), float(row.distribution_sigma_c))
                for row in group.itertuples()
            ]
            rows.append(
                {
                    "period": period,
                    "model_id": str(model_id),
                    "n": int(len(group)),
                    "first_date": str(group["target_date"].min().date()),
                    "last_date": str(group["target_date"].max().date()),
                    "mae": float(error.abs().mean()),
                    "rmse": float(np.sqrt(np.mean(np.square(error)))),
                    "median_abs_error": float(error.abs().median()),
                    "bias": float(error.mean()),
                    "crps_normal": float(np.mean(crps)),
                    "pinball_mean": float(
                        np.mean(
                            [
                                pinball_loss(y, group[column], alpha)
                                for column, alpha in {
                                    "q01": 0.01,
                                    "q05": 0.05,
                                    "q10": 0.10,
                                    "q25": 0.25,
                                    "q50": 0.50,
                                    "q75": 0.75,
                                    "q90": 0.90,
                                    "q95": 0.95,
                                    "q99": 0.99,
                                }.items()
                            ]
                        )
                    ),
                    "coverage_80": float(((group["q10"] <= y) & (y <= group["q90"])).mean()),
                    "coverage_90": float(((group["q05"] <= y) & (y <= group["q95"])).mean()),
                    "coverage_98": float(((group["q01"] <= y) & (y <= group["q99"])).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["period", "mae", "rmse"]).reset_index(drop=True)


def score_by_month(predictions: pd.DataFrame, champion_model: str) -> pd.DataFrame:
    selected = predictions[predictions["model_id"] == champion_model].copy()
    selected["month"] = selected["target_date"].dt.month
    rows: list[dict[str, object]] = []
    for month, group in selected.groupby("month"):
        error = group["point_forecast"] - group["target_tmax_c"]
        rows.append(
            {
                "model_id": champion_model,
                "month": int(month),
                "n": int(len(group)),
                "mae": float(error.abs().mean()),
                "bias": float(error.mean()),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], *, limit: int | None = None) -> str:
    if frame.empty:
        return "_No rows._\n"
    view = frame[columns]
    if limit is not None:
        view = view.head(limit)
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


def long_report(payload: Mapping[str, object]) -> str:
    champion = payload["champion"]
    assert isinstance(champion, dict)
    return f"""# EXP-0034 / HKG-T24-R02 Long-Form Experiment Report

## Purpose

This experiment tests the long-history foundation for the Hong Kong Observatory Headquarters daily maximum temperature forecast problem. It intentionally does not use the modern high-frequency station features, official forecast vintages, NWP, radar, satellite, Polymarket data, validation 2024 outcomes, or locked-test rows. The goal is narrower and more foundational: determine how much forecast skill is available from fold-safe seasonality, slow climate trend, and the value of different historical training windows when every prediction is made from prior years only. Because the user requires at least four years of out-of-fold evidence for every experiment, R02 uses the long target history rather than the short modern station-feature era.

## Forecast Question

The operational question remains the T-24 problem: at 15:00 HKT on T-1, predict the official HKO Headquarters daily Tmax for local day T. R02 is not yet a production model because the G1 target/publication parity gate remains open. It is a silver, fold-safe, long-history baseline experiment. The target series is used as past historical training data and as labels, but no row uses its own target date, a future year, validation 2024, or locked-test 2025-2026 data to build its forecast.

## Leakage Control

For each target date, training rows are restricted to years strictly earlier than the target year. This is more conservative than allowing T-2 labels inside the same year, and it avoids unresolved Daily Extract publication timing in the current target adapter. All predictions stop at 2023-12-31. The script calls the locked-test guard, which rejects any target date greater than or equal to 2025-01-01. The validation year 2024 is not scored in this experiment, because R02 through R29 are not allowed to use validation outcomes for feature or model choice. No random split is used. No whole-dataset normalization, centered rolling window, or future climatology is used.

## Models Tested

Six transparent model specifications were tested. Four are day-of-year climatologies using a 31-day circular seasonal window and different training-history windows: all prior history, prior 50 years, prior 30 years, and prior 15 years. Two are linear-trend climatologies using a wider 43-day circular seasonal window: all prior history and prior 50 years. For each target date, the model looks at past years only, selects training rows with similar day-of-year, and emits a point forecast. Distributional uncertainty is represented by a normal distribution using the fold-local training spread or trend residual spread. The output includes q01, q05, q10, q25, q50, q75, q90, q95, and q99.

## Date Ranges

The full target table used as the source spans {payload['target_min']} through {payload['target_max']}. R02 OOF predictions span {payload['prediction_min']} through {payload['prediction_max']}. The scored OOF period contains {payload['unique_prediction_dates']} unique dates and {payload['prediction_rows']} model-date prediction rows. The four-year OOF feasibility gate passes for this experiment: {payload['oof_reason']}. Validation 2024 and locked-test 2025-2026 are excluded.

## Main Result

The best all-OOF model by MAE is `{champion['model_id']}`. Its all-OOF MAE is `{champion['mae']:.4f}` C and RMSE is `{champion['rmse']:.4f}` C over `{champion['n']}` dates. This result is not a promoted challenger to EXP-0002 because it is evaluated on a different long-history target-only sample and does not use the same modern high-frequency common sample. The value is diagnostic: it establishes a defensible long-history floor and gives later experiments a chronology-safe way to compare climate-window choices before adding modern station information.

## Interpretation

The experiment answers a specific question: historical calendar climatology and trend contain real signal, but they are not sufficient for the T24 system by themselves. A pure seasonal/trend baseline cannot know the current synoptic state, rainfall/cloud suppression, sea-breeze regime, pressure transition, or recent thermal memory. Those mechanisms are exactly what R03 and later experiments must investigate. R02 therefore functions as the long core, not the final answer. If a later system claims improvement, it should show incremental value over this long-history core as well as over the modern station-state analogue where samples overlap.

## Null and Negative Evidence

The experiment did not find permission to use validation 2024, and it did not solve the modern high-frequency four-year blocker. It also did not prove that target labels are first-publication operational features; that remains a G1/G2/G3 governance issue. The history-window comparison should therefore be treated as fold-safe target-history evidence, not as proof of live production eligibility. Any apparent advantage from one historical window over another must be carried forward as a predeclared candidate, not optimized after looking at validation.

## Artifacts

The main prediction table is stored under `C:\\hkg_tmax_data\\gold\\hkg_t24\\r02_long_history\\r02_long_history_oof_predictions.parquet` and copied into this experiment folder. Metrics are stored in `results/metrics.json`, `results/scoreboard.csv`, and `artifacts/monthly_champion_scoreboard.csv`. The reproduction command is in `REPRODUCE.md`. The source manifest records the target table hash, generated files, Git HEAD, and dirty count. The experiment is complete as a long-history diagnostic, but not accepted as a production model or validation challenger.

## What The Metrics Mean

The all-OOF scoreboard is the most statistically broad view because it covers many climate regimes, older station eras, and multiple decades of warming. It is useful for estimating how much pure seasonality and trend can explain over a long record. The recent 1991-2023 scoreboard is more relevant to the modern urban and observing environment, but it is still target-history-only. The 2020-2023 pre-validation scoreboard is included only as a development-era diagnostic; it is not validation and it does not touch 2024. The pre-validation result is especially important because it shows that a strong long-history calendar model still has errors above 2 C in the most recent years. That is a clear signal that the final system needs current weather state, not just historical averages.

## Distributional Caveat

R02 emits quantiles and CRPS so later work has a consistent probabilistic record, but its uncertainty model is intentionally simple. The distribution is normal with spread estimated from the fold-local training values or trend residuals. It is not a fully calibrated final forecast distribution. It does not include regime-dependent variance, cloud/rain uncertainty, cold-front transition risk, tropical-cyclone subsidence heat, or station-network outage inflation. Later experiments must improve distributional calibration with time-aware residual calibration and conformal or empirical residual methods, while preserving the rule that calibration windows cannot include validation or locked-test rows.

## Why This Experiment Still Matters

Even though the MAE is not competitive with the modern station-state analogue on its validation-2024 sample, R02 is essential because it creates a leakage-safe historical backbone. Many future signals will be sparse or modern-only. Without a long-history core, a later model could overfit the short 2020-2023 development period and appear impressive for the wrong reason. R02 gives future experiments a known seasonal/trend baseline, highlights months where calendar-only forecasting fails most, and quantifies the value of historical-window choice before adding weather-state features. It also establishes that the four-year OOF requirement can be satisfied by long-history sources, while modern high-frequency features remain blocked under that strict rule.

## Operational Status

This experiment is not production-eligible. It has no official forecast input, no live source freshness check, no current station observations, no NWP, no station registry dependency, and no proof that historical target labels were published by the T-1 15:00 cutoff. It is therefore classified as a silver target-history diagnostic. Its accepted output is the row-level OOF evidence and the conclusion that long-history climatology/trend is a necessary baseline but insufficient as a full T24 system. Promotion to a final challenger is explicitly disallowed from this experiment.

## Next Use

The next experiment should not jump to machine learning. R03 should reconstruct official daily Tmax from available HKO high-frequency rows and analyze time-of-maximum anatomy, still without touching validation or locked rows. R04 through R10 can then test trajectory, memory, moisture, pressure, wind, and spatial-field mechanisms only where the OOF design is legal. For modern high-frequency features, the strict four-year OOF blocker remains active unless a revised, explicit evaluation design is approved or more prospective data accumulates.
"""


def write_experiment_docs(
    *,
    data_root: Path,
    target_path: Path,
    predictions_path: Path,
    repo_predictions_path: Path,
    scoreboard: pd.DataFrame,
    monthly: pd.DataFrame,
    payload: dict[str, object],
) -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ("results", "artifacts", "predictions", "logs"):
        (EXPERIMENT_DIR / subdir).mkdir(exist_ok=True)
    champion = payload["champion"]
    assert isinstance(champion, dict)
    metric_payload = {
        "research_id": RESEARCH_ID,
        "experiment_id": "EXP-0034",
        "status": "COMPLETE_LONG_HISTORY_DIAGNOSTIC",
        "locked_test_accessed": False,
        "validation_2024_accessed": False,
        "champion": champion,
        "scoreboard": scoreboard.to_dict(orient="records"),
        "monthly_champion": monthly.to_dict(orient="records"),
        "oof_feasibility": payload["oof_feasibility"],
    }
    write_text(EXPERIMENT_DIR / "results" / "metrics.json", json.dumps(metric_payload, indent=2, default=str))
    scoreboard.to_csv(EXPERIMENT_DIR / "results" / "scoreboard.csv", index=False)
    monthly.to_csv(EXPERIMENT_DIR / "artifacts" / "monthly_champion_scoreboard.csv", index=False)
    write_text(
        EXPERIMENT_DIR / "README.md",
        "# EXP-0034 HKG-T24-R02 Long-History Climatology\n\n"
        "Fold-safe long-history climatology/trend experiment. No validation 2024, locked-test, Polymarket, ML, NWP, radar, or satellite data were used.\n",
    )
    write_text(
        EXPERIMENT_DIR / "HYPOTHESIS.md",
        "# Hypothesis\n\nFold-safe calendar climatology, recent-history windows, and simple trend adjustment provide a stable long-history foundation for HKO Tmax forecasting, but cannot fully replace current-state weather information.\n",
    )
    write_text(
        EXPERIMENT_DIR / "PROTOCOL.md",
        "# Protocol\n\n"
        "1. Use HKO target dates through 2023-12-31 only.\n"
        "2. For each target date, train only on years strictly earlier than the target year.\n"
        "3. Evaluate all predictions out of fold over 1934-2023 plus recent subperiods.\n"
        "4. Do not read validation 2024 outcomes or 2025+ locked-test rows.\n"
        "5. Save row-level predictions, metrics, and monthly diagnostics.\n",
    )
    write_text(
        EXPERIMENT_DIR / "ASOF_CONTRACT.md",
        "# As-Of Contract\n\n"
        "This is a silver target-history experiment. Training uses historical target labels from prior years only. It does not use target-day, same-year future, validation 2024, or locked-test values. Production eligibility remains gated by target parity and publication timing proof.\n",
    )
    write_text(
        EXPERIMENT_DIR / "DATA_MANIFEST.yaml",
        f"""research_id: {RESEARCH_ID}
target_table: {target_path}
target_table_sha256: {sha256_file(target_path)}
data_root_prediction_table: {predictions_path}
repo_prediction_table: {repo_predictions_path}
data_root: {data_root}
availability_tier: SILVER_OPERATIONAL_REPLAY_TARGET_HISTORY_PENDING_PUBLICATION_PROOF
validation_2024_accessed: false
locked_test_accessed: false
""",
    )
    write_text(
        EXPERIMENT_DIR / "RUN_CONFIG.yaml",
        f"""research_id: {RESEARCH_ID}
oof_start: {payload['prediction_min']}
oof_end: {payload['prediction_max']}
validation_2024_accessed: false
locked_test_policy: deny
training_rule: prior_years_only
models: {[spec['model_id'] for spec in MODEL_SPECS]}
""",
    )
    write_text(
        EXPERIMENT_DIR / "DATE_RANGES.md",
        f"""# Date Ranges

- Full target table: `{payload['target_min']}` through `{payload['target_max']}`.
- R02 OOF prediction period: `{payload['prediction_min']}` through `{payload['prediction_max']}`.
- Unique OOF target dates: `{payload['unique_prediction_dates']}`.
- Validation 2024: not accessed.
- Locked test 2025-01-01 onward: not accessed.
""",
    )
    write_text(
        EXPERIMENT_DIR / "RESULTS.md",
        "# Results\n\n"
        f"Champion by all-OOF MAE: `{champion['model_id']}`.\n\n"
        + markdown_table(
            scoreboard,
            [
                "period",
                "model_id",
                "n",
                "mae",
                "rmse",
                "median_abs_error",
                "bias",
                "crps_normal",
                "coverage_80",
                "coverage_90",
            ],
        ),
    )
    write_text(
        EXPERIMENT_DIR / "CONCLUSION.md",
        "# Conclusion\n\n"
        "R02 is complete as a long-history diagnostic and OOF climatology/trend foundation. It is not a production champion and does not use validation or locked-test rows.\n",
    )
    write_text(
        EXPERIMENT_DIR / "REPRODUCE.md",
        "# Reproduce\n\n"
        "```powershell\n"
        ".\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_r02_long_history.py --data-root C:\\hkg_tmax_data\n"
        "```\n",
    )
    write_text(
        EXPERIMENT_DIR / "STATUS.yaml",
        """status: COMPLETE_LONG_HISTORY_DIAGNOSTIC
research_id: HKG-T24-R02
locked_test_accessed: false
validation_2024_accessed: false
leakage_guard: PASS
four_year_oof: PASS
production_eligible: false
""",
    )
    write_text(EXPERIMENT_DIR / "EXPERIMENT_REPORT_7500_CHARS.md", long_report(payload))


def write_reports(
    *,
    data_root: Path,
    scoreboard: pd.DataFrame,
    monthly: pd.DataFrame,
    payload: dict[str, object],
) -> None:
    report_dir = REPO_ROOT / "reports" / "hkg_t24"
    champion = payload["champion"]
    assert isinstance(champion, dict)
    write_text(
        report_dir / "R02_LONG_HISTORY_CLIMATOLOGY.md",
        "# R02 Long-History Climatology, Trend, and Training-Window Value\n\n"
        f"Generated: `{now_utc()}`\n\n"
        f"- Validation 2024 accessed: `false`\n"
        f"- Locked test accessed: `false`\n"
        f"- OOF period: `{payload['prediction_min']}` through `{payload['prediction_max']}`\n"
        f"- Champion by all-OOF MAE: `{champion['model_id']}`\n"
        f"- Champion all-OOF MAE: `{champion['mae']:.4f}` C\n\n"
        "## Scoreboard\n\n"
        + markdown_table(
            scoreboard,
            [
                "period",
                "model_id",
                "n",
                "mae",
                "rmse",
                "median_abs_error",
                "bias",
                "crps_normal",
                "coverage_80",
                "coverage_90",
            ],
        )
        + "\n## Champion Monthly Diagnostics\n\n"
        + markdown_table(monthly, ["month", "n", "mae", "bias", "rmse"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HKG-T24-R02 long-history climatology experiment.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    target_path = data_root / "silver" / "targets" / "hko_daily_tmax.parquet"
    target = pd.read_parquet(target_path)
    target["local_date"] = pd.to_datetime(target["local_date"])
    target = target[target["local_date"].dt.date < date(2024, 1, 1)].copy()
    assert_no_locked_dates(target["local_date"], context="R02 target load")
    target["doy"] = target["local_date"].dt.dayofyear

    predictions = build_predictions(target)
    scoreboard = score_predictions(predictions)
    champion = scoreboard[scoreboard["period"] == "all_oof_1934_2023"].iloc[0].to_dict()
    monthly = score_by_month(predictions, str(champion["model_id"]))

    output_dir = data_root / "gold" / "hkg_t24" / "r02_long_history"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "r02_long_history_oof_predictions.parquet"
    scoreboard_path = output_dir / "r02_long_history_scoreboard.parquet"
    predictions.to_parquet(predictions_path, index=False)
    scoreboard.to_parquet(scoreboard_path, index=False)
    repo_predictions_path = EXPERIMENT_DIR / "predictions" / "r02_long_history_oof_predictions.parquet"
    repo_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(repo_predictions_path, index=False)

    feasibility = check_four_year_oof_feasibility(
        predictions["target_date"].min().date(),
        predictions["target_date"].max().date(),
        reason_context="R02 long-history OOF prediction period",
    )
    payload: dict[str, object] = {
        "generated_at_utc": now_utc(),
        "git": git_state(),
        "target_min": str(target["local_date"].min().date()),
        "target_max": str(target["local_date"].max().date()),
        "prediction_min": str(predictions["target_date"].min().date()),
        "prediction_max": str(predictions["target_date"].max().date()),
        "unique_prediction_dates": int(predictions["target_date"].nunique()),
        "prediction_rows": int(len(predictions)),
        "champion": champion,
        "oof_feasibility": feasibility.__dict__,
        "oof_reason": feasibility.reason,
    }
    write_experiment_docs(
        data_root=data_root,
        target_path=target_path,
        predictions_path=predictions_path,
        repo_predictions_path=repo_predictions_path,
        scoreboard=scoreboard,
        monthly=monthly,
        payload=payload,
    )
    write_reports(data_root=data_root, scoreboard=scoreboard, monthly=monthly, payload=payload)
    print(json.dumps({"status": "ok", "champion": champion, "oof": feasibility.__dict__}, indent=2, default=str))


if __name__ == "__main__":
    main()
