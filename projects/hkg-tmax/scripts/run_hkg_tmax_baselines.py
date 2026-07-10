from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hkg_tmax.hkg_t24.guard import LOCKED_TEST_START, assert_no_locked_dates
from hkg_tmax.paths import ProjectPaths

PROJECT_PATHS = ProjectPaths.discover(Path(__file__))
REPO_ROOT = PROJECT_PATHS.project_root
DATA_ROOT = PROJECT_PATHS.data_root
REPORTS_ROOT = PROJECT_PATHS.run_root / "reports"
ANALYSIS_ROOT = PROJECT_PATHS.run_root / "experiments" / "legacy" / "hkg_tmax_t24"
EXPERIMENT_ROOT = ANALYSIS_ROOT / "EXP-0002-baseline-suite"
MODELS_ROOT = PROJECT_PATHS.run_root / "models" / "baselines"
PREDICTIONS_ROOT = PROJECT_PATHS.run_root / "predictions" / "baselines"

TARGET_PATH = DATA_ROOT / "silver" / "targets" / "hko_daily_tmax.parquet"
FEATURE_PATH = DATA_ROOT / "silver" / "features" / "t24_cutoff_feature_candidates.parquet"

QUANTILES = {
    "q05": (0.05, -1.6448536269514722),
    "q10": (0.10, -1.2815515655446004),
    "q25": (0.25, -0.6744897501960817),
    "q50": (0.50, 0.0),
    "q75": (0.75, 0.6744897501960817),
    "q90": (0.90, 1.2815515655446004),
    "q95": (0.95, 1.6448536269514722),
}

SPLITS = {
    "development": (pd.Timestamp("2021-07-01"), pd.Timestamp("2023-12-31")),
    "validation_2024": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
    "locked_test": (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-05-31")),
}

PENDING_BASELINES = [
    {
        "model_id": "raw_hko_official_forecast",
        "status": "pending_source_parser",
        "reason": "HKO forecast vintages are acquired but not yet parsed into last-eligible Tmax vintage rows.",
    },
    {
        "model_id": "bias_corrected_hko_official_forecast",
        "status": "pending_source_parser",
        "reason": "Requires raw HKO official forecast baseline first.",
    },
    {
        "model_id": "raw_deterministic_nwp",
        "status": "pending_historical_vintages",
        "reason": "Current GFS subsets exist, but historical point-in-time model cycles are not yet backtestable.",
    },
    {
        "model_id": "raw_ensemble_mean_distribution",
        "status": "pending_historical_vintages",
        "reason": "GEFS historical/live cycle contract is not complete.",
    },
    {
        "model_id": "simple_mos_correction",
        "status": "pending_historical_vintages",
        "reason": "MOS correction requires historical forecast-vs-target pairs.",
    },
]


@dataclass(frozen=True)
class Prediction:
    target_date: pd.Timestamp
    split: str
    model_id: str
    point_forecast: float
    target_tmax_c: float
    training_rows: int
    method_status: str
    notes: str


def ensure_dirs() -> None:
    for path in [
        REPORTS_ROOT,
        ANALYSIS_ROOT,
        EXPERIMENT_ROOT / "predictions",
        EXPERIMENT_ROOT / "results",
        EXPERIMENT_ROOT / "artifacts",
        EXPERIMENT_ROOT / "logs",
        MODELS_ROOT,
        PREDICTIONS_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def split_for_day(day: pd.Timestamp) -> str | None:
    for name, (start, end) in SPLITS.items():
        if start <= day <= end:
            return name
    return None


def circular_doy_distance(a: int, b: int) -> int:
    raw = abs(a - b)
    return min(raw, 366 - raw)


def seasonal_training(target: pd.DataFrame, target_day: pd.Timestamp, window: int = 15) -> pd.DataFrame:
    cutoff = target_day - pd.Timedelta(days=2)
    doy = int(target_day.dayofyear)
    train = target[target["local_date"] <= cutoff].copy()
    distance = train["doy"].apply(lambda value: circular_doy_distance(int(value), doy))
    return train[distance <= window]


def seasonal_mean(target: pd.DataFrame, target_day: pd.Timestamp, window: int = 15) -> tuple[float, int]:
    train = seasonal_training(target, target_day, window=window)
    if len(train) < 30:
        train = target[target["local_date"] <= target_day - pd.Timedelta(days=2)]
    return float(train["target_tmax_c"].mean()), len(train)


def seasonal_std(target: pd.DataFrame, target_day: pd.Timestamp, window: int = 15) -> float:
    train = seasonal_training(target, target_day, window=window)
    if len(train) < 30:
        train = target[target["local_date"] <= target_day - pd.Timedelta(days=2)]
    value = float(train["target_tmax_c"].std(ddof=1))
    return value if math.isfinite(value) and value > 0 else 1.5


def trend_adjusted(target: pd.DataFrame, target_day: pd.Timestamp) -> tuple[float, int]:
    train = seasonal_training(target, target_day, window=21)
    if len(train) < 120:
        return seasonal_mean(target, target_day, window=21)
    x = (train["local_date"] - pd.Timestamp("1900-01-01")).dt.days.to_numpy(dtype=float)
    y = train["target_tmax_c"].to_numpy(dtype=float)
    coef = np.polyfit(x, y, deg=1)
    pred_x = float((target_day - pd.Timestamp("1900-01-01")).days)
    return float(coef[0] * pred_x + coef[1]), len(train)


def recent_year_climatology(target: pd.DataFrame, target_day: pd.Timestamp) -> tuple[float, int]:
    train = seasonal_training(target, target_day, window=21)
    recent = train[train["local_date"] >= target_day - pd.DateOffset(years=10)]
    if len(recent) < 30:
        recent = train.tail(120)
    return float(recent["target_tmax_c"].mean()), len(recent)


def multi_day_memory(target_by_date: pd.Series, target_day: pd.Timestamp) -> tuple[float | None, int]:
    days = [target_day - pd.Timedelta(days=offset) for offset in range(2, 9)]
    values = [float(target_by_date.get(day, np.nan)) for day in days]
    valid = [value for value in values if math.isfinite(value)]
    if len(valid) < 3:
        return None, len(valid)
    weights = np.array([0.30, 0.22, 0.16, 0.12, 0.08, 0.07, 0.05])[: len(values)]
    weighted_values = []
    weighted_weights = []
    for value, weight in zip(values, weights, strict=False):
        if math.isfinite(value):
            weighted_values.append(value)
            weighted_weights.append(weight)
    return float(np.average(weighted_values, weights=weighted_weights)), len(valid)


def anomaly_persistence(
    target: pd.DataFrame, target_day: pd.Timestamp, tminus2_tmax: float | None
) -> tuple[float | None, int]:
    if tminus2_tmax is None or not math.isfinite(tminus2_tmax):
        return None, 0
    normal_t, n_t = seasonal_mean(target, target_day, window=15)
    normal_tminus2, n_tm2 = seasonal_mean(target, target_day - pd.Timedelta(days=2), window=15)
    return float(normal_t + (tminus2_tmax - normal_tminus2)), min(n_t, n_tm2)


def station_state_analogue(features: pd.DataFrame, row: pd.Series) -> tuple[float | None, int]:
    target_day = pd.Timestamp(row["local_date"])
    temp = row.get("hko_temp_at_tminus1_1500_c")
    if pd.isna(temp):
        return None, 0
    train = features[
        (features["local_date"] <= target_day - pd.Timedelta(days=2))
        & features["target_tmax_c"].notna()
        & features["hko_temp_at_tminus1_1500_c"].notna()
    ].copy()
    if len(train) < 50:
        return None, len(train)
    doy = int(target_day.dayofyear)
    train["doy_distance"] = train["doy"].apply(lambda value: circular_doy_distance(int(value), doy))
    train = train[train["doy_distance"] <= 45].copy()
    if len(train) < 25:
        return None, len(train)
    train["distance"] = (train["doy_distance"] / 30.0) ** 2 + (
        (train["hko_temp_at_tminus1_1500_c"] - float(temp)) / 1.75
    ) ** 2
    for column, scale in [
        ("hko_rh_at_tminus1_1500_pct", 20.0),
        ("hko_mslp_at_tminus1_1500_hpa", 8.0),
    ]:
        value = row.get(column)
        if pd.notna(value) and train[column].notna().sum() >= 25:
            train["distance"] = train["distance"] + (
                (train[column].fillna(float(value)) - float(value)) / scale
            ) ** 2
    nearest = train.nsmallest(25, "distance")
    return float(nearest["target_tmax_c"].mean()), len(nearest)


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


def add_distribution(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    fallback_sigma = float(out[out["split"] == "development"]["target_tmax_c"].std(ddof=1))
    sigmas: dict[str, float] = {}
    for model_id, group in out[out["split"] == "development"].groupby("model_id"):
        residual = group["target_tmax_c"] - group["point_forecast"]
        sigma = float(residual.std(ddof=1))
        if not math.isfinite(sigma) or sigma <= 0:
            sigma = fallback_sigma
        sigmas[str(model_id)] = max(sigma, 0.2)

    out["distribution_sigma_c"] = out["model_id"].map(sigmas).fillna(fallback_sigma)
    out["distribution_calibration"] = "normal_residual_sigma_from_development_split"
    for name, (_, z_value) in QUANTILES.items():
        out[name] = out["point_forecast"] + out["distribution_sigma_c"] * z_value
    return out


def build_point_predictions(target: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    target_by_date = target.set_index("local_date")["target_tmax_c"]
    rows: list[Prediction] = []
    eval_rows = features[
        (features["local_date"] >= SPLITS["development"][0])
        & (features["local_date"] <= SPLITS["locked_test"][1])
        & features["target_tmax_c"].notna()
        & features["hko_temp_at_tminus1_1500_c"].notna()
    ].copy()
    eval_rows["split"] = eval_rows["local_date"].apply(split_for_day)
    eval_rows = eval_rows[eval_rows["split"].notna()].copy()

    for _, row in eval_rows.iterrows():
        day = pd.Timestamp(row["local_date"])
        split = str(row["split"])
        target_value = float(row["target_tmax_c"])

        seasonal, n_seasonal = seasonal_mean(target, day)
        rows.append(
            Prediction(day, split, "day_of_year_climatology", seasonal, target_value, n_seasonal, "scored", "")
        )

        trend, n_trend = trend_adjusted(target, day)
        rows.append(
            Prediction(day, split, "trend_adjusted_climatology", trend, target_value, n_trend, "scored", "")
        )

        recent, n_recent = recent_year_climatology(target, day)
        rows.append(
            Prediction(day, split, "recent10y_climatology", recent, target_value, n_recent, "scored", "")
        )

        tminus2 = row.get("hko_tminus2_official_tmax_c")
        if pd.notna(tminus2):
            rows.append(
                Prediction(
                    day,
                    split,
                    "last_final_tmax_persistence",
                    float(tminus2),
                    target_value,
                    1,
                    "scored_proxy_publication_lag_needed",
                    "Uses T-2 official Tmax; publication timing still must be empirically proven.",
                )
            )

        anomaly, n_anomaly = anomaly_persistence(
            target, day, float(tminus2) if pd.notna(tminus2) else None
        )
        if anomaly is not None:
            rows.append(
                Prediction(
                    day,
                    split,
                    "seasonal_anomaly_persistence",
                    anomaly,
                    target_value,
                    n_anomaly,
                    "scored_proxy_publication_lag_needed",
                    "Uses T-2 official Tmax anomaly relative to seasonal normal.",
                )
            )

        memory, n_memory = multi_day_memory(target_by_date, day)
        if memory is not None:
            rows.append(
                Prediction(
                    day,
                    split,
                    "multi_day_thermal_memory",
                    memory,
                    target_value,
                    n_memory,
                    "scored_proxy_publication_lag_needed",
                    "Uses official target labels through T-2 only.",
                )
            )

        cutoff_temp = row.get("hko_temp_at_tminus1_1500_c")
        if pd.notna(cutoff_temp):
            rows.append(
                Prediction(
                    day,
                    split,
                    "cutoff_station_temperature_persistence",
                    float(cutoff_temp),
                    target_value,
                    1,
                    "scored",
                    "Latest HKO temperature available by cutoff with conservative latency.",
                )
            )

        analogue, n_analogue = station_state_analogue(features, row)
        if analogue is not None:
            rows.append(
                Prediction(
                    day,
                    split,
                    "station_state_analogue",
                    analogue,
                    target_value,
                    n_analogue,
                    "scored",
                    "Nearest historical days using season plus eligible cutoff station state.",
                )
            )

    point = pd.DataFrame([item.__dict__ for item in rows])
    if point.empty:
        raise RuntimeError("No baseline predictions were produced")

    blend_inputs = [
        "trend_adjusted_climatology",
        "seasonal_anomaly_persistence",
        "multi_day_thermal_memory",
        "station_state_analogue",
    ]
    wide = point[point["model_id"].isin(blend_inputs)].pivot_table(
        index=["target_date", "split", "target_tmax_c"], columns="model_id", values="point_forecast"
    )
    blend_rows: list[dict[str, object]] = []
    for index, values in wide.iterrows():
        available = values.dropna()
        if len(available) < 2:
            continue
        target_day, split, target_value = index
        blend_rows.append(
            {
                "target_date": target_day,
                "split": split,
                "model_id": "transparent_equal_weight_blend",
                "point_forecast": float(available.mean()),
                "target_tmax_c": float(target_value),
                "training_rows": int(len(available)),
                "method_status": "scored",
                "notes": "Equal-weight blend of predeclared transparent baselines.",
            }
        )
    if blend_rows:
        point = pd.concat([point, pd.DataFrame(blend_rows)], ignore_index=True)

    return add_distribution(point.sort_values(["target_date", "model_id"]))


def score_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_id, split), group in predictions.groupby(["model_id", "split"]):
        y = group["target_tmax_c"]
        point = group["point_forecast"]
        error = point - y
        n = len(group)
        central_80 = (group["q10"] <= y) & (y <= group["q90"])
        central_90 = (group["q05"] <= y) & (y <= group["q95"])
        crps = [
            normal_crps(float(row.target_tmax_c), float(row.point_forecast), float(row.distribution_sigma_c))
            for row in group.itertuples()
        ]
        rows.append(
            {
                "model_id": model_id,
                "split": split,
                "status": "scored",
                "n": n,
                "mae": float(error.abs().mean()),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "median_abs_error": float(error.abs().median()),
                "bias": float(error.mean()),
                "crps_normal": float(np.mean(crps)),
                "pinball_mean": float(
                    np.mean([pinball(y, group[name], alpha) for name, (alpha, _) in QUANTILES.items()])
                ),
                "coverage_80": float(central_80.mean()),
                "width_80": float((group["q90"] - group["q10"]).mean()),
                "coverage_90": float(central_90.mean()),
                "width_90": float((group["q95"] - group["q05"]).mean()),
                "method_status": str(group["method_status"].iloc[0]),
            }
        )
    scoreboard = pd.DataFrame(rows)
    for pending in PENDING_BASELINES:
        for split in SPLITS:
            row = {
                "model_id": pending["model_id"],
                "split": split,
                "status": pending["status"],
                "n": 0,
                "mae": np.nan,
                "rmse": np.nan,
                "median_abs_error": np.nan,
                "bias": np.nan,
                "crps_normal": np.nan,
                "pinball_mean": np.nan,
                "coverage_80": np.nan,
                "width_80": np.nan,
                "coverage_90": np.nan,
                "width_90": np.nan,
                "method_status": pending["reason"],
            }
            scoreboard = pd.concat([scoreboard, pd.DataFrame([row])], ignore_index=True)
    return add_mae_ci(scoreboard, predictions)


def block_bootstrap_mae_ci(group: pd.DataFrame, *, iterations: int = 400, block: int = 14) -> tuple[float, float]:
    if len(group) < block * 3:
        return (float("nan"), float("nan"))
    errors = (group["point_forecast"] - group["target_tmax_c"]).abs().to_numpy()
    rng = np.random.default_rng(20260620)
    estimates: list[float] = []
    max_start = len(errors) - block
    for _ in range(iterations):
        chunks: list[np.ndarray] = []
        while sum(len(chunk) for chunk in chunks) < len(errors):
            start = int(rng.integers(0, max_start + 1))
            chunks.append(errors[start : start + block])
        sample = np.concatenate(chunks)[: len(errors)]
        estimates.append(float(sample.mean()))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def add_mae_ci(scoreboard: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    low_values: list[float] = []
    high_values: list[float] = []
    for _, row in scoreboard.iterrows():
        if row["status"] != "scored":
            low_values.append(float("nan"))
            high_values.append(float("nan"))
            continue
        group = predictions[
            (predictions["model_id"] == row["model_id"]) & (predictions["split"] == row["split"])
        ]
        low, high = block_bootstrap_mae_ci(group)
        low_values.append(low)
        high_values.append(high)
    out = scoreboard.copy()
    out["mae_ci95_low"] = low_values
    out["mae_ci95_high"] = high_values
    return out.sort_values(["split", "mae"], na_position="last")


def choose_champion(scoreboard: pd.DataFrame) -> dict[str, object]:
    candidates = scoreboard[
        (scoreboard["split"] == "validation_2024")
        & (scoreboard["status"] == "scored")
        & (scoreboard["n"] >= 300)
        & scoreboard["mae"].notna()
    ].sort_values(["mae", "rmse"])
    if candidates.empty:
        raise RuntimeError("No champion baseline candidate met validation coverage rules")
    champion = candidates.iloc[0].to_dict()
    champion["selection_rule"] = "lowest validation_2024 MAE among predeclared scored baselines with n >= 300; RMSE as tie-breaker"
    return champion


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._\n"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for _, row in df[columns].iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_reports(
    predictions: pd.DataFrame,
    scoreboard: pd.DataFrame,
    champion: dict[str, object],
    target: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    scoreboard_display = scoreboard.copy()
    numeric_cols = [
        "mae",
        "rmse",
        "median_abs_error",
        "bias",
        "crps_normal",
        "pinball_mean",
        "coverage_80",
        "width_80",
        "coverage_90",
        "width_90",
        "mae_ci95_low",
        "mae_ci95_high",
    ]
    for column in numeric_cols:
        scoreboard_display[column] = scoreboard_display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
        )
    scoreboard_display["n"] = scoreboard_display["n"].astype(int).astype(str)

    write(
        REPORTS_ROOT / "SPLIT_FREEZE.md",
        """# Split Freeze

Primary T-24 common-sample split is frozen before advanced model fitting.

| Split | Start | End | Role |
|---|---|---|---|
| Development | 2021-07-01 | 2023-12-31 | EDA and baseline design |
| Validation 2024 | 2024-01-01 | 2024-12-31 | Champion baseline selection |
| Locked test | 2025-01-01 | 2026-05-31 | Final untouched comparison for this baseline suite |

Common-sample rule: target exists, HKO high-frequency cutoff temperature exists, and the row is within the frozen dates.

The split was not altered after seeing baseline results.
""",
    )

    write(
        REPORTS_ROOT / "BASELINE_SCOREBOARD.md",
        "# Baseline Scoreboard\n\n"
        f"Champion baseline: `{champion['model_id']}` selected by `{champion['selection_rule']}`.\n\n"
        + markdown_table(
            scoreboard_display,
            [
                "model_id",
                "split",
                "status",
                "n",
                "mae",
                "rmse",
                "median_abs_error",
                "bias",
                "crps_normal",
                "pinball_mean",
                "coverage_80",
                "width_80",
                "coverage_90",
                "width_90",
                "mae_ci95_low",
                "mae_ci95_high",
                "method_status",
            ],
        )
        + "\n## Leakage Statement\n\n"
        "- Common sample begins on 2021-07-01 because pressure/modern high-frequency archives are not complete before then.\n"
        "- Target labels are used only as labels or as lagged values through T-2.\n"
        "- HKO T-1 15:00 station features are selected by `available_at <= cutoff` under a +20 minute conservative latency assumption.\n"
        "- HKO official forecast and NWP baselines are not scored until historical vintages are parsed.\n",
    )

    calibration = scoreboard_display[
        [
            "model_id",
            "split",
            "n",
            "crps_normal",
            "pinball_mean",
            "coverage_80",
            "width_80",
            "coverage_90",
            "width_90",
        ]
    ]
    write(
        REPORTS_ROOT / "BASELINE_CALIBRATION.md",
        "# Baseline Calibration\n\n"
        "All scored baselines output a normal residual distribution with sigma calibrated on the development split. "
        "This is intentionally simple and will be challenged by conformal/distributional experiments later.\n\n"
        + markdown_table(calibration, list(calibration.columns)),
    )

    champion_preds = predictions[predictions["model_id"] == champion["model_id"]].copy()
    champion_preds["abs_error"] = (champion_preds["point_forecast"] - champion_preds["target_tmax_c"]).abs()
    failures = champion_preds.sort_values("abs_error", ascending=False).head(30)
    write(
        REPORTS_ROOT / "BASELINE_FAILURE_CASES.md",
        "# Baseline Failure Cases\n\n"
        f"Champion baseline: `{champion['model_id']}`.\n\n"
        + markdown_table(
            failures[
                [
                    "target_date",
                    "split",
                    "target_tmax_c",
                    "point_forecast",
                    "abs_error",
                    "q05",
                    "q95",
                ]
            ].round(3),
            ["target_date", "split", "target_tmax_c", "point_forecast", "abs_error", "q05", "q95"],
        ),
    )

    status = f"""# HKG Tmax T-24 Status

Current completed phases in this work session:

- Phase A partial: HKO daily climate and cutoff-relevant HKO high-frequency station archive parsed.
- Phase B partial: target anatomy, cutoff station state, station-network cutoff contrast and initial hypothesis reports generated.
- Phase C complete for current common sample: split frozen at 2021-07-01/2024/2025-2026 boundaries.
- Phase D partial: available non-forecast/NWP baselines scored; official forecast and NWP baselines remain pending parsers/vintages.

Champion baseline: `{champion['model_id']}` selected on validation_2024 MAE.

No Polymarket work was performed. No advanced ML was started.
"""
    write(REPO_ROOT / "STATUS.md", status)
    write(ANALYSIS_ROOT / "STATUS.md", status)

    if not (REPO_ROOT / "MILESTONES.md").exists():
        write(
            REPO_ROOT / "MILESTONES.md",
            "# Milestones\n\nNo accepted improvement milestone is recorded yet. The champion baseline is frozen as a comparison asset, not as an improvement claim.\n",
        )
    else:
        existing = (REPO_ROOT / "MILESTONES.md").read_text(encoding="utf-8")
        if "Baseline freeze note" not in existing:
            write(
                REPO_ROOT / "MILESTONES.md",
                existing.rstrip()
                + "\n\n## Baseline freeze note\n\nThe current champion baseline is frozen for comparison, but no challenger improvement milestone is accepted yet.\n",
            )

    metadata = {
        "target_rows": len(target),
        "feature_rows": len(features),
        "prediction_rows": len(predictions),
        "scoreboard_rows": len(scoreboard),
        "champion": champion,
    }
    write(EXPERIMENT_ROOT / "artifacts" / "run_summary.json", json.dumps(metadata, indent=2, default=str))


def write_experiment_docs(
    champion: dict[str, object], scoreboard: pd.DataFrame, *, locked_test_policy: str
) -> None:
    metrics = {
        "champion": champion,
        "validation_scoreboard": scoreboard[scoreboard["split"] == "validation_2024"].to_dict(orient="records"),
        "locked_test_policy": locked_test_policy,
    }
    champion_model = str(champion["model_id"])
    validation_row = scoreboard[
        (scoreboard["split"] == "validation_2024") & (scoreboard["model_id"] == champion_model)
    ].iloc[0]
    locked_row = scoreboard[
        (scoreboard["split"] == "locked_test")
        & (scoreboard["model_id"] == champion_model)
        & (scoreboard["status"] == "scored")
    ]
    if locked_row.empty:
        locked_summary = "Locked-test result: not computed under deny policy."
    else:
        locked_result = locked_row.iloc[0]
        locked_summary = (
            f"Locked-test result: MAE `{float(locked_result['mae']):.4f} C`, "
            f"RMSE `{float(locked_result['rmse']):.4f} C`."
        )
    write(EXPERIMENT_ROOT / "metrics.json", json.dumps(metrics, indent=2, default=str))
    write(EXPERIMENT_ROOT / "results" / "metrics.json", json.dumps(metrics, indent=2, default=str))
    write(
        EXPERIMENT_ROOT / "README.md",
        f"""# EXP-0002 Baseline Suite

This folder contains the self-contained record for the first leakage-safe HKO
T-24 Tmax baseline experiment.

Main local files:

- `EXPERIMENT_REPORT_7500_CHARS.md` - plain-language long-form explanation.
- `DATE_RANGES.md` - exact data and split date ranges used.
- `LOCAL_SCOREBOARD.md` - compact local scoreboard and champion summary.
- `results/metrics.json` - machine-readable metrics inside the experiment folder.
- `predictions/hkg_tmax_baseline_predictions.parquet` - prediction rows.

Champion baseline: `{champion_model}`.

Validation 2024 result: MAE `{float(validation_row['mae']):.4f} C`, RMSE `{float(validation_row['rmse']):.4f} C`.
{locked_summary}
""",
    )
    write(
        EXPERIMENT_ROOT / "HYPOTHESIS.md",
        "# Hypothesis\n\nTransparent climatology, persistence, memory and cutoff station-state baselines establish a reproducible champion that later experiments must beat on the same split.\n",
    )
    write(
        EXPERIMENT_ROOT / "PROTOCOL.md",
        "# Protocol\n\n1. Use target dates from 2021-07-01 through 2026-05-31 with target and eligible HKO cutoff temperature present.\n2. Freeze development, validation and locked-test dates before selecting the champion.\n3. Score predeclared baselines and pending-source placeholders.\n4. Select champion by validation_2024 MAE with RMSE tie-breaker.\n5. Do not tune on locked-test outcomes.\n",
    )
    write(
        EXPERIMENT_ROOT / "ASOF_CONTRACT.md",
        "# As-Of Contract\n\nPrimary cutoff is T-1 15:00 HKT. HKO station features are selected by `available_at <= cutoff` using a conservative +20 minute publication latency. Official target labels may only enter as target values or lagged values through T-2.\n",
    )
    write(
        EXPERIMENT_ROOT / "DATA_MANIFEST.yaml",
        f"""target_table: {TARGET_PATH}
feature_table: {FEATURE_PATH}
prediction_table: {PREDICTIONS_ROOT / "hkg_tmax_baseline_predictions.parquet"}
scoreboard_table: {PREDICTIONS_ROOT / "hkg_tmax_baseline_scoreboard.parquet"}
data_root: {DATA_ROOT}
""",
    )
    write(
        EXPERIMENT_ROOT / "RUN_CONFIG.yaml",
        f"""cutoff: T-1 15:00 HKT
development: 2021-07-01/2023-12-31
validation: 2024-01-01/2024-12-31
locked_test: {"2025-01-01/2026-05-31" if locked_test_policy == "allow" else "denied_not_scored"}
locked_test_policy: {locked_test_policy}
selection_metric: validation_2024_mae
distribution: normal_residual_sigma_from_development_split
""",
    )
    write(
        EXPERIMENT_ROOT / "RESULTS.md",
        "# Results\n\nSee `reports/BASELINE_SCOREBOARD.md`, `reports/BASELINE_CALIBRATION.md`, and `reports/BASELINE_FAILURE_CASES.md`.\n",
    )
    write(
        EXPERIMENT_ROOT / "CONCLUSION.md",
        f"# Conclusion\n\nChampion baseline frozen: `{champion['model_id']}`. This is a comparison baseline, not an improvement milestone.\n",
    )
    write(
        EXPERIMENT_ROOT / "REPRODUCE.md",
        "# Reproduce\n\nRun:\n\n```powershell\n.\\.venv\\Scripts\\python.exe scripts\\build_hkg_tmax_phase_ab_analysis.py\n.\\.venv\\Scripts\\python.exe scripts\\run_hkg_tmax_baselines.py\n```\n",
    )
    write(
        EXPERIMENT_ROOT / "STATUS.yaml",
        "status: complete\nphase: D_partial\npolymarket: not_performed\nadvanced_ml: not_started\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HKG Tmax T-24 transparent baselines.")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument(
        "--locked-test-policy",
        choices=("deny", "allow"),
        default="deny",
        help="Default deny prevents ordinary research runs from scoring 2025+ locked-test dates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    target_path = data_root / "silver" / "targets" / "hko_daily_tmax.parquet"
    feature_path = data_root / "silver" / "features" / "t24_cutoff_feature_candidates.parquet"
    ensure_dirs()
    target = pd.read_parquet(target_path)
    features = pd.read_parquet(feature_path)
    target["local_date"] = pd.to_datetime(target["local_date"])
    features["local_date"] = pd.to_datetime(features["local_date"])
    if args.locked_test_policy == "deny":
        target = target[target["local_date"].dt.date < LOCKED_TEST_START].copy()
        features = features[features["local_date"].dt.date < LOCKED_TEST_START].copy()
        assert_no_locked_dates(features["local_date"], context="baseline command")
    target["doy"] = target["local_date"].dt.dayofyear
    features["doy"] = features["local_date"].dt.dayofyear

    predictions = build_point_predictions(target, features)
    scoreboard = score_predictions(predictions)
    champion = choose_champion(scoreboard)

    predictions.to_parquet(PREDICTIONS_ROOT / "hkg_tmax_baseline_predictions.parquet", index=False)
    predictions.to_parquet(EXPERIMENT_ROOT / "predictions" / "hkg_tmax_baseline_predictions.parquet", index=False)
    scoreboard.to_parquet(PREDICTIONS_ROOT / "hkg_tmax_baseline_scoreboard.parquet", index=False)
    write(MODELS_ROOT / "champion_baseline.json", json.dumps(champion, indent=2, default=str))
    write_reports(predictions, scoreboard, champion, target, features)
    write_experiment_docs(champion, scoreboard, locked_test_policy=str(args.locked_test_policy))
    print(json.dumps({"champion": champion, "predictions": len(predictions)}, indent=2, default=str))


if __name__ == "__main__":
    main()
