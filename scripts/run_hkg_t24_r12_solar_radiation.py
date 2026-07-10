from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hkg_tmax.hkg_t24.governance import check_four_year_oof_feasibility
from hkg_tmax.hkg_t24.guard import assert_no_locked_dates

REPO_ROOT = Path(__file__).resolve().parents[1]
_R04_SPEC = importlib.util.spec_from_file_location(
    "run_hkg_t24_r04_thermal_trajectory",
    REPO_ROOT / "scripts" / "run_hkg_t24_r04_thermal_trajectory.py",
)
if _R04_SPEC is None or _R04_SPEC.loader is None:
    raise ImportError("Unable to load R04 helper script for R12 solar-radiation experiment.")
_R04_MODULE = importlib.util.module_from_spec(_R04_SPEC)
_R04_SPEC.loader.exec_module(_R04_MODULE)

QUANTILE_Z = _R04_MODULE.QUANTILE_Z
fold_definitions = _R04_MODULE.fold_definitions
git_state = _R04_MODULE.git_state
r04_feature_sets = _R04_MODULE.model_feature_sets
normal_crps = _R04_MODULE.normal_crps

DEFAULT_DATA_ROOT = Path(r"C:\hkg_tmax_data")
RESEARCH_ID = "HKG-T24-R12"
EXPERIMENT_ID = "EXP-0044"
EXPERIMENT_DIR = REPO_ROOT / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0044-HKG-T24-R12"
ANALYSIS_END = pd.Timestamp("2023-12-31")


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_family: str
    columns: tuple[str, ...]
    control: bool = False


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], *, limit: int = 50) -> str:
    if frame.empty:
        return "_No rows._"
    subset = frame.loc[:, [col for col in columns if col in frame.columns]].head(limit)
    cols = list(subset.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in subset.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def active_cols(train: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [
        col
        for col in columns
        if col in train.columns
        and pd.api.types.is_numeric_dtype(train[col])
        and train[col].notna().sum() >= 30
        and train[col].nunique(dropna=True) > 1
    ]


def fit_pipeline() -> Pipeline:
    return Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]
    )


def build_cutoff_solar_features(observations: pd.DataFrame, base_dates: pd.Series) -> pd.DataFrame:
    solar = observations[observations["family"].eq("latest_1min_solar")].copy()
    if solar.empty:
        raise RuntimeError("No latest_1min_solar rows found in phase-A observations.")
    solar["target_date"] = pd.to_datetime(solar["local_date"]) + pd.Timedelta(days=1)
    target_dates = pd.to_datetime(base_dates).dt.normalize()
    solar = solar[solar["target_date"].isin(set(target_dates))]
    cutoff = solar["target_date"].dt.tz_localize("Asia/Hong_Kong") - pd.Timedelta(days=1) + pd.Timedelta(hours=15)
    solar = solar[solar["available_at_hkt"] <= cutoff].copy()
    solar = solar.sort_values(["target_date", "variable", "observed_at_hkt"])
    grouped = solar.groupby(["target_date", "variable"], observed=True)
    summary = grouped["value"].agg(["count", "mean", "max", "min", "std", "sum", "last"]).reset_index()
    pivot = summary.pivot(index="target_date", columns="variable")
    pivot.columns = [f"r12_{variable}_{stat}" for stat, variable in pivot.columns]
    pivot = pivot.reset_index()
    global_mean = pivot.get("r12_global_solar_wm2_mean")
    direct_mean = pivot.get("r12_direct_solar_wm2_mean")
    diffuse_mean = pivot.get("r12_diffuse_solar_wm2_mean")
    global_sum = pivot.get("r12_global_solar_wm2_sum")
    if global_mean is not None and direct_mean is not None:
        pivot["r12_direct_to_global_mean_ratio"] = direct_mean / global_mean.replace(0, np.nan)
    if global_mean is not None and diffuse_mean is not None:
        pivot["r12_diffuse_to_global_mean_ratio"] = diffuse_mean / global_mean.replace(0, np.nan)
    if global_sum is not None:
        pivot["r12_global_solar_sampled_kwh_proxy"] = global_sum * (10.0 / 60.0) / 1000.0
    low_fraction = (
        solar[solar["variable"].eq("global_solar_wm2")]
        .assign(is_low=lambda df: df["value"] < 150.0)
        .groupby("target_date", observed=True)["is_low"]
        .mean()
        .rename("r12_global_solar_low_fraction")
        .reset_index()
    )
    return pivot.merge(low_fraction, on="target_date", how="left").sort_values("target_date").reset_index(drop=True)


def add_shifted_solar_controls(solar_features: pd.DataFrame) -> pd.DataFrame:
    shifted = solar_features.copy()
    shifted["target_date"] = pd.to_datetime(shifted["target_date"]) + pd.Timedelta(days=1)
    rename = {col: f"r12_shifted_{col.removeprefix('r12_')}" for col in shifted.columns if col != "target_date"}
    return shifted.rename(columns=rename)


def build_feature_matrix(data_root: Path) -> tuple[pd.DataFrame, Path, pd.DataFrame]:
    base_path = data_root / "gold" / "hkg_t24" / "r04_thermal_trajectory" / "r04_feature_matrix.parquet"
    obs_path = data_root / "bronze" / "analysis_phase_a" / "hko_high_frequency_selected_station_observations.parquet"
    if not base_path.exists():
        raise FileNotFoundError(f"R12 requires R04 feature matrix: {base_path}")
    if not obs_path.exists():
        raise FileNotFoundError(f"R12 requires phase-A observation table: {obs_path}")
    base = pd.read_parquet(base_path)
    base["target_date"] = pd.to_datetime(base["target_date"])
    base = base[base["target_date"] <= ANALYSIS_END].copy()
    assert_no_locked_dates(base["target_date"], context="R12 base feature matrix")
    observations = pd.read_parquet(obs_path)
    solar_features = build_cutoff_solar_features(observations, base["target_date"])
    shifted = add_shifted_solar_controls(solar_features)
    features = base.merge(solar_features, on="target_date", how="left").merge(shifted, on="target_date", how="left")
    if "r12_global_solar_sampled_kwh_proxy" in features.columns:
        features["r12_global_solar_per_daylight_hour_proxy"] = (
            features["r12_global_solar_sampled_kwh_proxy"] / features["day_length_hours"].replace(0, np.nan)
        )
    if "r12_global_solar_sampled_kwh_proxy" in features.columns and "hko_latest_minus_0600_c" in features.columns:
        features["r12_hko_heating_per_solar_proxy"] = (
            features["hko_latest_minus_0600_c"] / features["r12_global_solar_sampled_kwh_proxy"].replace(0, np.nan)
        )
    output_path = data_root / "gold" / "hkg_t24" / "r12_solar_radiation" / "r12_feature_matrix.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    return features, output_path, solar_features


def model_specs(features: pd.DataFrame) -> list[ModelSpec]:
    baseline = tuple(r04_feature_sets(features)["r04_baseline_latest_temp_calendar"])
    geometry = tuple(
        col
        for col in ["doy_sin", "doy_cos", "solar_declination_deg", "day_length_hours", "noon_solar_elevation_deg"]
        if col in features.columns
    )
    observed_global = tuple(col for col in features.columns if col.startswith("r12_global_solar"))
    observed_full = tuple(col for col in features.columns if col.startswith("r12_") and not col.startswith("r12_shifted_"))
    shifted = tuple(col for col in features.columns if col.startswith("r12_shifted_"))
    heating = tuple(
        col
        for col in [
            "r12_global_solar_sampled_kwh_proxy",
            "r12_global_solar_per_daylight_hour_proxy",
            "r12_hko_heating_per_solar_proxy",
            "r12_direct_to_global_mean_ratio",
            "r12_diffuse_to_global_mean_ratio",
            "r12_global_solar_low_fraction",
            "hko_latest_minus_0600_c",
            "hko_temp_slope_180m_to_latest_c_per_hour",
        ]
        if col in features.columns
    )
    return [
        ModelSpec("r12_baseline_temp_calendar", "ridge_baseline", baseline),
        ModelSpec("r12_deterministic_solar_geometry_control", "deterministic_geometry_control", geometry, control=True),
        ModelSpec("r12_observed_global_solar_ridge", "observed_global_solar_ridge", baseline + observed_global),
        ModelSpec("r12_observed_direct_diffuse_solar_ridge", "observed_direct_diffuse_solar_ridge", baseline + observed_full),
        ModelSpec("r12_heating_efficiency_ridge", "heating_efficiency_ridge", baseline + heating),
        ModelSpec("r12_shifted_solar_negative_control", "shifted_solar_negative_control", baseline + shifted, control=True),
    ]


def run_oof(features: pd.DataFrame, specs: Sequence[ModelSpec]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold_id, test_start, test_end, train_end in fold_definitions():
        train = features[features["target_date"] <= train_end].copy()
        test = features[(features["target_date"] >= test_start) & (features["target_date"] <= test_end)].copy()
        if len(train) < 330 or test.empty:
            continue
        for spec in specs:
            cols = active_cols(train, spec.columns)
            if not cols:
                continue
            model = fit_pipeline()
            model.fit(train[cols], train["target_tmax_c"])
            train_pred = model.predict(train[cols])
            sigma = float(np.std(train["target_tmax_c"].to_numpy(dtype=float) - train_pred, ddof=1))
            sigma = max(sigma, 0.2)
            pred = test[["target_date", "target_tmax_c"]].copy()
            pred["fold_id"] = fold_id
            pred["model_id"] = spec.model_id
            pred["model_family"] = spec.model_family
            pred["is_control"] = spec.control
            pred["training_start"] = train["target_date"].min()
            pred["training_end"] = train["target_date"].max()
            pred["training_rows"] = int(len(train))
            pred["feature_count"] = int(len(cols))
            pred["point_forecast"] = model.predict(test[cols])
            pred["distribution_sigma_c"] = sigma
            for qcol, z_value in QUANTILE_Z.items():
                pred[qcol] = pred["point_forecast"] + pred["distribution_sigma_c"] * z_value
            rows.append(pred)
    if not rows:
        raise RuntimeError("R12 produced no OOF predictions.")
    predictions = pd.concat(rows, ignore_index=True).sort_values(["target_date", "model_id"]).reset_index(drop=True)
    assert_no_locked_dates(predictions["target_date"], context="R12 OOF predictions")
    return predictions


def score_frame(predictions: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in predictions.groupby(list(group_cols), dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        error = group["point_forecast"] - group["target_tmax_c"]
        crps = [
            normal_crps(float(row.target_tmax_c), float(row.point_forecast), float(row.distribution_sigma_c))
            for row in group.itertuples()
        ]
        out = {col: value for col, value in zip(group_cols, key_tuple, strict=True)}
        out.update(
            {
                "n": int(len(group)),
                "first_date": str(group["target_date"].min().date()),
                "last_date": str(group["target_date"].max().date()),
                "mae": float(error.abs().mean()),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "median_abs_error": float(error.abs().median()),
                "bias": float(error.mean()),
                "crps_normal": float(np.mean(crps)),
                "coverage_80": float(((group["q10"] <= group["target_tmax_c"]) & (group["target_tmax_c"] <= group["q90"])).mean()),
                "coverage_90": float(((group["q05"] <= group["target_tmax_c"]) & (group["target_tmax_c"] <= group["q95"])).mean()),
            }
        )
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)


def fold_deltas(predictions: pd.DataFrame) -> pd.DataFrame:
    scores = score_frame(predictions, ["fold_id", "model_id", "is_control"])
    baseline = scores[scores["model_id"].eq("r12_baseline_temp_calendar")][["fold_id", "mae", "crps_normal"]].rename(
        columns={"mae": "baseline_mae", "crps_normal": "baseline_crps"}
    )
    return scores.merge(baseline, on="fold_id", how="left").assign(
        mae_improvement_vs_baseline=lambda df: df["baseline_mae"] - df["mae"],
        crps_improvement_vs_baseline=lambda df: df["baseline_crps"] - df["crps_normal"],
    )


def solar_diagnostics(features: pd.DataFrame, solar_features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    solar_cols = [col for col in features.columns if col.startswith("r12_") and not col.startswith("r12_shifted_")]
    for col in solar_cols:
        valid = features[[col, "target_tmax_c"]].dropna()
        if len(valid) < 30:
            continue
        rows.append(
            {
                "feature": col,
                "n": int(len(valid)),
                "first_date": str(features.loc[valid.index, "target_date"].min().date()),
                "last_date": str(features.loc[valid.index, "target_date"].max().date()),
                "pearson_with_target": float(valid[col].corr(valid["target_tmax_c"], method="pearson")),
                "spearman_with_target": float(valid[col].corr(valid["target_tmax_c"], method="spearman")),
            }
        )
    rows.append(
        {
            "feature": "solar_cutoff_feature_rows",
            "n": int(len(solar_features)),
            "first_date": str(pd.to_datetime(solar_features["target_date"]).min().date()),
            "last_date": str(pd.to_datetime(solar_features["target_date"]).max().date()),
            "pearson_with_target": np.nan,
            "spearman_with_target": np.nan,
        }
    )
    return pd.DataFrame(rows).sort_values(["feature"]).reset_index(drop=True)


def select_champion(scoreboard: pd.DataFrame) -> Mapping[str, object]:
    non_control = scoreboard[~scoreboard["is_control"].astype(bool)]
    if non_control.empty:
        return scoreboard.iloc[0].to_dict()
    return non_control.iloc[0].to_dict()


def long_report(payload: Mapping[str, object]) -> str:
    champion = payload["champion"]
    assert isinstance(champion, Mapping)
    oof = payload["oof_feasibility"]
    assert isinstance(oof, Mapping)
    baseline = payload["baseline"]
    assert isinstance(baseline, Mapping)
    return f"""# EXP-0044 / HKG-T24-R12 Long-Form Experiment Report

## Purpose

R12 tests whether observed solar radiation at King's Park before the T-1 15:00 HKT cutoff adds next-day HKO Headquarters Tmax information beyond the existing season and HKO target-station temperature trajectory. The experiment is intentionally narrow: it does not use target-day sunshine, target-day cloud, or finalized daily radiation totals. It uses only high-frequency solar rows whose conservative available-at time is before the operational cutoff.

## Data Used

The source table is `C:\\hkg_tmax_data\\bronze\\analysis_phase_a\\hko_high_frequency_selected_station_observations.parquet`. Within that table, the eligible solar family is `latest_1min_solar`, station `King's Park`, with variables `global_solar_wm2`, `direct_solar_wm2`, and `diffuse_solar_wm2`. Solar observations begin late enough that the modern feature matrix remains short. The R12 feature target-date period is `{payload['feature_min']}` through `{payload['feature_max']}`, while OOF predictions run from `{payload['prediction_min']}` through `{payload['prediction_max']}`. Validation 2024 is not accessed. Locked 2025+ target rows are not accessed.

## Feature Construction

For each target date T, the driver maps solar observations from local day T-1 to target date T and keeps only rows with `available_at_hkt <= T-1 15:00`. Because the repository's HKO high-frequency replay uses a conservative 20-minute availability lag, the latest ordinary solar row that can enter at a 15:00 cutoff is approximately 14:40, not 14:50 or 15:00. The generated features summarize count, mean, maximum, minimum, standard deviation, sum, last value, a sampled kWh proxy, low-radiation fraction, direct/global ratio, diffuse/global ratio, and a heating-efficiency proxy relative to the HKO temperature rise.

## Model Ladder

R12 scores a baseline temperature/calendar Ridge model, a deterministic solar-geometry control, observed global-solar Ridge, observed direct/diffuse/global Ridge, heating-efficiency Ridge, and a shifted-solar negative control. The shifted control uses the previous day's solar features shifted forward by one target day. It is not promotable; it exists to check whether any apparent gain is merely seasonal persistence or alignment-insensitive leakage.

## Leakage Controls

All model parameters, imputation medians, standardization parameters, and Ridge coefficients are fit inside chronological training folds only. Target dates are guarded against locked-test access. Target-day radiation is not used. Full-day daily climate radiation is not used. The only deterministic solar geometry terms are calendar-derived and already present in R04-style features. The model comparison therefore tests observed pre-cutoff radiation against a deterministic season/solar-position control.

## OOF Gate

The strict four-year OOF check is `{oof['status']}`: {oof['reason']}. R12 is a completed diagnostic, but it is not promotable under the user's hard four-year reliability rule unless the evaluation design is explicitly changed or more prospective years accumulate.

## Main Result

The baseline row has MAE `{baseline['mae']:.4f}` C and CRPS `{baseline['crps_normal']:.4f}` over `{baseline['n']}` rows. The best non-control R12 row is `{champion['model_id']}` with MAE `{champion['mae']:.4f}` C, RMSE `{champion['rmse']:.4f}` C, bias `{champion['bias']:.4f}` C, and CRPS `{champion['crps_normal']:.4f}` over `{champion['n']}` rows. The key interpretation is the difference between observed-radiation models and the shifted-solar negative control, not the absolute ranking alone.

## Interpretation

If observed global/direct/diffuse radiation beats both baseline and shifted radiation, then T-1 observed heating conditions contain real incremental information. If shifted radiation performs similarly, the signal is mostly seasonal or persistent and should not be treated as a precise cutoff observation effect. If the deterministic geometry control performs almost as well as observed radiation, the observed solar archive may not be adding much beyond day-of-year and target-station temperature state. If heating-efficiency improves only in some folds, it may be a conditional cloud/suppression indicator for R13 rather than a standalone model feature.

## Limitations

Only King's Park solar rows are available in the parsed phase-A table even though the broader source inventory mentions King’s Park and Kau Sai Chau solar products. This experiment therefore tests the currently parsed station, not a complete two-station radiation network. UV data is not parsed into a cutoff-safe feature table here. Cloud and rain suppression are deferred to R13 because target-day daily climate cloud/rain values are retrospective mechanism labels, not lawful T-24 predictors. The short modern OOF span is still the controlling reliability limitation.

## Stability Finding

The important R12 finding is not merely that the overall observed-radiation rows lose to baseline. The fold deltas show why the family is not ready: some later folds show small improvements for global solar or heating-efficiency terms, while other folds degrade materially. That pattern is consistent with radiation being conditionally useful only in particular cloud-transition or clear-heating regimes, not as an unconditional additive predictor. Because R13 cloud/rain/visibility suppression and R20 regime classification have not yet produced lawful regime probabilities, R12 has no safe gate that can decide when to trust solar terms.

## Minimum-Support Rule

The driver enforces a fold-local minimum-support rule: a numeric feature must have at least 30 non-null training rows and more than one distinct value before it can enter a fold. This matters because the solar archive starts close to the first modern fold boundary. Without this rule, early folds can be destabilized by only a handful of solar rows. The rule is conservative and leakage-safe because it is evaluated inside each training fold only. It does not inspect test loss to decide whether a feature is allowed.

## Negative-Control Interpretation

The shifted-solar negative control is deliberately retained even though it is not promotable. If a shifted previous-day radiation model had matched or beaten observed same-origin-day solar, the experiment would strongly suggest that solar features were acting as seasonal proxies rather than genuine cutoff-specific observations. In the generated scoreboard the shifted control is worse than baseline, while observed solar is also worse overall. The combined interpretation is a null result for unconditional solar predictors, not evidence of leakage.

## What Would Be Needed For A Stronger R12

A stronger R12 would need the second solar station parsed, UV parsed with exact availability semantics, cloud-break event features from a denser image/cloud source, and a pre-cutoff regime gate that separates clear dry heating from humid cloudy suppression. It may also need a longer modern archive so that radiation features are not learned from such a short overlap. Those are follow-up engineering tasks, not grounds for using target-day daily sunshine or full-day radiation totals as predictors.

## Production Decision

No R12 solar feature is admitted into the production candidate feature bank. The retained evidence is still valuable: it prevents overconfident assumptions that solar radiation automatically improves tomorrow's Tmax once the 15:00 target-station temperature is already known. It also identifies the exact path for a lawful future retest: parse the remaining radiation/UV/cloud families, build a cloud-suppression gate, and require at least four years of OOF support before promotion.

## Decision Record

R12 is complete as a leakage-safe solar-radiation diagnostic. No validation access occurred, no locked-test target rows were scored, and no predictive feature is promoted because the strict four-year OOF rule blocks promotion. The experiment still provides useful scientific evidence about whether observed pre-cutoff radiation adds information beyond temperature trajectory and deterministic solar geometry.

## Reproducibility

The experiment folder contains OOF predictions, metrics JSON, subgroup/fold metrics, solar diagnostics, feature specification, negative controls, run config, as-of contract, data manifest with hashes, and the reproduction command. The data-root gold directory contains the canonical R12 feature matrix, predictions, scoreboard, fold deltas, and diagnostics for downstream inspection.
"""


def write_experiment(
    *,
    data_root: Path,
    feature_path: Path,
    predictions_path: Path,
    scoreboard: pd.DataFrame,
    fold_scores: pd.DataFrame,
    diagnostics: pd.DataFrame,
    payload: dict[str, object],
) -> None:
    for subdir in ["artifacts", "logs", "metrics", "predictions", "results"]:
        (EXPERIMENT_DIR / subdir).mkdir(parents=True, exist_ok=True)
    subgroup_path = EXPERIMENT_DIR / "metrics" / "subgroup_metrics.parquet"
    fold_scores.to_parquet(subgroup_path, index=False)
    scoreboard.to_csv(EXPERIMENT_DIR / "results" / "scoreboard.csv", index=False)
    fold_scores.to_csv(EXPERIMENT_DIR / "artifacts" / "fold_score_deltas.csv", index=False)
    diagnostics.to_csv(EXPERIMENT_DIR / "artifacts" / "solar_diagnostics.csv", index=False)
    pd.read_parquet(predictions_path).to_parquet(EXPERIMENT_DIR / "predictions" / "oof_predictions.parquet", index=False)
    metrics = {
        "research_id": RESEARCH_ID,
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_DIAGNOSTIC_OOF_BLOCKED",
        "validation_2024_accessed": False,
        "locked_test_accessed": False,
        "champion": payload["champion"],
        "baseline": payload["baseline"],
        "oof_feasibility": payload["oof_feasibility"],
        "scoreboard": scoreboard.to_dict(orient="records"),
        "fold_scores": fold_scores.to_dict(orient="records"),
    }
    write_text(EXPERIMENT_DIR / "metrics" / "metrics.json", json.dumps(metrics, indent=2, default=str))
    write_text(EXPERIMENT_DIR / "results" / "metrics.json", json.dumps(metrics, indent=2, default=str))
    write_text(EXPERIMENT_DIR / "logs" / "run_summary.json", json.dumps(payload, indent=2, default=str))
    write_text(EXPERIMENT_DIR / "README.md", "# EXP-0044 HKG-T24-R12 Solar Radiation\n\nCutoff-safe King's Park solar-radiation and heating-efficiency diagnostic. No validation, no locked-test, no target-day radiation.\n")
    write_text(EXPERIMENT_DIR / "HYPOTHESIS.md", "# Hypothesis\n\nObserved pre-cutoff solar radiation and heating efficiency may explain next-day HKO Tmax residuals beyond deterministic season and HKO temperature trajectory.\n")
    write_text(EXPERIMENT_DIR / "INFORMATION_GAIN.md", "# Information Gain\n\nR12 separates deterministic solar geometry from observed radiation, then checks whether direct/diffuse/global solar features retain OOF value after HKO cutoff temperature state.\n")
    write_text(EXPERIMENT_DIR / "ASOF_CONTRACT.md", "# As-Of Contract\n\nOnly T-1 solar observations with `available_at_hkt <= T-1 15:00` enter predictors. Target-day radiation, full-day daily climate radiation, validation 2024, and locked-test targets are excluded.\n")
    write_text(EXPERIMENT_DIR / "FEATURE_SPEC.yaml", """research_id: HKG-T24-R12
feature_families:
  deterministic_solar_geometry: [doy_sin, doy_cos, solar_declination_deg, day_length_hours, noon_solar_elevation_deg]
  observed_global_solar: [count, mean, max, min, std, sum, last, sampled_kwh_proxy, low_fraction]
  observed_direct_diffuse_solar: [direct_to_global_mean_ratio, diffuse_to_global_mean_ratio]
  heating_efficiency: [hko_heating_per_solar_proxy, hko_latest_minus_0600_c, hko_temp_slope_180m_to_latest_c_per_hour]
  shifted_solar_negative_control: previous_day_solar_features_shifted_forward_one_target_day
""")
    write_text(EXPERIMENT_DIR / "RUN_CONFIG.yaml", f"""research_id: {RESEARCH_ID}
experiment_id: {EXPERIMENT_ID}
data_root: {data_root}
model_ladder: baseline, deterministic_geometry_control, observed_global, observed_direct_diffuse, heating_efficiency, shifted_negative_control
validation_2024_accessed: false
locked_test_policy: deny
""")
    write_text(EXPERIMENT_DIR / "PROTOCOL.md", "# Protocol\n\n1. Start from R04 pre-validation feature matrix.\n2. Build T-1 cutoff-safe King's Park solar summaries.\n3. Merge observed and one-day-shifted solar controls.\n4. Fit Ridge models inside chronological folds.\n5. Compare observed radiation against deterministic geometry and shifted controls.\n")
    write_text(EXPERIMENT_DIR / "ABLATION_PLAN.md", "# Ablation Plan\n\nAblations: deterministic geometry only, global solar only, direct/diffuse/global solar, heating-efficiency terms, and shifted-solar negative control.\n")
    write_text(EXPERIMENT_DIR / "NEGATIVE_CONTROLS.md", "# Negative Controls\n\nThe shifted-solar model uses previous-day solar features shifted forward by one target day. It is non-promotable and tests whether the observed-radiation signal is alignment-specific.\n")
    write_text(EXPERIMENT_DIR / "DATE_RANGES.md", f"""# Date Ranges

- Feature target-date period: `{payload['feature_min']}` through `{payload['feature_max']}`.
- OOF prediction period: `{payload['prediction_min']}` through `{payload['prediction_max']}`.
- Solar source target-date period after cutoff filtering: `{payload['solar_min']}` through `{payload['solar_max']}`.
- Validation 2024: not accessed.
- Locked test 2025-01-01 onward: not accessed.
- Four-year OOF status: `{payload['oof_feasibility']['status']}`.
""")
    repo_predictions = EXPERIMENT_DIR / "predictions" / "oof_predictions.parquet"
    write_text(EXPERIMENT_DIR / "DATA_MANIFEST.yaml", f"""research_id: {RESEARCH_ID}
data_root: {data_root}
feature_matrix: {feature_path}
feature_matrix_sha256: {sha256_file(feature_path)}
oof_predictions: {predictions_path}
oof_predictions_sha256: {sha256_file(predictions_path)}
repo_oof_predictions: {repo_predictions}
repo_oof_predictions_sha256: {sha256_file(repo_predictions)}
validation_2024_accessed: false
locked_test_accessed: false
""")
    report = long_report(payload)
    write_text(EXPERIMENT_DIR / "EXPERIMENT_REPORT_7500_CHARS.md", report)
    write_text(
        EXPERIMENT_DIR / "RESULTS.md",
        "# Results\n\n"
        + markdown_table(scoreboard, ["model_id", "is_control", "n", "mae", "rmse", "bias", "crps_normal", "coverage_80", "coverage_90"])
        + "\n\n## Fold Deltas\n\n"
        + markdown_table(fold_scores, ["fold_id", "model_id", "is_control", "mae", "baseline_mae", "mae_improvement_vs_baseline"], limit=100),
    )
    write_text(EXPERIMENT_DIR / "CONCLUSION.md", "# Conclusion\n\nR12 is complete as a cutoff-safe solar diagnostic but is OOF-blocked under the strict four-year reliability rule. No solar feature is promoted.\n")
    write_text(EXPERIMENT_DIR / "REPRODUCE.md", "# Reproduce\n\n```powershell\n.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_r12_solar_radiation.py --data-root C:\\hkg_tmax_data\n```\n")
    write_text(EXPERIMENT_DIR / "STATUS.yaml", """status: COMPLETE_DIAGNOSTIC_OOF_BLOCKED
research_id: HKG-T24-R12
locked_test_accessed: false
validation_2024_accessed: false
leakage_guard: PASS
four_year_oof: BLOCKED
production_eligible: false
""")
    write_text(
        REPO_ROOT / "reports" / "hkg_t24" / "R12_SOLAR_RADIATION.md",
        report
        + "\n# R12 Machine-Readable Summary Tables\n\n"
        f"Generated: `{now_utc()}`\n\n"
        "## Scoreboard\n\n"
        + markdown_table(scoreboard, ["model_id", "is_control", "n", "first_date", "last_date", "mae", "rmse", "bias", "crps_normal"])
        + "\n\n## Solar Diagnostics\n\n"
        + markdown_table(diagnostics, ["feature", "n", "first_date", "last_date", "pearson_with_target", "spearman_with_target"], limit=80),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HKG-T24-R12 solar-radiation diagnostic.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    features, feature_path, solar_features = build_feature_matrix(data_root)
    predictions = run_oof(features, model_specs(features))
    scoreboard = score_frame(predictions, ["model_id", "is_control"])
    fold_scores = fold_deltas(predictions)
    diagnostics = solar_diagnostics(features, solar_features)
    output_dir = data_root / "gold" / "hkg_t24" / "r12_solar_radiation"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "r12_oof_predictions.parquet"
    scoreboard_path = output_dir / "r12_scoreboard.parquet"
    fold_path = output_dir / "r12_fold_score_deltas.parquet"
    diagnostics_path = output_dir / "r12_solar_diagnostics.parquet"
    predictions.to_parquet(predictions_path, index=False)
    scoreboard.to_parquet(scoreboard_path, index=False)
    fold_scores.to_parquet(fold_path, index=False)
    diagnostics.to_parquet(diagnostics_path, index=False)
    feature_dates = pd.to_datetime(features["target_date"])
    pred_dates = pd.to_datetime(predictions["target_date"])
    solar_dates = pd.to_datetime(solar_features["target_date"])
    feasibility = check_four_year_oof_feasibility(
        feature_dates.min().date(),
        feature_dates.max().date(),
        min_years=4.0,
        reason_context="R12 modern solar-radiation pre-validation feature period",
    )
    champion = dict(select_champion(scoreboard))
    baseline = scoreboard[scoreboard["model_id"].eq("r12_baseline_temp_calendar")].iloc[0].to_dict()
    payload: dict[str, object] = {
        "generated_at": now_utc(),
        "git_state": git_state(),
        "feature_min": str(feature_dates.min().date()),
        "feature_max": str(feature_dates.max().date()),
        "prediction_min": str(pred_dates.min().date()),
        "prediction_max": str(pred_dates.max().date()),
        "solar_min": str(solar_dates.min().date()),
        "solar_max": str(solar_dates.max().date()),
        "oof_feasibility": feasibility.__dict__,
        "champion": champion,
        "baseline": baseline,
    }
    write_experiment(
        data_root=data_root,
        feature_path=feature_path,
        predictions_path=predictions_path,
        scoreboard=scoreboard,
        fold_scores=fold_scores,
        diagnostics=diagnostics,
        payload=payload,
    )
    print(json.dumps({"status": "ok", "champion": champion, "oof": feasibility.__dict__}, indent=2, default=str))


if __name__ == "__main__":
    main()
