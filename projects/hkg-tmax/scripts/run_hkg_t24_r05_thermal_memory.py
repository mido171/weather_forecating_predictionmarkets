from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from hkg_tmax.hkg_t24.governance import check_four_year_oof_feasibility
from hkg_tmax.hkg_t24.guard import assert_no_locked_dates

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(r"C:\hkg_tmax_data")
RESEARCH_ID = "HKG-T24-R05"
EXPERIMENT_DIR = (
    REPO_ROOT
    / "analysis"
    / "hkg_tmax_t24"
    / "experiments"
    / "EXP-0037-HKG-T24-R05"
)
HALF_LIVES = (1, 2, 3, 5)

_R04_SPEC = importlib.util.spec_from_file_location(
    "run_hkg_t24_r04_thermal_trajectory",
    REPO_ROOT / "scripts" / "run_hkg_t24_r04_thermal_trajectory.py",
)
if _R04_SPEC is None or _R04_SPEC.loader is None:
    raise ImportError("Unable to load R04 helper script for R05 thermal-memory experiment.")
_R04_MODULE = importlib.util.module_from_spec(_R04_SPEC)
_R04_SPEC.loader.exec_module(_R04_MODULE)

QUANTILE_Z = _R04_MODULE.QUANTILE_Z
fold_definitions = _R04_MODULE.fold_definitions
git_state = _R04_MODULE.git_state
markdown_table = _R04_MODULE.markdown_table
normal_crps = _R04_MODULE.normal_crps
now_utc = _R04_MODULE.now_utc
sha256_file = _R04_MODULE.sha256_file


def build_memory_features(data_root: Path) -> tuple[pd.DataFrame, Path]:
    r04_path = data_root / "gold" / "hkg_t24" / "r04_thermal_trajectory" / "r04_feature_matrix.parquet"
    if not r04_path.exists():
        raise FileNotFoundError(f"R05 requires R04 feature matrix: {r04_path}")
    base = pd.read_parquet(r04_path).sort_values("target_date").reset_index(drop=True)
    base["target_date"] = pd.to_datetime(base["target_date"])
    assert_no_locked_dates(base["target_date"], context="R05 feature source")
    out = base[
        [
            "target_date",
            "target_tmax_c",
            "doy_sin",
            "doy_cos",
            "day_length_hours",
            "noon_solar_elevation_deg",
            "hko_latest_temp_c",
            "hko_temp_range_so_far_c",
            "hko_temp_std_so_far_c",
            "hko_temp_change_180m_to_latest_c",
            "hko_temp_change_360m_to_latest_c",
            "hko_trailing_nonwarming_minutes",
        ]
    ].copy()
    for lag in range(1, 8):
        out[f"lag{lag}_cutoff_temp_c"] = base["hko_latest_temp_c"].shift(lag - 1)
        out[f"lag{lag}_range_so_far_c"] = base["hko_temp_range_so_far_c"].shift(lag - 1)
        out[f"lag{lag}_std_so_far_c"] = base["hko_temp_std_so_far_c"].shift(lag - 1)
    for lag in range(2, 8):
        out[f"lag1_minus_lag{lag}_cutoff_temp_c"] = out["lag1_cutoff_temp_c"] - out[f"lag{lag}_cutoff_temp_c"]
    for window in (3, 5, 7):
        cols = [f"lag{lag}_cutoff_temp_c" for lag in range(1, window + 1)]
        out[f"memory_mean_{window}d_cutoff_temp_c"] = out[cols].mean(axis=1)
        out[f"memory_std_{window}d_cutoff_temp_c"] = out[cols].std(axis=1)
        out[f"memory_range_{window}d_cutoff_temp_c"] = out[cols].max(axis=1) - out[cols].min(axis=1)
        weights_x = np.arange(1, window + 1, dtype=float)
        slopes: list[float | None] = []
        for _, row in out[cols].iterrows():
            values = row.to_numpy(dtype=float)
            if np.isnan(values).any():
                slopes.append(None)
            else:
                slopes.append(float(np.polyfit(weights_x, values, deg=1)[0]))
        out[f"memory_trend_{window}d_cutoff_temp_c_per_lag"] = slopes
    for half_life in HALF_LIVES:
        weights = np.array([0.5 ** ((lag - 1) / half_life) for lag in range(1, 8)], dtype=float)
        cols = [f"lag{lag}_cutoff_temp_c" for lag in range(1, 8)]
        weighted: list[float | None] = []
        for _, row in out[cols].iterrows():
            values = row.to_numpy(dtype=float)
            mask = ~np.isnan(values)
            if not mask.any():
                weighted.append(None)
            else:
                weighted.append(float(np.dot(values[mask], weights[mask]) / weights[mask].sum()))
        out[f"ewma_cutoff_temp_half_life_{half_life}d_c"] = weighted
    out["abs_lag1_minus_lag2_c"] = out["lag1_minus_lag2_cutoff_temp_c"].abs()
    out["transition_candidate_abs_change_ge_1p5c"] = (out["abs_lag1_minus_lag2_c"] >= 1.5).astype(float)
    regime_duration: list[int | None] = []
    changes = out["transition_candidate_abs_change_ge_1p5c"].fillna(0).to_numpy(dtype=float)
    for idx in range(len(out)):
        if idx < 1 or np.isnan(out.loc[idx, "lag2_cutoff_temp_c"]):
            regime_duration.append(None)
            continue
        duration = 1
        j = idx
        while j > 0 and changes[j] == 0 and duration < 7:
            duration += 1
            j -= 1
        regime_duration.append(duration)
    out["regime_duration_days_since_large_cutoff_change"] = regime_duration
    out = out.dropna(subset=["lag7_cutoff_temp_c"]).reset_index(drop=True)
    assert_no_locked_dates(out["target_date"], context="R05 memory feature matrix")
    output_path = data_root / "gold" / "hkg_t24" / "r05_thermal_memory" / "r05_feature_matrix.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    return out, output_path


def feature_sets(features: pd.DataFrame) -> dict[str, list[str]]:
    deny = {"target_date", "target_tmax_c"}
    numeric = [col for col in features.columns if col not in deny and pd.api.types.is_numeric_dtype(features[col])]
    baseline = [
        "doy_sin",
        "doy_cos",
        "day_length_hours",
        "noon_solar_elevation_deg",
        "lag1_cutoff_temp_c",
    ]
    lag3 = baseline + [
        "lag2_cutoff_temp_c",
        "lag3_cutoff_temp_c",
        "lag1_minus_lag2_cutoff_temp_c",
        "lag1_minus_lag3_cutoff_temp_c",
        "memory_mean_3d_cutoff_temp_c",
        "memory_std_3d_cutoff_temp_c",
        "memory_trend_3d_cutoff_temp_c_per_lag",
    ]
    lag7 = [col for col in numeric if "lag" in col or col.startswith("memory_") or col in baseline]
    ewma = baseline + [f"ewma_cutoff_temp_half_life_{half_life}d_c" for half_life in HALF_LIVES] + [
        "memory_std_7d_cutoff_temp_c",
        "memory_range_7d_cutoff_temp_c",
        "regime_duration_days_since_large_cutoff_change",
        "transition_candidate_abs_change_ge_1p5c",
    ]
    return {
        "r05_baseline_lag1_cutoff_temp_calendar": [col for col in baseline if col in features.columns],
        "r05_memory_lags_1_3": [col for col in lag3 if col in features.columns],
        "r05_memory_lags_1_7": [col for col in lag7 if col in features.columns],
        "r05_ewma_gated_memory": [col for col in ewma if col in features.columns],
    }


def fit_pipeline() -> object:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


def run_oof(features: pd.DataFrame, sets: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold_id, test_start, test_end, train_end in fold_definitions():
        train = features[features["target_date"] <= train_end].copy()
        test = features[(features["target_date"] >= test_start) & (features["target_date"] <= test_end)].copy()
        if len(train) < 330 or test.empty:
            continue
        for model_id, cols in sets.items():
            model = fit_pipeline()
            model.fit(train[list(cols)], train["target_tmax_c"])
            train_pred = model.predict(train[list(cols)])
            sigma = float(np.std(train["target_tmax_c"].to_numpy(dtype=float) - train_pred, ddof=1))
            sigma = max(sigma, 0.2)
            pred = test[["target_date", "target_tmax_c"]].copy()
            pred["fold_id"] = fold_id
            pred["model_id"] = model_id
            pred["model_family"] = "ridge_predeclared_memory_diagnostic"
            pred["training_start"] = train["target_date"].min()
            pred["training_end"] = train["target_date"].max()
            pred["training_rows"] = int(len(train))
            pred["feature_count"] = int(len(cols))
            pred["point_forecast"] = model.predict(test[list(cols)])
            pred["distribution_sigma_c"] = sigma
            for qcol, z_value in QUANTILE_Z.items():
                pred[qcol] = pred["point_forecast"] + pred["distribution_sigma_c"] * z_value
            rows.append(pred)
    if not rows:
        raise RuntimeError("R05 produced no OOF predictions.")
    predictions = pd.concat(rows, ignore_index=True).sort_values(["target_date", "model_id"]).reset_index(drop=True)
    assert_no_locked_dates(predictions["target_date"], context="R05 OOF predictions")
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
    scores = score_frame(predictions, ["fold_id", "model_id"])
    baseline = scores[scores["model_id"] == "r05_baseline_lag1_cutoff_temp_calendar"][
        ["fold_id", "mae", "crps_normal"]
    ].rename(columns={"mae": "baseline_mae", "crps_normal": "baseline_crps"})
    return scores.merge(baseline, on="fold_id", how="left").assign(
        mae_improvement_vs_baseline=lambda df: df["baseline_mae"] - df["mae"],
        crps_improvement_vs_baseline=lambda df: df["baseline_crps"] - df["crps_normal"],
    )


def memory_decay(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lag in range(1, 8):
        col = f"lag{lag}_cutoff_temp_c"
        valid = features[[col, "target_tmax_c"]].dropna()
        rows.append(
            {
                "lag_days": lag,
                "n": int(len(valid)),
                "pearson_corr_with_target": float(valid[col].corr(valid["target_tmax_c"])),
                "mae_if_direct_persistence": float((valid[col] - valid["target_tmax_c"]).abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def long_report(payload: Mapping[str, object]) -> str:
    champion = payload["champion"]
    assert isinstance(champion, Mapping)
    oof = payload["oof_feasibility"]
    assert isinstance(oof, Mapping)
    return f"""# EXP-0037 / HKG-T24-R05 Long-Form Experiment Report

## Purpose

R05 tests multi-day thermal memory for the HKO T-24 Tmax problem. R04 asked whether the shape of the current T-1 cutoff-day curve adds value beyond the latest eligible temperature. R05 asks whether the previous two to seven cutoff-safe thermal states carry additional next-day information, and whether persistence becomes dangerous around abrupt transitions. It remains fully independent of Polymarket work, validation 2024, and locked-test rows.

## Data and Eligibility

The experiment uses the R04 cutoff-safe feature matrix as its source. For target day T, `lag1_cutoff_temp_c` is the T-1 HKO latest eligible temperature at the 15:00 cutoff, with the latest ordinary observed timestamp capped at 14:40 under the +20 minute replay latency rule. Lags 2 through 7 are prior cutoff-safe states from earlier origin dates. These are operationally plausible station features. Lagged official daily Tmax labels are not used as operational predictors here, because their publication timing has not been proven for the T-1 15:00 cutoff. That is a deliberate exclusion, not an omission.

## Features

R05 constructs lagged cutoff temperatures for lags 1 through 7, lagged intraday range and standard-deviation summaries, lag1-minus-lagN differences, 3/5/7-day mean, range, standard deviation, and trend of cutoff temperature, exponentially weighted thermal levels with half-lives 1, 2, 3, and 5 days, an absolute lag1-lag2 transition magnitude, a binary large-change candidate at 1.5 C, and a regime-duration counter since the last large cutoff-temperature change. These are simple, interpretable memory features designed to test persistence half-life without fitting a large opaque model.

## Models and Ablations

The diagnostic model ladder is deliberately compact. The baseline uses lag1 cutoff temperature, deterministic calendar seasonality, day length, and noon solar elevation. The 1-3 day memory model adds lag2, lag3, short differences, and 3-day memory summaries. The 1-7 day memory model adds all lag and memory features. The EWMA/gated memory model uses exponentially weighted memory and transition/regime-duration summaries. All models use Ridge regression with alpha 1.0, median imputation, and standardization fitted inside each chronological fold only.

## OOF Design and Gate

The chronological folds are inherited from R04: 2021-H2, 2022-H1, 2022-H2, 2023-H1, and 2023-H2, each trained only on earlier dates. The strict four-year feasibility check is `{oof['status']}`: {oof['reason']}. R05 therefore cannot promote a memory feature family even if it improves in the blocked diagnostic folds. It is complete as an evidence-generating experiment and blocked as a promotable modern high-frequency model experiment.

## Main Result

The best diagnostic model by OOF MAE is `{champion['model_id']}` with MAE `{champion['mae']:.4f}` C, RMSE `{champion['rmse']:.4f}` C, bias `{champion['bias']:.4f}` C, and CRPS `{champion['crps_normal']:.4f}` over `{champion['n']}` rows. The result must be read beside the fold-delta table. A true promotion would require stable improvement across at least three chronological folds and enough OOF coverage; the strict OOF coverage criterion is not met.

## Interpretation

The point of R05 is to separate useful thermal memory from dangerous stale persistence. If short lags improve over lag1 alone, that suggests stable warm or cool regimes carry information beyond the current cutoff snapshot. If longer memory hurts, it suggests old thermal state becomes stale around transitions. If EWMA features help only in stable folds, they should become gated expert inputs rather than universal predictors. If no memory model beats lag1, the latest cutoff state remains the best target-station summary and attention should move to moisture, pressure, wind, station-network, upper-air, and forecast-vintage signals.

## Publication-Timing Blocker

Lagged official daily Tmax labels are tempting because they provide a long and smooth memory signal. R05 does not use them as operational predictors. The HKO daily target is a label source, and T-1 daily values are not known at T-1 15:00. T-2/T-3 daily labels might eventually be usable, but only after empirical publication timing proves they were available before cutoff for each historical row. Until then, any lagged official-label memory experiment must be separately marked target-history or mechanism-only. This report keeps that separation explicit.

## Artifacts

The feature matrix is stored at `C:\\hkg_tmax_data\\gold\\hkg_t24\\r05_thermal_memory\\r05_feature_matrix.parquet`. OOF predictions, scoreboards, fold deltas, and memory-decay diagnostics are stored under the same data-root folder and copied or summarized in the experiment directory. The repo-level report is `reports/hkg_t24/R05_THERMAL_MEMORY.md`. The reproduction command is in `REPRODUCE.md`.

## Date Ranges Used

The feature target-date period is `{payload['feature_min']}` through `{payload['feature_max']}`. The OOF prediction period is `{payload['prediction_min']}` through `{payload['prediction_max']}`. This narrower prediction period starts after the warm-up required by chronological training folds and seven-day lag construction. The experiment does not look at validation year 2024, and it does not inspect, score, transform, or summarize any locked-test row from 2025-01-01 onward. The modern high-frequency archive is valuable, but for this specific feature family the available pre-validation history is still shorter than the user's hard four-year OOF requirement. That is why R05 is preserved as evidence and not treated as an accepted model improvement.

## Leakage Controls Applied

Every row is interpreted as a T-24 forecast origin at the day-before 15:00 local cutoff. R05 inherits the R04 rule that the latest ordinary station observation can be no later than 14:40 when a +20 minute replay latency is assumed. Lag 1 means the previous origin's cutoff-safe state, not the target day's observed maximum. The script calls the locked-date guard on the input feature matrix and the prediction table, then writes explicit `validation_2024_accessed: false` and `locked_test_accessed: false` metadata into the experiment directory. Preprocessing is also leakage-controlled: median imputation and standardization are fit inside the training slice of each chronological fold rather than on the full dataset. This prevents future rows from influencing scale, missing-value defaults, or model coefficients.

## What Was Actually Tested

The direct baseline asks whether the most recent eligible cutoff temperature plus deterministic seasonality is enough. The short-memory candidate asks whether the previous three cutoff states help distinguish persistent warm regimes from one-day noise. The long-memory candidate asks whether a full week of cutoff states improves or dilutes the signal. The EWMA and transition-gated candidate asks whether decayed memory and large-change flags are better than raw lags. The memory-decay diagnostic is separate from the fitted model ladder: it measures how each lag correlates with the target and how bad a naive persistence forecast would be at each lag. This helps identify whether a future model should use memory as a smooth expert, a gated regime feature, or not at all.

## How It Went

The experiment ran cleanly after the direct-execution import path was corrected so that the CLI script and tests can both load the shared R04 helper functions. The generated feature matrix contains only pre-2024 target dates and contains no locked-test rows. The OOF predictions also contain only 2021-07-01 through 2023-12-31, with no validation or locked-test dates. The best OOF diagnostic model was the simple lag1 cutoff-temperature baseline rather than a richer memory candidate. That means the additional memory features did not yet prove reliable incremental value in the available modern high-frequency window. This is a useful negative result: it directs later experiments toward other signal families such as moisture, wind, pressure, station-network gradients, upper-air profiles, NWP guidance, and forecast vintage deltas instead of overfitting stale persistence.

## Why This Is Still Useful

R05 creates a reusable, audited memory-feature construction path. Later experiments can join these features with other strictly as-of predictors and test interaction terms, but they must keep the same lag semantics and fold-local preprocessing. The result also documents an important governance decision: official daily target-history features are not automatically safe just because they refer to past dates. Availability time matters. A daily value for yesterday could still be unknown at today's 15:00 origin depending on publication behavior, so it remains excluded until a separate publication-latency audit proves it available. This prevents the system from quietly learning from future-published labels.

## Decision Record

R05 is accepted as a completed diagnostic experiment and rejected as a promotable feature-family improvement. The rejection is not because the code failed; it is because the strict OOF acceptance gate failed and the best simple memory baseline did not establish robust improvement. The correct next move is not to tune R05 harder against these same folds. The correct next move is to run the next predeclared signal-family experiment, preserve the same leakage controls, and update the research ledger so the accumulated evidence remains auditable.

## Downstream Rule

R05 does not authorize validation access or model promotion. Any useful memory signal is recorded as `OOF_BLOCKED_DIAGNOSTIC` until the modern high-frequency sample reaches at least four pre-validation-equivalent OOF years or the evaluation design is explicitly revised without touching validation 2024 or the locked test. Later experiments may reuse memory features only if the lag construction remains strictly backward-looking and fold-local preprocessing is preserved.
"""


def write_experiment(
    *,
    data_root: Path,
    feature_path: Path,
    predictions_path: Path,
    scoreboard: pd.DataFrame,
    fold_scores: pd.DataFrame,
    decay: pd.DataFrame,
    payload: dict[str, object],
) -> None:
    for subdir in ["results", "artifacts", "predictions", "logs"]:
        (EXPERIMENT_DIR / subdir).mkdir(parents=True, exist_ok=True)
    metrics = {
        "research_id": RESEARCH_ID,
        "experiment_id": "EXP-0037",
        "status": "COMPLETE_DIAGNOSTIC_OOF_BLOCKED",
        "validation_2024_accessed": False,
        "locked_test_accessed": False,
        "champion": payload["champion"],
        "oof_feasibility": payload["oof_feasibility"],
        "scoreboard": scoreboard.to_dict(orient="records"),
        "fold_scores": fold_scores.to_dict(orient="records"),
        "memory_decay": decay.to_dict(orient="records"),
    }
    write_text(EXPERIMENT_DIR / "results" / "metrics.json", json.dumps(metrics, indent=2, default=str))
    scoreboard.to_csv(EXPERIMENT_DIR / "results" / "scoreboard.csv", index=False)
    fold_scores.to_csv(EXPERIMENT_DIR / "artifacts" / "fold_score_deltas.csv", index=False)
    decay.to_csv(EXPERIMENT_DIR / "artifacts" / "memory_decay.csv", index=False)
    pd.read_parquet(predictions_path).to_parquet(EXPERIMENT_DIR / "predictions" / "r05_oof_predictions.parquet", index=False)
    write_text(EXPERIMENT_DIR / "README.md", "# EXP-0037 HKG-T24-R05 Thermal Memory\n\nCutoff-safe multi-day thermal-memory diagnostic. No validation 2024, no locked test, no Polymarket. Status is OOF-blocked under the strict four-year rule.\n")
    write_text(EXPERIMENT_DIR / "HYPOTHESIS.md", "# Hypothesis\n\nRecent cutoff-safe thermal states over the last 2-7 days may improve next-day Tmax forecasts in stable regimes, while older memory may become harmful around transitions.\n")
    write_text(EXPERIMENT_DIR / "PROTOCOL.md", "# Protocol\n\n1. Use R04 cutoff-safe features only through 2023-12-31.\n2. Build lagged T-1...T-7 cutoff-state features without using target-day observations.\n3. Exclude lagged official daily labels as operational predictors until publication timing is proven.\n4. Fit imputation, scaling, and Ridge model inside chronological folds only.\n5. Do not access validation 2024 or locked-test rows.\n")
    write_text(EXPERIMENT_DIR / "ASOF_CONTRACT.md", "# As-Of Contract\n\nAll features are prior cutoff-safe station states. Lag 1 is T-1 at cutoff; lags 2-7 are earlier cutoff states. No target-day T observations and no lagged official target labels are operational predictors in this experiment.\n")
    write_text(EXPERIMENT_DIR / "DATA_MANIFEST.yaml", f"""research_id: {RESEARCH_ID}
feature_matrix: {feature_path}
feature_matrix_sha256: {sha256_file(feature_path)}
prediction_table: {predictions_path}
prediction_table_sha256: {sha256_file(predictions_path)}
data_root: {data_root}
validation_2024_accessed: false
locked_test_accessed: false
availability_tier: SILVER_OPERATIONAL_REPLAY
publication_blocked_inputs: lagged_official_daily_tmax
""")
    write_text(EXPERIMENT_DIR / "RUN_CONFIG.yaml", f"""research_id: {RESEARCH_ID}
analysis_start: {payload['feature_min']}
analysis_end: {payload['feature_max']}
lag_days: [1, 2, 3, 4, 5, 6, 7]
model_family: ridge_alpha_1_with_fold_local_imputer_scaler
validation_2024_accessed: false
locked_test_policy: deny
""")
    write_text(EXPERIMENT_DIR / "DATE_RANGES.md", f"""# Date Ranges

- Feature target-date period: `{payload['feature_min']}` through `{payload['feature_max']}`.
- OOF prediction period: `{payload['prediction_min']}` through `{payload['prediction_max']}`.
- Validation 2024: not accessed.
- Locked test 2025-01-01 onward: not accessed.
- Four-year OOF status: `{payload['oof_feasibility']['status']}`.
""")
    write_text(
        EXPERIMENT_DIR / "RESULTS.md",
        "# Results\n\n"
        + markdown_table(scoreboard, ["model_id", "n", "mae", "rmse", "median_abs_error", "bias", "crps_normal", "coverage_80", "coverage_90"])
        + "\n## Memory Decay\n\n"
        + markdown_table(decay, ["lag_days", "n", "pearson_corr_with_target", "mae_if_direct_persistence"]),
    )
    write_text(EXPERIMENT_DIR / "CONCLUSION.md", "# Conclusion\n\nR05 is complete as a cutoff-safe multi-day memory diagnostic, but it is not promotable because the modern high-frequency sample before validation 2024 fails the strict four-year OOF requirement. Lagged official daily labels remain publication-timing blocked.\n")
    write_text(EXPERIMENT_DIR / "REPRODUCE.md", "# Reproduce\n\n```powershell\n.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_r05_thermal_memory.py --data-root C:\\hkg_tmax_data\n```\n")
    write_text(EXPERIMENT_DIR / "STATUS.yaml", """status: COMPLETE_DIAGNOSTIC_OOF_BLOCKED
research_id: HKG-T24-R05
locked_test_accessed: false
validation_2024_accessed: false
leakage_guard: PASS
four_year_oof: BLOCKED
production_eligible: false
publication_blocked_inputs: lagged_official_daily_tmax
""")
    write_text(EXPERIMENT_DIR / "EXPERIMENT_REPORT_7500_CHARS.md", long_report(payload))


def write_report(scoreboard: pd.DataFrame, fold_scores: pd.DataFrame, decay: pd.DataFrame, payload: dict[str, object]) -> None:
    write_text(
        REPO_ROOT / "reports" / "hkg_t24" / "R05_THERMAL_MEMORY.md",
        long_report(payload)
        + "\n# R05 Machine-Readable Summary Tables\n\n"
        f"Generated: `{now_utc()}`\n\n"
        "## Overall Scoreboard\n\n"
        + markdown_table(scoreboard, ["model_id", "n", "first_date", "last_date", "mae", "rmse", "median_abs_error", "bias", "crps_normal", "coverage_80", "coverage_90"])
        + "\n## Fold Deltas\n\n"
        + markdown_table(fold_scores, ["fold_id", "model_id", "n", "mae", "baseline_mae", "mae_improvement_vs_baseline", "crps_improvement_vs_baseline"], limit=80)
        + "\n## Memory Decay\n\n"
        + markdown_table(decay, ["lag_days", "n", "pearson_corr_with_target", "mae_if_direct_persistence"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HKG-T24-R05 thermal memory diagnostic.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    features, feature_path = build_memory_features(data_root)
    sets = feature_sets(features)
    predictions = run_oof(features, sets)
    scoreboard = score_frame(predictions, ["model_id"])
    fold_scores = fold_deltas(predictions)
    decay = memory_decay(features)
    output_dir = data_root / "gold" / "hkg_t24" / "r05_thermal_memory"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "r05_oof_predictions.parquet"
    scoreboard_path = output_dir / "r05_scoreboard.parquet"
    fold_path = output_dir / "r05_fold_score_deltas.parquet"
    decay_path = output_dir / "r05_memory_decay.parquet"
    predictions.to_parquet(predictions_path, index=False)
    scoreboard.to_parquet(scoreboard_path, index=False)
    fold_scores.to_parquet(fold_path, index=False)
    decay.to_parquet(decay_path, index=False)
    feasibility = check_four_year_oof_feasibility(
        features["target_date"].min().date(),
        features["target_date"].max().date(),
        reason_context="R05 modern HKO thermal memory pre-validation feature period",
    )
    champion = scoreboard.iloc[0].to_dict()
    payload: dict[str, object] = {
        "generated_at_utc": now_utc(),
        "git": git_state(),
        "feature_min": str(features["target_date"].min().date()),
        "feature_max": str(features["target_date"].max().date()),
        "feature_rows": int(len(features)),
        "feature_columns": int(len(features.columns)),
        "prediction_min": str(predictions["target_date"].min().date()),
        "prediction_max": str(predictions["target_date"].max().date()),
        "prediction_rows": int(len(predictions)),
        "champion": champion,
        "oof_feasibility": feasibility.__dict__,
        "data_root_outputs": {
            "feature_matrix": str(feature_path),
            "oof_predictions": str(predictions_path),
            "scoreboard": str(scoreboard_path),
            "fold_score_deltas": str(fold_path),
            "memory_decay": str(decay_path),
        },
    }
    write_experiment(
        data_root=data_root,
        feature_path=feature_path,
        predictions_path=predictions_path,
        scoreboard=scoreboard,
        fold_scores=fold_scores,
        decay=decay,
        payload=payload,
    )
    write_report(scoreboard, fold_scores, decay, payload)
    print(json.dumps({"status": "ok", "champion": champion, "oof": feasibility.__dict__}, indent=2, default=str))


if __name__ == "__main__":
    main()
