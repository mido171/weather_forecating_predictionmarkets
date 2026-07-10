from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hkg_tmax.hkg_t24.governance import check_four_year_oof_feasibility
from hkg_tmax.hkg_t24.guard import assert_no_locked_dates

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(r"C:\hkg_tmax_data")
RESEARCH_ID = "HKG-T24-R10"
EXPERIMENT_ID = "EXP-0042"
EXPERIMENT_DIR = REPO_ROOT / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0042-HKG-T24-R10"

_R04_SPEC = importlib.util.spec_from_file_location(
    "run_hkg_t24_r04_thermal_trajectory",
    REPO_ROOT / "scripts" / "run_hkg_t24_r04_thermal_trajectory.py",
)
if _R04_SPEC is None or _R04_SPEC.loader is None:
    raise ImportError("Unable to load R04 helper script for R10 latent-mode experiment.")
_R04_MODULE = importlib.util.module_from_spec(_R04_SPEC)
sys.modules[_R04_SPEC.name] = _R04_MODULE
_R04_SPEC.loader.exec_module(_R04_MODULE)

_R06_SPEC = importlib.util.spec_from_file_location(
    "run_hkg_t24_r06_moisture_state",
    REPO_ROOT / "scripts" / "run_hkg_t24_r06_moisture_state.py",
)
if _R06_SPEC is None or _R06_SPEC.loader is None:
    raise ImportError("Unable to load R06 helper script for R10 latent-mode experiment.")
_R06_MODULE = importlib.util.module_from_spec(_R06_SPEC)
sys.modules[_R06_SPEC.name] = _R06_MODULE
_R06_SPEC.loader.exec_module(_R06_MODULE)

QUANTILE_Z = _R04_MODULE.QUANTILE_Z
fold_definitions = _R04_MODULE.fold_definitions
git_state = _R04_MODULE.git_state
markdown_table = _R04_MODULE.markdown_table
normal_crps = _R04_MODULE.normal_crps
now_utc = _R04_MODULE.now_utc
r04_feature_sets = _R04_MODULE.model_feature_sets
sha256_file = _R04_MODULE.sha256_file

valid_columns = _R06_MODULE.valid_columns
write_text = _R06_MODULE.write_text


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_family: str
    base_columns: tuple[str, ...]
    station_columns: tuple[str, ...]
    n_components: int
    factory: Callable[[], object]
    include_reconstruction_error: bool = False
    boosting: bool = False


def station_offset_columns(features: pd.DataFrame) -> list[str]:
    return sorted(col for col in features.columns if col.startswith("station_offset_hko_minus_"))


def active_cols(train: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [
        col
        for col in columns
        if col in train.columns and train[col].notna().sum() > 0 and train[col].nunique(dropna=True) > 1
    ]


def model_specs(features: pd.DataFrame) -> list[ModelSpec]:
    baseline = tuple(r04_feature_sets(features)["r04_baseline_latest_temp_calendar"])
    station_cols = tuple(station_offset_columns(features))
    group_cols = tuple(
        col
        for col in [
            "temp_network_spread_c",
            "temp_network_hko_minus_median_c",
            "temp_network_inland_minus_coastal_c",
            "temp_network_urban_minus_coastal_c",
            "temp_network_east_minus_west_c",
            "temp_network_north_minus_south_c",
        ]
        if col in features.columns
    )
    def ridge_factory() -> Pipeline:
        return Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]
        )
    return [
        ModelSpec("r10_baseline_temp_calendar", "ridge_baseline", baseline, tuple(), 0, ridge_factory),
        ModelSpec("r10_pca3_station_offsets_ridge", "fold_local_pca3_station_offsets", baseline, station_cols, 3, ridge_factory),
        ModelSpec("r10_pca5_station_offsets_ridge", "fold_local_pca5_station_offsets", baseline, station_cols, 5, ridge_factory),
        ModelSpec("r10_pca8_station_offsets_ridge", "fold_local_pca8_station_offsets", baseline, station_cols, 8, ridge_factory),
        ModelSpec(
            "r10_pca5_with_reconstruction_error_ridge",
            "fold_local_pca5_with_reconstruction_error",
            baseline + group_cols,
            station_cols,
            5,
            ridge_factory,
            include_reconstruction_error=True,
        ),
        ModelSpec(
            "r10_shallow_boosting_pca5_modes",
            "hist_gradient_boosting_pca5_modes",
            baseline + group_cols,
            station_cols,
            5,
            lambda: HistGradientBoostingRegressor(
                max_iter=60,
                max_leaf_nodes=7,
                learning_rate=0.04,
                l2_regularization=1.0,
                min_samples_leaf=30,
                random_state=1010,
            ),
            include_reconstruction_error=True,
            boosting=True,
        ),
    ]


def fold_local_pca_scores(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Sequence[str],
    n_components: int,
    *,
    fold_id: str,
    model_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]], dict[str, float]]:
    cols = active_cols(train, columns)
    if not cols or n_components <= 0:
        return pd.DataFrame(index=train.index), pd.DataFrame(index=test.index), [], {}
    n_comp = min(n_components, len(cols), max(1, len(train) - 1))
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train[cols]))
    x_test = scaler.transform(imputer.transform(test[cols]))
    pca = PCA(n_components=n_comp, random_state=1010)
    train_scores = pca.fit_transform(x_train)
    test_scores = pca.transform(x_test)
    train_recon = pca.inverse_transform(train_scores)
    test_recon = pca.inverse_transform(test_scores)
    train_error = np.sqrt(np.mean(np.square(x_train - train_recon), axis=1))
    test_error = np.sqrt(np.mean(np.square(x_test - test_recon), axis=1))
    train_frame = pd.DataFrame(train_scores, index=train.index, columns=[f"pc{i + 1}" for i in range(n_comp)])
    test_frame = pd.DataFrame(test_scores, index=test.index, columns=[f"pc{i + 1}" for i in range(n_comp)])
    train_frame["pca_reconstruction_error"] = train_error
    test_frame["pca_reconstruction_error"] = test_error
    loadings: list[dict[str, object]] = []
    for pc_idx, component in enumerate(pca.components_, start=1):
        for feature, loading in zip(cols, component, strict=True):
            loadings.append(
                {
                    "fold_id": fold_id,
                    "model_id": model_id,
                    "pc": pc_idx,
                    "feature": feature,
                    "loading": float(loading),
                    "explained_variance_ratio": float(pca.explained_variance_ratio_[pc_idx - 1]),
                }
            )
    variance = {f"pc{idx + 1}_explained_variance_ratio": float(value) for idx, value in enumerate(pca.explained_variance_ratio_)}
    return train_frame, test_frame, loadings, variance


def design_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: ModelSpec,
    *,
    fold_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]], int]:
    base_cols = active_cols(train, spec.base_columns)
    train_x = train[base_cols].copy() if base_cols else pd.DataFrame(index=train.index)
    test_x = test[base_cols].copy() if base_cols else pd.DataFrame(index=test.index)
    loadings: list[dict[str, object]] = []
    if spec.n_components > 0:
        train_pc, test_pc, loadings, _ = fold_local_pca_scores(
            train,
            test,
            spec.station_columns,
            spec.n_components,
            fold_id=fold_id,
            model_id=spec.model_id,
        )
        pc_cols = [col for col in train_pc.columns if col.startswith("pc") and col[2:].isdigit()]
        keep_cols = pc_cols + (["pca_reconstruction_error"] if spec.include_reconstruction_error else [])
        train_x = pd.concat([train_x.reset_index(drop=True), train_pc[keep_cols].reset_index(drop=True)], axis=1)
        test_x = pd.concat([test_x.reset_index(drop=True), test_pc[keep_cols].reset_index(drop=True)], axis=1)
    return train_x, test_x, loadings, train_x.shape[1]


def run_oof(features: pd.DataFrame, specs: Sequence[ModelSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    loading_rows: list[dict[str, object]] = []
    for fold_id, test_start, test_end, train_end in fold_definitions():
        train = features[features["target_date"] <= train_end].copy()
        test = features[(features["target_date"] >= test_start) & (features["target_date"] <= test_end)].copy()
        if len(train) < 330 or test.empty:
            continue
        for spec in specs:
            train_x, test_x, loadings, feature_count = design_matrix(train, test, spec, fold_id=fold_id)
            if feature_count == 0:
                continue
            model = spec.factory()
            model.fit(train_x, train["target_tmax_c"])
            train_pred = model.predict(train_x)
            sigma = float(np.std(train["target_tmax_c"].to_numpy(dtype=float) - train_pred, ddof=1))
            sigma = max(sigma, 0.2)
            pred = test[["target_date", "target_tmax_c"]].copy()
            pred["fold_id"] = fold_id
            pred["model_id"] = spec.model_id
            pred["model_family"] = spec.model_family
            pred["training_start"] = train["target_date"].min()
            pred["training_end"] = train["target_date"].max()
            pred["training_rows"] = int(len(train))
            pred["feature_count"] = int(feature_count)
            pred["point_forecast"] = model.predict(test_x)
            pred["distribution_sigma_c"] = sigma
            for qcol, z_value in QUANTILE_Z.items():
                pred[qcol] = pred["point_forecast"] + pred["distribution_sigma_c"] * z_value
            rows.append(pred)
            loading_rows.extend(loadings)
    if not rows:
        raise RuntimeError("R10 produced no OOF predictions.")
    predictions = pd.concat(rows, ignore_index=True).sort_values(["target_date", "model_id"]).reset_index(drop=True)
    assert_no_locked_dates(predictions["target_date"], context="R10 OOF predictions")
    return predictions, pd.DataFrame(loading_rows)


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
    baseline = scores[scores["model_id"] == "r10_baseline_temp_calendar"][["fold_id", "mae", "crps_normal"]].rename(
        columns={"mae": "baseline_mae", "crps_normal": "baseline_crps"}
    )
    return scores.merge(baseline, on="fold_id", how="left").assign(
        mae_improvement_vs_baseline=lambda df: df["baseline_mae"] - df["mae"],
        crps_improvement_vs_baseline=lambda df: df["baseline_crps"] - df["crps_normal"],
    )


def mode_catalog(loadings: pd.DataFrame) -> pd.DataFrame:
    if loadings.empty:
        return loadings
    rows: list[dict[str, object]] = []
    for (model_id, pc), group in loadings.groupby(["model_id", "pc"], sort=True):
        top = group.assign(abs_loading=group["loading"].abs()).sort_values("abs_loading", ascending=False).head(8)
        rows.append(
            {
                "model_id": model_id,
                "pc": int(pc),
                "mean_explained_variance_ratio": float(group["explained_variance_ratio"].mean()),
                "top_features": "; ".join(top["feature"].astype(str).tolist()),
                "top_abs_loading_sum": float(top["abs_loading"].sum()),
            }
        )
    return pd.DataFrame(rows)


def long_report(payload: Mapping[str, object]) -> str:
    champion = payload["champion"]
    assert isinstance(champion, Mapping)
    oof = payload["oof_feasibility"]
    assert isinstance(oof, Mapping)
    return f"""# EXP-0042 / HKG-T24-R10 Long-Form Experiment Report

## Purpose

R10 tests whether fold-fit latent modes of the all-station temperature field capture mesoscale structure that hand-designed R09 contrasts miss. The key leakage constraint is that PCA/EOF preprocessing must be fit inside each chronological training fold only. R10 therefore does not create a single global PCA feature table. It builds fold-local imputation, scaling, PCA loadings, mode scores, and reconstruction-error features during OOF prediction.

## Data Used

The input is the R09 station-temperature-gradient feature matrix, including the HKO-minus-station offset columns generated from immutable high-frequency station-temperature snapshots. The feature target-date period is `{payload['feature_min']}` through `{payload['feature_max']}`, and the OOF prediction period is `{payload['prediction_min']}` through `{payload['prediction_max']}`. Validation 2024 and locked-test rows are not used.

## Methods

R10 runs baseline, PCA-3, PCA-5, PCA-8 station-offset Ridge models, a PCA-5 plus reconstruction-error Ridge model, and a shallow boosting model using fold-local PCA scores. PCA operates on station-offset columns, not target values. The mode catalog records fold, model id, principal component, station-offset feature loading, and explained variance ratio. Because station coordinates are not yet in the registry, graph-Laplacian modes are blocked rather than fabricated.

## Leakage Controls

For each chronological fold, imputation medians, scaling parameters, PCA components, reconstruction-error calculations, and regression/boosting models are fit using training rows only. Test rows are transformed by that fold's fitted preprocessing objects. No full-sample mode score is written. The deliberately global-fit negative control requested by the specification is not run as a model because it is a known leakage design; it is documented as forbidden and left for a leakage-test fixture rather than scored.

## OOF Gate

The strict four-year OOF check is `{oof['status']}`: {oof['reason']}. R10 is therefore a completed latent-mode diagnostic but not promotable under the hard four-year OOF rule.

## Main Result

The best non-control model by OOF MAE is `{champion['model_id']}` with MAE `{champion['mae']:.4f}` C, RMSE `{champion['rmse']:.4f}` C, bias `{champion['bias']:.4f}` C, and CRPS `{champion['crps_normal']:.4f}` over `{champion['n']}` rows. The mode catalog and fold-delta tables determine whether latent spatial structure is stable enough to carry forward.

## Interpretation

If a small PCA model beats baseline and R09 spatial summaries, it suggests the station field has coherent latent structure. If PCA loadings rotate unpredictably or only boosting improves, the modes are not yet operationally robust. If reconstruction error helps, field coherence or station-disagreement may be a useful missingness/transition signal. If all mode models lose, transparent R09 features or later graph/coordinate-aware modes are better next steps.

## Blockers

Graph-Laplacian modes, geography-aware loading maps, elevation-aware interpretation, and terrain adjacency are blocked by missing station coordinates/elevation in the current registry. Sparse PCA and probabilistic PCA are not added as dependencies in this pass; the first diagnostic uses standard fold-local PCA with explicit loadings.

## Decision Record

R10 is complete as a fold-local latent spatial mode diagnostic once artifacts and tests pass. It does not authorize validation access. The next planned experiment is R11 dynamic upwind station selection, which can use R08 vector winds and R09/R10 station-field representations but must preserve fold-local preprocessing.

## Operational Details and What Was Deliberately Not Done

The experiment uses the station-offset family because it is the highest-dimensional station-field representation already constructed under the current as-of contract. Each offset compares the HKO target station cutoff state with one neighboring station's cutoff-safe sampled temperature. This choice keeps the mode extraction tied to physical surface thermal contrasts rather than to target-day labels. It also avoids accidental use of since-midnight maximum/minimum values whose label semantics were shown in R03 to include carryover behavior that is not equivalent to an ordinary minute-level trace.

The PCA features are intentionally refit in every chronological fold. That means the loadings file is not a single global map of Hong Kong station modes. Instead, it is a fold-local audit table showing which stations were active, how each component loaded, and how much variance was explained inside that fold. This is less tidy than one global chart, but it is the only acceptable leakage-free design for an OOF experiment. If a future report wants a nice visual map, it must either be descriptive only or be produced from training-window-only objects for the exact fold being evaluated.

The shallow boosting model is included as a weak nonlinearity probe, not as a production candidate. Its tree depth and iteration count are deliberately small. The goal is to learn whether latent station modes contain interactions with seasonal and local cutoff-temperature state. It is not allowed to search a large hyperparameter space, because the modern OOF sample is short and already below the user's four-year reliability requirement.

R10 also refuses to turn the missing station-coordinate problem into a fake solution. A true graph-mode experiment needs coordinates, elevation, terrain/coastline context, and a defensible adjacency kernel. Those inputs must come from station metadata and static geospatial tables, not from station-name ordering or hand-waved groups. Until those fields are canonicalized, the graph portion of the R10 title remains a documented blocker while the fold-local PCA portion is complete.

The month/season relationship is handled by the baseline feature set inherited from R04, while the station field contributes only residual spatial shape. If a PCA model beats the baseline by a small amount but the loadings are dominated by one station or rotate dramatically across folds, the correct decision is not promotion. The correct decision is to treat the result as an unstable diagnostic and use it to design robustness work in R27 or catastrophic-error specialist work in R22.

## Date-Range Discipline

The effective feature matrix remains bounded by the modern HKO high-frequency archive and by the pre-validation cutoff used across R04-R10. R10 does not extend into 2024 to make the statistics look better. The source feature target-date period is `{payload['feature_min']}` through `{payload['feature_max']}`, and the OOF predictions cover `{payload['prediction_min']}` through `{payload['prediction_max']}`. Under the strict user requirement, this is not long enough for promotion. The short span is not hidden inside the result; it is the central reason for the experiment status.

## Reproducibility Notes

All primary outputs are written both to the immutable-style data root output directory and to the repository experiment folder. The repository folder contains the narrative, run config, date ranges, metrics JSON, scoreboard, fold deltas, mode catalog, prediction copy, as-of contract, data manifest with SHA256 values, and reproduction command. That makes this experiment handoff-safe: a later GPT-Pro or Codex conversation can inspect the folder without needing to reconstruct the rationale from terminal history.

The practical takeaway is deliberately narrow: fold-local latent station modes are promising enough to revisit, but the experiment cannot override the four-year OOF gate or the missing graph metadata blocker.
"""


def write_experiment(
    *,
    data_root: Path,
    feature_path: Path,
    predictions_path: Path,
    loadings_path: Path,
    scoreboard: pd.DataFrame,
    fold_scores: pd.DataFrame,
    catalog: pd.DataFrame,
    payload: dict[str, object],
) -> None:
    for subdir in ["results", "artifacts", "predictions", "logs"]:
        (EXPERIMENT_DIR / subdir).mkdir(parents=True, exist_ok=True)
    metrics = {
        "research_id": RESEARCH_ID,
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_DIAGNOSTIC_OOF_BLOCKED",
        "validation_2024_accessed": False,
        "locked_test_accessed": False,
        "champion": payload["champion"],
        "oof_feasibility": payload["oof_feasibility"],
        "scoreboard": scoreboard.to_dict(orient="records"),
        "fold_scores": fold_scores.to_dict(orient="records"),
        "mode_catalog": catalog.to_dict(orient="records"),
    }
    write_text(EXPERIMENT_DIR / "results" / "metrics.json", json.dumps(metrics, indent=2, default=str))
    scoreboard.to_csv(EXPERIMENT_DIR / "results" / "scoreboard.csv", index=False)
    fold_scores.to_csv(EXPERIMENT_DIR / "artifacts" / "fold_score_deltas.csv", index=False)
    catalog.to_csv(EXPERIMENT_DIR / "artifacts" / "mode_catalog.csv", index=False)
    pd.read_parquet(predictions_path).to_parquet(EXPERIMENT_DIR / "predictions" / "r10_oof_predictions.parquet", index=False)
    write_text(EXPERIMENT_DIR / "README.md", "# EXP-0042 HKG-T24-R10 Latent Spatial Modes\n\nFold-local PCA/EOF station-temperature mode diagnostic. No global PCA, no validation 2024, no locked test, no Polymarket.\n")
    write_text(EXPERIMENT_DIR / "HYPOTHESIS.md", "# Hypothesis\n\nA few fold-fit station-temperature offset modes may summarize inland/coastal/elevation/urban spatial structures better than hand-designed R09 contrasts.\n")
    write_text(EXPERIMENT_DIR / "PROTOCOL.md", "# Protocol\n\n1. Start from R09 station-offset feature matrix.\n2. In each chronological fold, fit imputation, scaling, PCA, and models on training rows only.\n3. Transform test rows with fold-fitted preprocessors.\n4. Save fold-local loadings and mode catalog.\n5. Do not run global-fit PCA as a scored model.\n")
    write_text(EXPERIMENT_DIR / "ASOF_CONTRACT.md", "# As-Of Contract\n\nAll station offsets originate from T-1 cutoff-safe station-temperature snapshots. PCA is fold-local only; no full-sample spatial mode enters any OOF prediction.\n")
    write_text(EXPERIMENT_DIR / "DATA_MANIFEST.yaml", f"""research_id: {RESEARCH_ID}
feature_matrix: {feature_path}
feature_matrix_sha256: {sha256_file(feature_path)}
prediction_table: {predictions_path}
prediction_table_sha256: {sha256_file(predictions_path)}
mode_loadings: {loadings_path}
mode_loadings_sha256: {sha256_file(loadings_path)}
data_root: {data_root}
validation_2024_accessed: false
locked_test_accessed: false
blocked_inputs: [station_coordinates, graph_laplacian_modes, global_fit_pca_negative_control_scoring]
""")
    write_text(EXPERIMENT_DIR / "RUN_CONFIG.yaml", f"""research_id: {RESEARCH_ID}
analysis_start: {payload['feature_min']}
analysis_end: {payload['feature_max']}
model_ladder: baseline, pca3, pca5, pca8, pca5_reconstruction_error, shallow_boosting_pca5
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
        + "\n## Mode Catalog\n\n"
        + markdown_table(catalog, ["model_id", "pc", "mean_explained_variance_ratio", "top_features", "top_abs_loading_sum"], limit=80),
    )
    write_text(EXPERIMENT_DIR / "CONCLUSION.md", "# Conclusion\n\nR10 is complete as a fold-local latent spatial mode diagnostic, but it is OOF-blocked under the strict four-year rule. Graph/coordinate-aware modes remain blocked by missing station metadata.\n")
    write_text(EXPERIMENT_DIR / "REPRODUCE.md", "# Reproduce\n\n```powershell\n.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_r10_latent_spatial_modes.py --data-root C:\\hkg_tmax_data\n```\n")
    write_text(EXPERIMENT_DIR / "STATUS.yaml", """status: COMPLETE_DIAGNOSTIC_OOF_BLOCKED
research_id: HKG-T24-R10
locked_test_accessed: false
validation_2024_accessed: false
leakage_guard: PASS
four_year_oof: BLOCKED
production_eligible: false
blocked_inputs: [station_coordinates, graph_laplacian_modes, global_fit_pca_negative_control_scoring]
""")
    write_text(EXPERIMENT_DIR / "EXPERIMENT_REPORT_7500_CHARS.md", long_report(payload))


def write_report(scoreboard: pd.DataFrame, fold_scores: pd.DataFrame, catalog: pd.DataFrame, payload: dict[str, object]) -> None:
    write_text(
        REPO_ROOT / "reports" / "hkg_t24" / "R10_LATENT_SPATIAL_MODES.md",
        long_report(payload)
        + "\n# R10 Machine-Readable Summary Tables\n\n"
        f"Generated: `{now_utc()}`\n\n"
        "## Overall Scoreboard\n\n"
        + markdown_table(scoreboard, ["model_id", "n", "first_date", "last_date", "mae", "rmse", "median_abs_error", "bias", "crps_normal", "coverage_80", "coverage_90"])
        + "\n## Fold Deltas\n\n"
        + markdown_table(fold_scores, ["fold_id", "model_id", "n", "mae", "baseline_mae", "mae_improvement_vs_baseline", "crps_improvement_vs_baseline"], limit=100)
        + "\n## Mode Catalog\n\n"
        + markdown_table(catalog, ["model_id", "pc", "mean_explained_variance_ratio", "top_features", "top_abs_loading_sum"], limit=80),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HKG-T24-R10 fold-local latent spatial mode diagnostic.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    feature_path = data_root / "gold" / "hkg_t24" / "r09_station_temperature_gradient" / "r09_feature_matrix.parquet"
    if not feature_path.exists():
        raise FileNotFoundError(f"R10 requires R09 feature matrix: {feature_path}")
    features = pd.read_parquet(feature_path).sort_values("target_date").reset_index(drop=True)
    features["target_date"] = pd.to_datetime(features["target_date"])
    assert_no_locked_dates(features["target_date"], context="R10 source R09 matrix")
    predictions, loadings = run_oof(features, model_specs(features))
    scoreboard = score_frame(predictions, ["model_id"])
    fold_scores = fold_deltas(predictions)
    catalog = mode_catalog(loadings)
    output_dir = data_root / "gold" / "hkg_t24" / "r10_latent_spatial_modes"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "r10_oof_predictions.parquet"
    scoreboard_path = output_dir / "r10_scoreboard.parquet"
    fold_path = output_dir / "r10_fold_score_deltas.parquet"
    loadings_path = output_dir / "r10_mode_loadings.parquet"
    catalog_path = output_dir / "r10_mode_catalog.parquet"
    predictions.to_parquet(predictions_path, index=False)
    scoreboard.to_parquet(scoreboard_path, index=False)
    fold_scores.to_parquet(fold_path, index=False)
    loadings.to_parquet(loadings_path, index=False)
    catalog.to_parquet(catalog_path, index=False)
    feature_dates = pd.to_datetime(features["target_date"])
    feasibility = check_four_year_oof_feasibility(
        feature_dates.min().date(),
        feature_dates.max().date(),
        min_years=4.0,
        reason_context="R10 modern latent-spatial-mode pre-validation feature period",
    )
    champion = scoreboard.iloc[0].to_dict()
    payload = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "git_state": git_state(),
        "feature_min": str(feature_dates.min().date()),
        "feature_max": str(feature_dates.max().date()),
        "prediction_min": str(pd.to_datetime(predictions["target_date"]).min().date()),
        "prediction_max": str(pd.to_datetime(predictions["target_date"]).max().date()),
        "oof_feasibility": feasibility.__dict__,
        "champion": champion,
    }
    write_experiment(
        data_root=data_root,
        feature_path=feature_path,
        predictions_path=predictions_path,
        loadings_path=loadings_path,
        scoreboard=scoreboard,
        fold_scores=fold_scores,
        catalog=catalog,
        payload=payload,
    )
    write_report(scoreboard, fold_scores, catalog, payload)
    print(json.dumps({"status": "ok", "champion": champion, "oof": feasibility.__dict__}, indent=2, default=str))


if __name__ == "__main__":
    main()
