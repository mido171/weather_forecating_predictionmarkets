from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_beastmode_signal_discovery import (  # noqa: E402
    CONFIRMATION_START,
    RESEARCH_ROOT,
    markdown_table,
    require_no_confirmation_dates,
    score_prediction_frame,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_official_anchor_expert_blend_screen import (  # noqa: E402
    past_only_expert_blend,
)
from scripts.run_hkg_t24_residual_failure_cluster_discovery import (  # noqa: E402
    available_numeric_features,
    build_failure_frame,
    diagnostic_feature_candidates,
    simple_kmeans,
)
from scripts.run_hkg_t24_smooth_residual_archetype_specialists import (  # noqa: E402
    weighted_mean,
)

FOLDER_NAME = "0034_cluster_centroid_soft_gating"
MIN_HISTORY = 240
TOP_BLEND_EXPERTS = 14


@dataclass(frozen=True)
class ClusterCentroidSpec:
    anchor_col: str
    mode: str
    same_source: bool
    failure_quantile: float
    n_clusters: int
    k_neighbors: int = 40
    min_history: int = MIN_HISTORY
    min_failure_rows: int = 80
    shrinkage: float = 80.0
    correction_clip_c: float = 2.0
    gate_distance_quantile: float = 0.85
    min_local_mae_improvement_c: float = 0.0


@dataclass(frozen=True)
class ClusterCentroidModel:
    features: tuple[str, ...]
    means: np.ndarray
    stds: np.ndarray
    centroids: np.ndarray
    cluster_residual_means: np.ndarray
    cluster_rows: np.ndarray
    failure_scaled: np.ndarray
    failure_residuals: np.ndarray
    failure_labels: np.ndarray
    failure_dates: np.ndarray
    gate_distance: float
    distance_scale: float
    failure_threshold_c: float
    prior_rows: int
    failure_rows: int


@dataclass(frozen=True)
class CentroidCorrection:
    correction: float
    selected_rows: int
    nearest_cluster: int
    min_centroid_distance: float
    centroid_affinity: float
    gate_passed: bool
    local_anchor_mae: float
    local_corrected_mae: float


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 150) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def quarter_start(value: pd.Timestamp) -> pd.Timestamp:
    date = pd.Timestamp(value).normalize()
    month = ((int(date.month) - 1) // 3) * 3 + 1
    return pd.Timestamp(year=int(date.year), month=month, day=1)


def candidate_id_for_spec(spec: ClusterCentroidSpec) -> str:
    source = "same_source" if spec.same_source else "all_prior"
    return slug(
        f"cluster_centroid_{spec.anchor_col}_{spec.mode}_q{int(spec.failure_quantile * 100)}"
        f"_c{spec.n_clusters}_k{spec.k_neighbors}_{source}"
    )


def scale_features(values: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return (values - means) / stds


def finite_feature_matrix(frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    return frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def build_cluster_centroid_model(
    ordered: pd.DataFrame,
    *,
    features: tuple[str, ...],
    spec: ClusterCentroidSpec,
    prior_mask: np.ndarray,
) -> ClusterCentroidModel | None:
    if not features:
        return None
    feature_matrix = finite_feature_matrix(ordered, features)
    target = pd.to_numeric(ordered["target_tmax_c"], errors="coerce").to_numpy(dtype=float)
    anchor = pd.to_numeric(ordered[spec.anchor_col], errors="coerce").to_numpy(dtype=float)
    residual = target - anchor
    valid_prior = prior_mask & np.isfinite(target) & np.isfinite(anchor) & np.isfinite(residual)
    valid_prior &= np.isfinite(feature_matrix).all(axis=1)
    prior_index = np.flatnonzero(valid_prior)
    if len(prior_index) < spec.min_history:
        return None

    prior_abs_error = np.abs(residual[prior_index])
    threshold = max(float(np.nanquantile(prior_abs_error, spec.failure_quantile)), 1.0)
    failure_index = prior_index[prior_abs_error >= threshold]
    if len(failure_index) < max(spec.min_failure_rows, spec.n_clusters):
        return None

    prior_features = feature_matrix[prior_index]
    means = np.nanmean(prior_features, axis=0)
    stds = np.nanstd(prior_features, axis=0)
    stds = np.where((stds <= 1e-9) | ~np.isfinite(stds), 1.0, stds)
    failure_scaled = scale_features(feature_matrix[failure_index], means, stds)
    labels = simple_kmeans(failure_scaled, spec.n_clusters)

    centroids: list[np.ndarray] = []
    residual_means: list[float] = []
    cluster_rows: list[int] = []
    assigned_distances: list[float] = []
    for cluster_id in range(max(labels) + 1):
        mask = labels == cluster_id
        if not mask.any():
            continue
        cluster_matrix = failure_scaled[mask]
        centroid = cluster_matrix.mean(axis=0)
        centroids.append(centroid)
        residual_means.append(float(np.mean(residual[failure_index][mask])))
        cluster_rows.append(int(mask.sum()))
        assigned_distances.extend(np.sqrt(np.nanmean(np.square(cluster_matrix - centroid), axis=1)).tolist())

    if not centroids:
        return None
    distances = np.array(assigned_distances, dtype=float)
    finite_distances = distances[np.isfinite(distances)]
    if len(finite_distances) == 0:
        return None
    positive = finite_distances[finite_distances > 1e-9]
    distance_scale = float(np.nanmedian(positive)) if len(positive) else 1.0
    if not np.isfinite(distance_scale) or distance_scale <= 1e-9:
        distance_scale = 1.0
    gate_distance = float(np.nanquantile(finite_distances, spec.gate_distance_quantile))
    if not np.isfinite(gate_distance) or gate_distance <= 1e-9:
        gate_distance = distance_scale

    return ClusterCentroidModel(
        features=features,
        means=means,
        stds=stds,
        centroids=np.vstack(centroids),
        cluster_residual_means=np.array(residual_means, dtype=float),
        cluster_rows=np.array(cluster_rows, dtype=int),
        failure_scaled=failure_scaled,
        failure_residuals=residual[failure_index],
        failure_labels=labels,
        failure_dates=pd.to_datetime(ordered.iloc[failure_index]["target_date"]).dt.normalize().to_numpy(dtype="datetime64[ns]"),
        gate_distance=gate_distance,
        distance_scale=distance_scale,
        failure_threshold_c=threshold,
        prior_rows=int(len(prior_index)),
        failure_rows=int(len(failure_index)),
    )


def correction_from_model(
    model: ClusterCentroidModel,
    current_scaled: np.ndarray,
    *,
    spec: ClusterCentroidSpec,
) -> CentroidCorrection:
    centroid_distances = np.sqrt(np.nanmean(np.square(model.centroids - current_scaled), axis=1))
    if not np.isfinite(centroid_distances).any():
        return CentroidCorrection(0.0, 0, -1, math.nan, math.nan, False, math.nan, math.nan)
    nearest_cluster = int(np.nanargmin(centroid_distances))
    min_distance = float(centroid_distances[nearest_cluster])
    if min_distance > model.gate_distance:
        return CentroidCorrection(0.0, 0, nearest_cluster, min_distance, 0.0, False, math.nan, math.nan)

    centroid_weights = np.exp(-centroid_distances / model.distance_scale)
    if spec.mode == "centroid_mean":
        raw = weighted_mean(model.cluster_residual_means, centroid_weights * model.cluster_rows.astype(float))
        selected_mask = model.failure_labels == nearest_cluster
        if int(selected_mask.sum()) < min(spec.k_neighbors, len(model.failure_labels)):
            selected_mask = np.ones(len(model.failure_labels), dtype=bool)
        row_matrix = model.failure_scaled[selected_mask]
        row_residuals = model.failure_residuals[selected_mask]
        row_distances = np.sqrt(np.nanmean(np.square(row_matrix - current_scaled), axis=1))
        k = min(spec.k_neighbors, len(row_distances))
        order = np.argpartition(row_distances, k - 1)[:k]
        selected_residuals = row_residuals[order]
        selected_distances = row_distances[order]
    elif spec.mode == "failure_neighbor":
        row_distances = np.sqrt(np.nanmean(np.square(model.failure_scaled - current_scaled), axis=1))
        valid = np.isfinite(row_distances) & np.isfinite(model.failure_residuals)
        if not valid.any():
            return CentroidCorrection(0.0, 0, nearest_cluster, min_distance, 0.0, False, math.nan, math.nan)
        valid_distances = row_distances[valid]
        valid_residuals = model.failure_residuals[valid]
        k = min(spec.k_neighbors, len(valid_distances))
        order = np.argpartition(valid_distances, k - 1)[:k]
        selected_residuals = valid_residuals[order]
        selected_distances = valid_distances[order]
        positive = selected_distances[selected_distances > 1e-9]
        row_scale = float(np.nanmedian(positive)) if len(positive) else model.distance_scale
        if not np.isfinite(row_scale) or row_scale <= 1e-9:
            row_scale = model.distance_scale
        raw = weighted_mean(selected_residuals, np.exp(-selected_distances / row_scale))
    else:
        raise ValueError(f"Unsupported centroid correction mode: {spec.mode}")

    if not np.isfinite(raw):
        return CentroidCorrection(0.0, 0, nearest_cluster, min_distance, 0.0, False, math.nan, math.nan)
    shrink = len(selected_residuals) / (len(selected_residuals) + float(spec.shrinkage))
    correction = float(np.clip(raw * shrink, -spec.correction_clip_c, spec.correction_clip_c))

    positive_distances = selected_distances[selected_distances > 1e-9]
    row_scale = float(np.nanmedian(positive_distances)) if len(positive_distances) else model.distance_scale
    if not np.isfinite(row_scale) or row_scale <= 1e-9:
        row_scale = model.distance_scale
    local_weights = np.exp(-selected_distances / row_scale)
    local_anchor_mae = weighted_mean(np.abs(selected_residuals), local_weights)
    local_corrected_mae = weighted_mean(np.abs(selected_residuals - correction), local_weights)
    gate_passed = bool(
        np.isfinite(local_anchor_mae)
        and np.isfinite(local_corrected_mae)
        and local_corrected_mae <= local_anchor_mae - spec.min_local_mae_improvement_c
    )
    if not gate_passed:
        correction = 0.0
    affinity = float(np.exp(-min_distance / model.distance_scale))
    return CentroidCorrection(
        correction=correction,
        selected_rows=int(len(selected_residuals)),
        nearest_cluster=nearest_cluster,
        min_centroid_distance=min_distance,
        centroid_affinity=affinity,
        gate_passed=gate_passed,
        local_anchor_mae=float(local_anchor_mae),
        local_corrected_mae=float(local_corrected_mae),
    )


def past_only_cluster_centroid_predictions(
    frame: pd.DataFrame,
    spec: ClusterCentroidSpec,
    features: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True).copy()
    dates = pd.to_datetime(ordered["target_date"], errors="coerce").dt.normalize()
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    sources = ordered["forecast_source_family"].astype(str).to_numpy()
    target = pd.to_numeric(ordered["target_tmax_c"], errors="coerce").to_numpy(dtype=float)
    official = pd.to_numeric(ordered["official_raw"], errors="coerce").to_numpy(dtype=float)
    anchor = pd.to_numeric(ordered[spec.anchor_col], errors="coerce").to_numpy(dtype=float)
    feature_matrix = finite_feature_matrix(ordered, features)

    predictions: list[float] = []
    corrections: list[float] = []
    model_prior_rows: list[int] = []
    model_failure_rows: list[int] = []
    selected_rows: list[int] = []
    nearest_clusters: list[int] = []
    min_distances: list[float] = []
    affinities: list[float] = []
    gate_passed: list[bool] = []
    local_anchor_maes: list[float] = []
    local_corrected_maes: list[float] = []
    failure_thresholds: list[float] = []
    model_cache: dict[tuple[pd.Timestamp, str], ClusterCentroidModel | None] = {}
    model_rows: list[dict[str, object]] = []

    for index, target_date in enumerate(date_values):
        fallback = float(anchor[index]) if np.isfinite(anchor[index]) else math.nan
        if not np.isfinite(anchor[index]) or not np.isfinite(feature_matrix[index]).all():
            predictions.append(fallback)
            corrections.append(0.0)
            model_prior_rows.append(0)
            model_failure_rows.append(0)
            selected_rows.append(0)
            nearest_clusters.append(-1)
            min_distances.append(math.nan)
            affinities.append(math.nan)
            gate_passed.append(False)
            local_anchor_maes.append(math.nan)
            local_corrected_maes.append(math.nan)
            failure_thresholds.append(math.nan)
            continue

        period_start = quarter_start(pd.Timestamp(target_date))
        source_key = sources[index] if spec.same_source else "__all__"
        cache_key = (period_start, source_key)
        if cache_key not in model_cache:
            prior_mask = dates < period_start
            if spec.same_source:
                prior_mask &= ordered["forecast_source_family"].astype(str).eq(source_key)
            model = build_cluster_centroid_model(
                ordered,
                features=features,
                spec=spec,
                prior_mask=prior_mask.to_numpy(),
            )
            model_cache[cache_key] = model
            model_rows.append(
                {
                    "period_start": str(period_start.date()),
                    "source_key": source_key,
                    "model_available": model is not None,
                    "prior_rows": 0 if model is None else model.prior_rows,
                    "failure_rows": 0 if model is None else model.failure_rows,
                    "n_clusters": 0 if model is None else int(len(model.cluster_rows)),
                    "failure_threshold_c": math.nan if model is None else model.failure_threshold_c,
                    "gate_distance": math.nan if model is None else model.gate_distance,
                    "distance_scale": math.nan if model is None else model.distance_scale,
                }
            )
        model = model_cache[cache_key]
        if model is None:
            predictions.append(fallback)
            corrections.append(0.0)
            model_prior_rows.append(0)
            model_failure_rows.append(0)
            selected_rows.append(0)
            nearest_clusters.append(-1)
            min_distances.append(math.nan)
            affinities.append(math.nan)
            gate_passed.append(False)
            local_anchor_maes.append(math.nan)
            local_corrected_maes.append(math.nan)
            failure_thresholds.append(math.nan)
            continue

        current_scaled = scale_features(feature_matrix[index], model.means, model.stds)
        result = correction_from_model(model, current_scaled, spec=spec)
        predictions.append(float(anchor[index] + result.correction))
        corrections.append(result.correction)
        model_prior_rows.append(model.prior_rows)
        model_failure_rows.append(model.failure_rows)
        selected_rows.append(result.selected_rows)
        nearest_clusters.append(result.nearest_cluster)
        min_distances.append(result.min_centroid_distance)
        affinities.append(result.centroid_affinity)
        gate_passed.append(result.gate_passed)
        local_anchor_maes.append(result.local_anchor_mae)
        local_corrected_maes.append(result.local_corrected_mae)
        failure_thresholds.append(model.failure_threshold_c)

    out = ordered[["target_date", "forecast_source_family", "primary_regime"]].copy()
    out["target_tmax_c"] = target
    out["official_raw"] = official
    out["anchor_prediction_c"] = anchor
    out["candidate_prediction_c"] = predictions
    out["residual_correction_c"] = corrections
    out["model_prior_rows"] = model_prior_rows
    out["model_failure_rows"] = model_failure_rows
    out["selected_rows"] = selected_rows
    out["nearest_cluster"] = nearest_clusters
    out["min_centroid_distance"] = min_distances
    out["centroid_affinity"] = affinities
    out["do_no_harm_gate_passed"] = gate_passed
    out["local_anchor_mae"] = local_anchor_maes
    out["local_corrected_mae"] = local_corrected_maes
    out["failure_threshold_c"] = failure_thresholds
    return out, pd.DataFrame(model_rows)


def build_specs() -> list[ClusterCentroidSpec]:
    specs: list[ClusterCentroidSpec] = []
    for anchor_col in ("prediction_0018_c", "prediction_0026_c"):
        for mode in ("centroid_mean", "failure_neighbor"):
            for same_source in (False, True):
                for failure_quantile in (0.70, 0.75):
                    for n_clusters in (4, 8):
                        specs.append(
                            ClusterCentroidSpec(
                                anchor_col=anchor_col,
                                mode=mode,
                                same_source=same_source,
                                failure_quantile=failure_quantile,
                                n_clusters=n_clusters,
                            )
                        )
    return specs


def score_candidate(predictions: pd.DataFrame, spec: ClusterCentroidSpec, candidate_id: str) -> dict[str, object]:
    candidate = score_prediction_frame(predictions.rename(columns={"candidate_prediction_c": "prediction"}), "prediction")
    anchor = score_prediction_frame(predictions.rename(columns={"anchor_prediction_c": "prediction"}), "prediction")
    official = score_prediction_frame(predictions.rename(columns={"official_raw": "prediction"}), "prediction")
    corrected = predictions["do_no_harm_gate_passed"].astype(bool)
    return {
        "candidate_id": candidate_id,
        "anchor_col": spec.anchor_col,
        "mode": spec.mode,
        "same_source": spec.same_source,
        "failure_quantile": spec.failure_quantile,
        "n_clusters": spec.n_clusters,
        "k_neighbors": spec.k_neighbors,
        **candidate,
        "anchor_same_rows_mae": anchor["mae"],
        "anchor_same_rows_rmse": anchor["rmse"],
        "delta_vs_anchor_same_rows": float(candidate["mae"] - anchor["mae"]),
        "official_same_rows_mae": official["mae"],
        "official_same_rows_rmse": official["rmse"],
        "delta_vs_official_same_rows": float(candidate["mae"] - official["mae"]),
        "corrected_rows": int(corrected.sum()),
        "fallback_rows": int((~corrected).sum()),
        "mean_model_failure_rows": float(predictions.loc[corrected, "model_failure_rows"].mean()) if corrected.any() else 0.0,
        "mean_selected_rows": float(predictions.loc[corrected, "selected_rows"].mean()) if corrected.any() else 0.0,
        "mean_abs_correction_c": float(predictions.loc[corrected, "residual_correction_c"].abs().mean())
        if corrected.any()
        else 0.0,
        "mean_centroid_affinity": float(predictions.loc[corrected, "centroid_affinity"].mean())
        if corrected.any()
        else math.nan,
        "mean_local_anchor_mae": float(predictions.loc[corrected, "local_anchor_mae"].mean())
        if corrected.any()
        else math.nan,
        "mean_local_corrected_mae": float(predictions.loc[corrected, "local_corrected_mae"].mean())
        if corrected.any()
        else math.nan,
    }


def run_centroid_screen(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = available_numeric_features(frame, diagnostic_feature_candidates())
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    spec_rows: list[dict[str, object]] = []
    model_frames: list[pd.DataFrame] = []
    for spec in build_specs():
        candidate_id = candidate_id_for_spec(spec)
        predictions, models = past_only_cluster_centroid_predictions(frame, spec, features)
        predictions["candidate_id"] = candidate_id
        predictions["anchor_col"] = spec.anchor_col
        predictions["mode"] = spec.mode
        predictions["same_source"] = spec.same_source
        predictions["failure_quantile"] = spec.failure_quantile
        predictions["n_clusters"] = spec.n_clusters
        if not models.empty:
            models["candidate_id"] = candidate_id
        prediction_frames.append(predictions)
        model_frames.append(models)
        score_rows.append(score_candidate(predictions, spec, candidate_id))
        spec_rows.append(
            {
                "candidate_id": candidate_id,
                "anchor_col": spec.anchor_col,
                "mode": spec.mode,
                "same_source": spec.same_source,
                "failure_quantile": spec.failure_quantile,
                "n_clusters": spec.n_clusters,
                "k_neighbors": spec.k_neighbors,
                "features": ",".join(features),
                "feature_count": len(features),
            }
        )
    scoreboard = pd.DataFrame(score_rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    specs = pd.DataFrame(spec_rows)
    model_summary = pd.concat(model_frames, ignore_index=True) if model_frames else pd.DataFrame()
    return scoreboard, predictions, specs, model_summary


def mode_summary(scoreboard: pd.DataFrame) -> pd.DataFrame:
    if scoreboard.empty:
        return pd.DataFrame()
    return (
        scoreboard.groupby(["anchor_col", "mode", "same_source"], observed=True)
        .agg(
            candidates=("candidate_id", "count"),
            best_mae=("mae", "min"),
            best_rmse=("rmse", "min"),
            best_delta_vs_anchor=("delta_vs_anchor_same_rows", "min"),
            best_delta_vs_official=("delta_vs_official_same_rows", "min"),
            max_corrected_rows=("corrected_rows", "max"),
            median_corrected_rows=("corrected_rows", "median"),
        )
        .reset_index()
        .sort_values(["best_mae", "best_rmse"])
    )


def build_blend_frame(
    frame: pd.DataFrame,
    scoreboard: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = frame[["target_date", "forecast_source_family", "target_tmax_c", "official_raw"]].copy()
    if scoreboard.empty or predictions.empty:
        return base, pd.DataFrame()
    top_ids = scoreboard.head(TOP_BLEND_EXPERTS)["candidate_id"].to_list()
    mapping = scoreboard[scoreboard["candidate_id"].isin(top_ids)][
        [
            "candidate_id",
            "anchor_col",
            "mode",
            "same_source",
            "failure_quantile",
            "n_clusters",
            "k_neighbors",
            "mae",
            "rmse",
            "delta_vs_official_same_rows",
            "delta_vs_anchor_same_rows",
            "corrected_rows",
        ]
    ].copy()
    mapping["expert_id"] = [
        f"cc_{rank:02d}_{slug(row.candidate_id, limit=46)}"
        for rank, row in enumerate(mapping.itertuples(index=False), start=1)
    ]
    long = predictions[predictions["candidate_id"].isin(top_ids)][
        ["target_date", "candidate_id", "candidate_prediction_c"]
    ].copy()
    long = long.merge(mapping[["candidate_id", "expert_id"]], on="candidate_id", how="inner")
    wide = (
        long.pivot_table(index="target_date", columns="expert_id", values="candidate_prediction_c", aggfunc="last")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return base.merge(wide, on="target_date", how="left").sort_values("target_date").reset_index(drop=True), mapping


def run_blend_screen(
    frame: pd.DataFrame,
    scoreboard: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blend_frame, mapping = build_blend_frame(frame, scoreboard, predictions)
    if mapping.empty:
        return pd.DataFrame(), pd.DataFrame(), mapping
    experts = ["official_raw", *[column for column in blend_frame.columns if column.startswith("cc_")]]
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for mode in ("best", "inverse_mae"):
        for same_source in (False, True):
            candidate_id = f"cluster_centroid_blend_{mode}_{'same_source' if same_source else 'all_prior'}"
            blend_predictions = past_only_expert_blend(
                blend_frame,
                experts=experts,
                mode=mode,
                same_source=same_source,
                min_history=MIN_HISTORY,
            )
            blend_predictions["candidate_id"] = candidate_id
            candidate = score_prediction_frame(
                blend_predictions.rename(columns={"expert_prediction_c": "prediction"}), "prediction"
            )
            official = score_prediction_frame(blend_predictions.rename(columns={"official_raw": "prediction"}), "prediction")
            score_rows.append(
                {
                    "candidate_id": candidate_id,
                    "mode": mode,
                    "same_source": same_source,
                    **candidate,
                    "official_same_rows_mae": official["mae"],
                    "official_same_rows_rmse": official["rmse"],
                    "delta_vs_official_same_rows": float(candidate["mae"] - official["mae"]),
                    "fallback_rows": int(blend_predictions["selected_expert"].eq("official_raw_fallback").sum()),
                }
            )
            prediction_frames.append(blend_predictions)
    blend_scoreboard = pd.DataFrame(score_rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    return blend_scoreboard, pd.concat(prediction_frames, ignore_index=True), mapping


def baseline_comparison(frame: pd.DataFrame, scoreboard: pd.DataFrame, blend_scoreboard: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prior_paths = [
        (
            "0033_best_smooth_archetype",
            RESEARCH_ROOT / "0033_smooth_residual_archetype_specialists" / "artifacts" / "smooth_scoreboard.csv",
        ),
        (
            "0033_best_smooth_blend",
            RESEARCH_ROOT / "0033_smooth_residual_archetype_specialists" / "artifacts" / "blend_scoreboard.csv",
        ),
        (
            "0032_best_archetype",
            RESEARCH_ROOT / "0032_residual_failure_cluster_discovery" / "artifacts" / "archetype_scoreboard.csv",
        ),
        (
            "0031_best_selector",
            RESEARCH_ROOT / "0031_regime_gated_specialist_selector" / "artifacts" / "selector_scoreboard.csv",
        ),
    ]
    for system, path in prior_paths:
        if not path.exists():
            continue
        table = pd.read_csv(path)
        if table.empty:
            continue
        best = table.sort_values(["mae", "rmse"]).iloc[0]
        rows.append(
            {
                "system": system,
                "candidate_id": str(best["candidate_id"]),
                "mae": float(best["mae"]),
                "rmse": float(best["rmse"]),
                "delta_vs_official": float(best.get("delta_vs_official_same_rows", math.nan)),
                "n": int(best.get("n", 0)),
                "first_date": str(best.get("first_date", "")),
                "last_date": str(best.get("last_date", "")),
            }
        )
    for system, col in [
        ("official_raw", "official_raw"),
        ("0018_official_expert_blend", "prediction_0018_c"),
        ("0026_pressure_gradient_blend", "prediction_0026_c"),
    ]:
        score = score_prediction_frame(frame.rename(columns={col: "prediction"}), "prediction")
        official = score_prediction_frame(frame.rename(columns={"official_raw": "prediction"}), "prediction")
        rows.append(
            {
                "system": system,
                "candidate_id": system,
                "mae": score["mae"],
                "rmse": score["rmse"],
                "delta_vs_official": float(score["mae"] - official["mae"]),
                "n": score["n"],
                "first_date": score["first_date"],
                "last_date": score["last_date"],
            }
        )
    if not scoreboard.empty:
        best = scoreboard.iloc[0]
        rows.append(
            {
                "system": "0034_best_cluster_centroid",
                "candidate_id": str(best["candidate_id"]),
                "mae": float(best["mae"]),
                "rmse": float(best["rmse"]),
                "delta_vs_official": float(best["delta_vs_official_same_rows"]),
                "n": int(best["n"]),
                "first_date": str(best["first_date"]),
                "last_date": str(best["last_date"]),
            }
        )
    if not blend_scoreboard.empty:
        best = blend_scoreboard.iloc[0]
        rows.append(
            {
                "system": "0034_best_cluster_centroid_blend",
                "candidate_id": str(best["candidate_id"]),
                "mae": float(best["mae"]),
                "rmse": float(best["rmse"]),
                "delta_vs_official": float(best["delta_vs_official_same_rows"]),
                "n": int(best["n"]),
                "first_date": str(best["first_date"]),
                "last_date": str(best["last_date"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)


def write_readme(
    *,
    folder: Path,
    manifest: dict[str, object],
    scoreboard: pd.DataFrame,
    summary: pd.DataFrame,
    blend_scoreboard: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    best_single = scoreboard.iloc[0] if not scoreboard.empty else None
    best_blend = blend_scoreboard.iloc[0] if not blend_scoreboard.empty else None
    best_single_text = "No cluster-centroid specialist was scoreable."
    if best_single is not None:
        best_single_text = (
            f"Best cluster-centroid specialist: `{best_single['candidate_id']}` with MAE "
            f"`{best_single['mae']:.4f}`, RMSE `{best_single['rmse']:.4f}`, official delta "
            f"`{best_single['delta_vs_official_same_rows']:.4f}`, and anchor delta "
            f"`{best_single['delta_vs_anchor_same_rows']:.4f}`."
        )
    best_blend_text = "No cluster-centroid blend was scoreable."
    if best_blend is not None:
        best_blend_text = (
            f"Best cluster-centroid blend: `{best_blend['candidate_id']}` with MAE `{best_blend['mae']:.4f}`, "
            f"RMSE `{best_blend['rmse']:.4f}`, and official delta "
            f"`{best_blend['delta_vs_official_same_rows']:.4f}`."
        )

    readme = f"""# Cluster-Centroid Soft Gating

Generated: `{manifest['generated_at_utc']}`

## Purpose

`0032` identified recurring large-miss clusters, and `0033` showed that manually named smooth archetypes can improve the current champion. This insight tests the next step: use failure-cluster centroids themselves as soft gates.

The implementation intentionally does **not** reuse full-sample 0032 centroids for scoring. That would leak future large-miss labels into earlier predictions. Instead, each calendar quarter builds centroids from rows strictly before that quarter starts. Current rows then receive a soft affinity to those prior-only failure centroids.

## Data Window

Rows used: `{manifest['official_rows']}` official forecast/target rows.

Date range: `{manifest['first_target_date']}` to `{manifest['last_target_date']}`.

Source counts: `{manifest['source_counts']}`.

## Leakage Contract

- All scored rows are earlier than `{CONFIRMATION_START.date()}`.
- Each centroid model is fit only on rows before the current row's quarter begins.
- Failure thresholds are computed from the prior slice only.
- Feature scaling, k-means centroids, centroid distances, residual corrections, do-no-harm gates, and blend weights are all fold-local.
- Same-source variants restrict the centroid model to the same official forecast source family.
- 2024+ confirmation labels are not loaded or scored.

## Main Results

{best_single_text}

{best_blend_text}

## Baseline Comparison

{markdown_table(comparison, max_rows=20)}

## Mode Summary

{markdown_table(summary, max_rows=30)}

## Cluster-Centroid Scoreboard

{markdown_table(scoreboard.head(30), max_rows=30)}

## Blend Scoreboard

{markdown_table(blend_scoreboard, max_rows=20)}

## Interpretation

This run tests whether observed failure-cluster geometry is more useful than manually named archetype gates. If it beats `0033`, the path is to promote centroid affinity as a first-class residual feature. If it does not, the result is still useful: it says that broad failure-centroid similarity is too blunt and the stronger route is targeted forecast-revision specialists plus additional source/vintage features.
"""
    write_text(folder / "README.md", readme)


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Cluster-Centroid Soft Gating\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_cluster_centroid_soft_gating.py`:

- `{FOLDER_NAME}`: fold-local failure-cluster centroid specialists using quarterly prior-only k-means, soft centroid affinity, and do-no-harm residual gates.

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Centroid candidates | {manifest['centroid_candidates']} |
| Blend candidates | {manifest['blend_candidates']} |
| Best centroid MAE | {manifest['best_centroid_mae']} |
| Best centroid RMSE | {manifest['best_centroid_rmse']} |
| Best centroid delta vs official | {manifest['best_centroid_delta_vs_official']} |
| Best centroid delta vs anchor | {manifest['best_centroid_delta_vs_anchor']} |
| Best blend MAE | {manifest['best_blend_mae']} |
| Best blend RMSE | {manifest['best_blend_rmse']} |
| Best blend delta vs official | {manifest['best_blend_delta_vs_official']} |

Leakage contract: all scored rows are before `{CONFIRMATION_START.date()}`; centroid models are refit from rows before the current quarter only; thresholds, scaling, k-means centroids, corrections, do-no-harm checks, and blend weights use strictly prior information.
"""
    write_text(index_path, text)


def write_outputs(
    *,
    frame: pd.DataFrame,
    scoreboard: pd.DataFrame,
    predictions: pd.DataFrame,
    specs: pd.DataFrame,
    model_summary: pd.DataFrame,
    blend_scoreboard: pd.DataFrame,
    blend_predictions: pd.DataFrame,
    blend_mapping: pd.DataFrame,
) -> dict[str, object]:
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    summary = mode_summary(scoreboard)
    comparison = baseline_comparison(frame, scoreboard, blend_scoreboard)
    top_ids = set(scoreboard.head(40)["candidate_id"].to_list()) if not scoreboard.empty else set()

    write_csv(artifacts / "centroid_specs.csv", specs)
    write_csv(artifacts / "centroid_scoreboard.csv", scoreboard)
    write_csv(artifacts / "mode_summary.csv", summary)
    write_csv(artifacts / "model_summary.csv", model_summary)
    write_csv(
        artifacts / "top_centroid_predictions.csv",
        predictions[predictions["candidate_id"].isin(top_ids)].copy() if not predictions.empty else predictions,
    )
    write_csv(artifacts / "blend_scoreboard.csv", blend_scoreboard)
    write_csv(artifacts / "blend_predictions.csv", blend_predictions)
    write_csv(artifacts / "blend_mapping.csv", blend_mapping)
    write_csv(artifacts / "baseline_comparison.csv", comparison)

    best_single = scoreboard.iloc[0] if not scoreboard.empty else None
    best_blend = blend_scoreboard.iloc[0] if not blend_scoreboard.empty else None
    manifest = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "official_rows": int(len(frame)),
        "first_target_date": str(frame["target_date"].min().date()),
        "last_target_date": str(frame["target_date"].max().date()),
        "source_counts": {str(k): int(v) for k, v in frame["forecast_source_family"].value_counts().to_dict().items()},
        "centroid_candidates": int(len(scoreboard)),
        "blend_candidates": int(len(blend_scoreboard)),
        "best_centroid": "" if best_single is None else str(best_single["candidate_id"]),
        "best_centroid_mae": None if best_single is None else float(best_single["mae"]),
        "best_centroid_rmse": None if best_single is None else float(best_single["rmse"]),
        "best_centroid_delta_vs_official": None
        if best_single is None
        else float(best_single["delta_vs_official_same_rows"]),
        "best_centroid_delta_vs_anchor": None if best_single is None else float(best_single["delta_vs_anchor_same_rows"]),
        "best_blend": "" if best_blend is None else str(best_blend["candidate_id"]),
        "best_blend_mae": None if best_blend is None else float(best_blend["mae"]),
        "best_blend_rmse": None if best_blend is None else float(best_blend["rmse"]),
        "best_blend_delta_vs_official": None
        if best_blend is None
        else float(best_blend["delta_vs_official_same_rows"]),
        "confirmation_start": str(CONFIRMATION_START.date()),
    }
    write_json(RESEARCH_ROOT / "cluster_centroid_soft_gating_manifest.json", manifest)
    write_readme(
        folder=folder,
        manifest=manifest,
        scoreboard=scoreboard,
        summary=summary,
        blend_scoreboard=blend_scoreboard,
        comparison=comparison,
    )
    update_master_index(manifest)
    return manifest


def run() -> dict[str, object]:
    frame, _prior_systems = build_failure_frame()
    require_no_confirmation_dates(frame["target_date"], context="cluster-centroid soft gating")
    scoreboard, predictions, specs, model_summary = run_centroid_screen(frame)
    blend_scoreboard, blend_predictions, blend_mapping = run_blend_screen(frame, scoreboard, predictions)
    return write_outputs(
        frame=frame,
        scoreboard=scoreboard,
        predictions=predictions,
        specs=specs,
        model_summary=model_summary,
        blend_scoreboard=blend_scoreboard,
        blend_predictions=blend_predictions,
        blend_mapping=blend_mapping,
    )


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 cluster-centroid soft gating.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
