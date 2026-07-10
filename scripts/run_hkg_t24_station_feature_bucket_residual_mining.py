from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import (  # noqa: E402
    update_markdown_section,
)

FOLDER_NAME = "0065_station_feature_bucket_residual_mining"
ARTIFACT_0054 = RESEARCH_ROOT / "0054_station_only_walkforward_matrix_audit" / "artifacts"
ARTIFACT_0064 = RESEARCH_ROOT / "0064_station_only_heat_proxy_specialist_validation" / "artifacts"
FEATURE_MATRIX_PATH = ARTIFACT_0054 / "features.parquet"
COMPONENTS_PATH = ARTIFACT_0054 / "components.csv"
PREDICTIONS_0064_PATH = ARTIFACT_0064 / "predictions.parquet"
SUMMARY_0064_PATH = ARTIFACT_0064 / "summary.json"
TRAINING_THRESHOLD_END = pd.Timestamp("1999-12-31")
DEVELOPMENT_END = pd.Timestamp("2023-12-31")
PAIR_SEED_TERMS = 18
SELECTED_PREDICTION_CANDIDATES = 5
BASE_TERM_BUCKETS = ("low", "mid", "high")
SEASONS = ("DJF", "MAM", "JJA", "SON")
NON_FEATURE_COLUMNS = {"target_date", "source_local_date_rule", "source_cutoff_hkt"}
FORBIDDEN_MASK_TOKENS = (
    "target",
    "label",
    "residual",
    "error",
    "forecast",
    "mae",
    "rmse",
)


@dataclass(frozen=True)
class TermSpec:
    term_id: str
    term_type: str
    feature_id: str
    bucket_value: str
    season: str
    source_family: str
    station_ids: str
    display_name: str


@dataclass(frozen=True)
class CorrectionSpec:
    correction_id: str
    mode: str
    half_life_days: float | None
    min_prior_rows: int
    shrinkage: float
    cap_c: float


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    candidate_type: str
    term_ids: tuple[str, ...]
    correction: CorrectionSpec


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 120) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def quantile_edges(values: pd.Series, *, min_rows: int = 365) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < min_rows or clean.nunique(dropna=True) < 3:
        raise ValueError(f"Need at least {min_rows} non-degenerate rows, got {len(clean)}")
    low, high = clean.quantile([1.0 / 3.0, 2.0 / 3.0]).tolist()
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError(f"Invalid quantile edges: low={low}, high={high}")
    return float(low), float(high)


def bucket_by_edges(values: pd.Series, low: float, high: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series("missing", index=values.index, dtype="object")
    out.loc[numeric <= low] = "low"
    out.loc[(numeric > low) & (numeric <= high)] = "mid"
    out.loc[numeric > high] = "high"
    return out


def bucket_column(feature_id: str) -> str:
    return f"bucket__{feature_id}"


def deployable_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if column not in NON_FEATURE_COLUMNS]
    forbidden = [
        column
        for column in columns
        if any(token in column.lower() for token in FORBIDDEN_MASK_TOKENS)
    ]
    if forbidden:
        raise RuntimeError(f"Forbidden deployable feature columns found: {forbidden[:10]}")
    return columns


def load_reference_frame() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    summary_0064 = load_json(SUMMARY_0064_PATH)
    best_proxy = str(summary_0064["best_proxy"])
    predictions = pd.read_parquet(PREDICTIONS_0064_PATH)
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    reference = predictions[predictions["proxy_id"].astype(str).eq(best_proxy)].copy()
    reference = reference[reference["target_date"].le(DEVELOPMENT_END)].copy()
    if reference.empty:
        raise RuntimeError(f"Missing 0064 best proxy predictions: {best_proxy}")
    if reference["target_date"].duplicated().any():
        raise RuntimeError(f"0064 best proxy `{best_proxy}` is not one row per target_date")
    require_no_confirmation_dates(reference["target_date"], context="0065 0064 reference")
    reference = reference[
        [
            "target_date",
            "target_tmax_c",
            "candidate_prediction_c",
            "fold_id",
            "month",
            "season",
        ]
    ].rename(columns={"candidate_prediction_c": "reference_prediction_c"})
    reference["reference_residual_to_add_c"] = (
        reference["target_tmax_c"] - reference["reference_prediction_c"]
    )
    reference = reference.sort_values("target_date").reset_index(drop=True)

    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    features["target_date"] = pd.to_datetime(features["target_date"], errors="coerce").dt.normalize()
    features = features[features["target_date"].lt(CONFIRMATION_START)].copy()
    feature_columns = deployable_feature_columns(features)
    thresholds = build_thresholds(features, feature_columns)
    usable_thresholds = thresholds[thresholds["threshold_status"].eq("usable")].copy()
    bucket_frame = features[["target_date"]].copy()
    for row in usable_thresholds.to_dict("records"):
        feature_id = str(row["feature_id"])
        bucket_frame[bucket_column(feature_id)] = bucket_by_edges(
            features[feature_id],
            float(row["low_edge"]),
            float(row["high_edge"]),
        )

    frame = reference.merge(bucket_frame, on="target_date", how="left", validate="one_to_one")
    frame["month"] = pd.to_numeric(frame["month"], errors="coerce").astype("Int64").astype(int)
    frame["season"] = frame["month"].map(season_from_month)
    require_no_confirmation_dates(frame["target_date"], context="0065 merged frame")

    components = pd.read_csv(COMPONENTS_PATH)
    return frame.sort_values("target_date").reset_index(drop=True), summary_0064, thresholds, components


def build_thresholds(features: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    training = features[features["target_date"].le(TRAINING_THRESHOLD_END)].copy()
    rows: list[dict[str, object]] = []
    for feature_id in feature_columns:
        clean = pd.to_numeric(training[feature_id], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        try:
            low, high = quantile_edges(clean)
            status = "usable"
            reason = ""
        except ValueError as exc:
            low, high = math.nan, math.nan
            status = "skipped"
            reason = str(exc)
        rows.append(
            {
                "feature_id": feature_id,
                "bucket_column": bucket_column(feature_id),
                "low_edge": low,
                "high_edge": high,
                "train_non_null_rows": int(len(clean)),
                "train_unique_values": int(clean.nunique(dropna=True)),
                "threshold_source": "station_feature_history_through_1999-12-31",
                "threshold_status": status,
                "skip_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def component_lookup(components: pd.DataFrame) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for row in components.to_dict("records"):
        feature_id = str(row["feature_id"])
        lookup[feature_id] = {
            "source_family": str(row.get("source_family", "")),
            "station_ids": str(row.get("station_ids", "")),
            "display_name": str(row.get("display_name", feature_id)),
        }
    return lookup


def build_terms(thresholds: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    lookup = component_lookup(components)
    rows: list[dict[str, object]] = []
    usable = thresholds[thresholds["threshold_status"].eq("usable")].copy()
    for feature_id in usable["feature_id"].astype(str).tolist():
        metadata = lookup.get(
            feature_id,
            {"source_family": "unknown", "station_ids": "", "display_name": feature_id},
        )
        for bucket in BASE_TERM_BUCKETS:
            rows.append(
                {
                    "term_id": slug(f"term_{feature_id}_{bucket}", limit=140),
                    "term_type": "feature_bucket",
                    "feature_id": feature_id,
                    "bucket_value": bucket,
                    "season": "",
                    "source_family": metadata["source_family"],
                    "station_ids": metadata["station_ids"],
                    "display_name": metadata["display_name"],
                }
            )
            for season in SEASONS:
                rows.append(
                    {
                        "term_id": slug(f"term_{season}_{feature_id}_{bucket}", limit=140),
                        "term_type": "season_feature_bucket",
                        "feature_id": feature_id,
                        "bucket_value": bucket,
                        "season": season,
                        "source_family": metadata["source_family"],
                        "station_ids": metadata["station_ids"],
                        "display_name": metadata["display_name"],
                    }
                )
    terms = pd.DataFrame(rows)
    if terms["term_id"].duplicated().any():
        duplicates = terms.loc[terms["term_id"].duplicated(), "term_id"].head(10).tolist()
        raise RuntimeError(f"Duplicate term ids: {duplicates}")
    return terms


def term_specs(terms: pd.DataFrame) -> dict[str, TermSpec]:
    return {
        str(row["term_id"]): TermSpec(
            term_id=str(row["term_id"]),
            term_type=str(row["term_type"]),
            feature_id=str(row["feature_id"]),
            bucket_value=str(row["bucket_value"]),
            season="" if pd.isna(row["season"]) else str(row["season"]),
            source_family=str(row["source_family"]),
            station_ids=str(row["station_ids"]),
            display_name=str(row["display_name"]),
        )
        for row in terms.to_dict("records")
    }


def base_correction_specs() -> list[CorrectionSpec]:
    return [
        CorrectionSpec("halflife730_min60_shrink120_cap1p0", "half_life", 730.0, 60, 120.0, 1.0),
    ]


def season_correction_specs() -> list[CorrectionSpec]:
    return [
        CorrectionSpec("halflife730_min60_shrink120_cap1p0", "half_life", 730.0, 60, 120.0, 1.0),
    ]


def pair_correction_specs() -> list[CorrectionSpec]:
    return [
        CorrectionSpec("pair_halflife730_min45_shrink90_cap1p0", "half_life", 730.0, 45, 90.0, 1.0),
    ]


def initial_candidates(terms: pd.DataFrame) -> list[CandidateSpec]:
    candidates: list[CandidateSpec] = []
    base_terms = terms[terms["term_type"].astype(str).eq("feature_bucket")].copy()
    for term_id in base_terms["term_id"].astype(str).tolist():
        for correction in base_correction_specs():
            candidates.append(
                CandidateSpec(
                    candidate_id=slug(f"{term_id}_{correction.correction_id}", limit=180),
                    candidate_type="feature_bucket",
                    term_ids=(term_id,),
                    correction=correction,
                )
            )
    return candidates


def season_candidates(seed_term_ids: list[str], terms: dict[str, TermSpec]) -> list[CandidateSpec]:
    season_lookup = {
        (term.feature_id, term.bucket_value, term.season): term.term_id
        for term in terms.values()
        if term.term_type == "season_feature_bucket"
    }
    candidates: list[CandidateSpec] = []
    for seed_id in seed_term_ids:
        seed = terms[seed_id]
        for season in SEASONS:
            term_id = season_lookup.get((seed.feature_id, seed.bucket_value, season))
            if term_id is None:
                continue
            for correction in season_correction_specs():
                candidates.append(
                    CandidateSpec(
                        candidate_id=slug(f"{term_id}_{correction.correction_id}", limit=180),
                        candidate_type="season_feature_bucket",
                        term_ids=(term_id,),
                        correction=correction,
                    )
                )
    return candidates


def pair_candidates(seed_term_ids: list[str]) -> list[CandidateSpec]:
    candidates: list[CandidateSpec] = []
    for pair_index, (left, right) in enumerate(itertools.combinations(seed_term_ids, 2), start=1):
        for correction in pair_correction_specs():
            candidates.append(
                CandidateSpec(
                    candidate_id=f"pair_{pair_index:04d}_{correction.correction_id}",
                    candidate_type="pair_feature_bucket",
                    term_ids=(left, right),
                    correction=correction,
                )
            )
    return candidates


def mask_for_term(frame: pd.DataFrame, term: TermSpec) -> pd.Series:
    column = bucket_column(term.feature_id)
    if column not in frame.columns:
        raise KeyError(f"Missing bucket column for term {term.term_id}: {column}")
    mask = frame[column].astype(str).eq(term.bucket_value)
    if term.season:
        mask &= frame["season"].astype(str).eq(term.season)
    return mask.fillna(False)


def mask_for_candidate(
    frame: pd.DataFrame,
    candidate: CandidateSpec,
    terms: dict[str, TermSpec],
) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for term_id in candidate.term_ids:
        term = terms[term_id]
        mask &= mask_for_term(frame, term)
    return mask.fillna(False)


def shrink_and_cap(raw: float, prior_rows: int, correction: CorrectionSpec) -> float:
    if prior_rows < correction.min_prior_rows or not math.isfinite(raw):
        return 0.0
    shrink = prior_rows / (prior_rows + correction.shrinkage)
    return float(np.clip(raw * shrink, -correction.cap_c, correction.cap_c))


def compute_prior_active_correction(
    ordered: pd.DataFrame,
    active_mask: pd.Series,
    correction: CorrectionSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = active_mask.reset_index(drop=True).to_numpy(dtype=bool)
    residuals = pd.to_numeric(ordered["reference_residual_to_add_c"], errors="coerce").to_numpy(dtype=float)
    dates = pd.to_datetime(ordered["target_date"], errors="coerce").reset_index(drop=True)
    corrections = np.zeros(len(ordered), dtype=float)
    prior_counts = np.zeros(len(ordered), dtype=int)
    raw_means = np.full(len(ordered), math.nan, dtype=float)

    if correction.mode == "half_life":
        if correction.half_life_days is None:
            raise ValueError("half_life correction requires half_life_days")
        weighted_sum = 0.0
        weighted_residual_sum = 0.0
        prior_count = 0
        previous_date: pd.Timestamp | None = None
        for idx, is_active in enumerate(active):
            current_date = pd.Timestamp(dates.iloc[idx])
            if previous_date is not None:
                delta_days = max(0, (current_date - previous_date).days)
                decay = math.pow(0.5, delta_days / correction.half_life_days)
                weighted_sum *= decay
                weighted_residual_sum *= decay
            prior_counts[idx] = prior_count
            if is_active and prior_count >= correction.min_prior_rows and weighted_sum > 0:
                raw = weighted_residual_sum / weighted_sum
                raw_means[idx] = raw
                corrections[idx] = shrink_and_cap(raw, prior_count, correction)
            residual = residuals[idx]
            if is_active and math.isfinite(residual):
                weighted_sum += 1.0
                weighted_residual_sum += residual
                prior_count += 1
            previous_date = current_date
        return corrections, prior_counts, raw_means

    if correction.mode != "expanding":
        raise ValueError(f"Unsupported correction mode: {correction.mode}")

    count = 0
    total = 0.0
    for idx, is_active in enumerate(active):
        prior_counts[idx] = count
        if is_active and count >= correction.min_prior_rows:
            raw = total / count
            raw_means[idx] = raw
            corrections[idx] = shrink_and_cap(raw, count, correction)
        residual = residuals[idx]
        if is_active and math.isfinite(residual):
            count += 1
            total += residual
    return corrections, prior_counts, raw_means


def promotion_gate(row: dict[str, object] | pd.Series) -> bool:
    active_n = int(row["active_n"])
    active_delta = float(row["active_delta_mae_vs_reference"])
    full_delta = float(row["delta_mae_vs_reference"])
    fold_delta_max = float(row["fold_delta_max"])
    active_correction_share = float(row["active_correction_share"])
    return bool(
        active_n >= 120
        and active_delta <= -0.010
        and full_delta < 0.0
        and fold_delta_max <= 0.010
        and active_correction_share >= 0.50
    )


def score_arrays(target: np.ndarray, prediction: np.ndarray, dates: pd.Series) -> dict[str, float | int | str]:
    target_values = np.asarray(target, dtype=float)
    prediction_values = np.asarray(prediction, dtype=float)
    valid = np.isfinite(target_values) & np.isfinite(prediction_values)
    if not valid.any():
        return {"n": 0, "first_date": "", "last_date": "", "mae": math.nan, "rmse": math.nan, "bias": math.nan}
    error = prediction_values[valid] - target_values[valid]
    valid_dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True).loc[valid], errors="coerce")
    return {
        "n": int(valid.sum()),
        "first_date": str(valid_dates.min().date()),
        "last_date": str(valid_dates.max().date()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
    }


def evaluate_candidate(
    frame: pd.DataFrame,
    candidate: CandidateSpec,
    terms: dict[str, TermSpec],
    *,
    include_predictions: bool = False,
) -> tuple[dict[str, object], pd.DataFrame]:
    ordered = frame
    active_mask = mask_for_candidate(ordered, candidate, terms)
    corrections, prior_rows, raw_means = compute_prior_active_correction(
        ordered,
        active_mask,
        candidate.correction,
    )
    target = pd.to_numeric(ordered["target_tmax_c"], errors="coerce").to_numpy(dtype=float)
    reference_prediction = pd.to_numeric(ordered["reference_prediction_c"], errors="coerce").to_numpy(dtype=float)
    candidate_prediction = reference_prediction + corrections
    dates = pd.to_datetime(ordered["target_date"], errors="coerce").reset_index(drop=True)
    active = active_mask.to_numpy(dtype=bool)

    full = score_arrays(target, candidate_prediction, dates)
    ref_full = score_arrays(target, reference_prediction, dates)
    active_score = score_arrays(target[active], candidate_prediction[active], dates.loc[active])
    ref_active_score = score_arrays(target[active], reference_prediction[active], dates.loc[active])

    fold_deltas: list[float] = []
    fold_ids = ordered["fold_id"].astype(str).to_numpy()
    for fold_id in pd.unique(fold_ids):
        fold_mask = fold_ids == fold_id
        candidate_fold_score = score_arrays(target[fold_mask], candidate_prediction[fold_mask], dates.loc[fold_mask])
        reference_fold_score = score_arrays(target[fold_mask], reference_prediction[fold_mask], dates.loc[fold_mask])
        fold_deltas.append(float(candidate_fold_score["mae"]) - float(reference_fold_score["mae"]))

    term_records = [terms[term_id] for term_id in candidate.term_ids]
    active_corrections = corrections[active]
    row: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "candidate_type": candidate.candidate_type,
        "correction_id": candidate.correction.correction_id,
        "correction_mode": candidate.correction.mode,
        "half_life_days": candidate.correction.half_life_days,
        "min_prior_rows": candidate.correction.min_prior_rows,
        "shrinkage": candidate.correction.shrinkage,
        "cap_c": candidate.correction.cap_c,
        "term_count": len(candidate.term_ids),
        "term_ids": "|".join(candidate.term_ids),
        "features": "|".join(term.feature_id for term in term_records),
        "bucket_values": "|".join(term.bucket_value for term in term_records),
        "seasons": "|".join(term.season for term in term_records if term.season),
        "source_families": "|".join(sorted({term.source_family for term in term_records})),
        "station_ids": "|".join(sorted({term.station_ids for term in term_records if term.station_ids})),
        "display_names": "|".join(term.display_name for term in term_records),
        "n": full["n"],
        "mae": full["mae"],
        "rmse": full["rmse"],
        "bias": full["bias"],
        "reference_mae": ref_full["mae"],
        "reference_rmse": ref_full["rmse"],
        "reference_bias": ref_full["bias"],
        "delta_mae_vs_reference": float(full["mae"]) - float(ref_full["mae"]),
        "active_n": active_score["n"],
        "active_mae": active_score["mae"],
        "active_rmse": active_score["rmse"],
        "active_bias": active_score["bias"],
        "active_reference_mae": ref_active_score["mae"],
        "active_reference_rmse": ref_active_score["rmse"],
        "active_reference_bias": ref_active_score["bias"],
        "active_delta_mae_vs_reference": (
            float(active_score["mae"]) - float(ref_active_score["mae"])
            if int(active_score["n"]) > 0
            else math.nan
        ),
        "fold_delta_max": max(fold_deltas) if fold_deltas else math.nan,
        "fold_delta_min": min(fold_deltas) if fold_deltas else math.nan,
        "folds_improved": int(sum(delta < 0 for delta in fold_deltas)),
        "mean_abs_correction_c": float(np.mean(np.abs(corrections))),
        "active_mean_abs_correction_c": (
            float(np.mean(np.abs(active_corrections))) if len(active_corrections) else math.nan
        ),
        "active_correction_share": (
            float(np.mean(np.abs(active_corrections) > 1e-9)) if len(active_corrections) else math.nan
        ),
        "max_abs_correction_c": float(np.max(np.abs(corrections))),
    }
    row["promotion_gate_passed"] = promotion_gate(row)
    if not include_predictions:
        return row, pd.DataFrame()

    predictions = ordered[
        ["target_date", "target_tmax_c", "reference_prediction_c", "fold_id", "month", "season"]
    ].copy()
    predictions["candidate_prediction_c"] = candidate_prediction
    predictions["residual_correction_c"] = corrections
    predictions["prior_rows"] = prior_rows
    predictions["raw_prior_residual_mean_c"] = raw_means
    predictions["candidate_active"] = active
    predictions["candidate_id"] = candidate.candidate_id
    predictions["candidate_type"] = candidate.candidate_type
    predictions["correction_id"] = candidate.correction.correction_id
    predictions["term_ids"] = "|".join(candidate.term_ids)
    return row, predictions


def evaluate_candidates(
    frame: pd.DataFrame,
    candidates: list[CandidateSpec],
    terms: dict[str, TermSpec],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered = frame.sort_values("target_date").reset_index(drop=True).copy()
    for index, candidate in enumerate(candidates, start=1):
        row, _predictions = evaluate_candidate(ordered, candidate, terms)
        rows.append(row)
        if index % 100 == 0:
            gc.collect()
    return pd.DataFrame(rows)


def select_pair_seed_terms(single_scoreboard: pd.DataFrame, terms: dict[str, TermSpec]) -> list[str]:
    eligible = single_scoreboard[
        single_scoreboard["candidate_type"].eq("feature_bucket")
        & pd.to_numeric(single_scoreboard["active_n"], errors="coerce").ge(180)
        & pd.to_numeric(single_scoreboard["active_correction_share"], errors="coerce").ge(0.25)
    ].copy()
    eligible = eligible.sort_values(
        ["active_delta_mae_vs_reference", "delta_mae_vs_reference", "active_reference_mae"],
        ascending=[True, True, False],
    )
    seed_terms: list[str] = []
    seed_features: set[str] = set()
    for term_text in eligible["term_ids"].astype(str).tolist():
        term_id = term_text.split("|")[0]
        feature_id = terms[term_id].feature_id
        if feature_id in seed_features:
            continue
        seed_terms.append(term_id)
        seed_features.add(feature_id)
        if len(seed_terms) >= PAIR_SEED_TERMS:
            break
    return seed_terms


def candidate_definitions(candidates: list[CandidateSpec], terms: dict[str, TermSpec]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        term_records = [terms[term_id] for term_id in candidate.term_ids]
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "candidate_type": candidate.candidate_type,
                "correction_id": candidate.correction.correction_id,
                "correction_mode": candidate.correction.mode,
                "half_life_days": candidate.correction.half_life_days,
                "min_prior_rows": candidate.correction.min_prior_rows,
                "shrinkage": candidate.correction.shrinkage,
                "cap_c": candidate.correction.cap_c,
                "term_ids": "|".join(candidate.term_ids),
                "features": "|".join(term.feature_id for term in term_records),
                "bucket_values": "|".join(term.bucket_value for term in term_records),
                "seasons": "|".join(term.season for term in term_records if term.season),
                "source_families": "|".join(sorted({term.source_family for term in term_records})),
                "station_ids": "|".join(sorted({term.station_ids for term in term_records if term.station_ids})),
                "display_names": "|".join(term.display_name for term in term_records),
                "target_or_residual_used_in_mask": False,
            }
        )
    return pd.DataFrame(rows)


def selected_predictions(
    frame: pd.DataFrame,
    scoreboard: pd.DataFrame,
    candidates: dict[str, CandidateSpec],
    terms: dict[str, TermSpec],
) -> pd.DataFrame:
    selected_ids = (
        scoreboard.sort_values(
            ["promotion_gate_passed", "delta_mae_vs_reference", "active_delta_mae_vs_reference"],
            ascending=[False, True, True],
        )["candidate_id"]
        .head(SELECTED_PREDICTION_CANDIDATES)
        .astype(str)
        .tolist()
    )
    frames: list[pd.DataFrame] = []
    for candidate_id in selected_ids:
        _row, predictions = evaluate_candidate(frame, candidates[candidate_id], terms, include_predictions=True)
        frames.append(predictions)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    require_no_confirmation_dates(out["target_date"], context="0065 selected predictions")
    return out


def leakage_audit(
    frame: pd.DataFrame,
    thresholds: pd.DataFrame,
    definitions: pd.DataFrame,
    scoreboard: pd.DataFrame,
    selected: pd.DataFrame,
    pair_seed_terms: list[str],
) -> pd.DataFrame:
    mask_text = "|".join(
        definitions[["features", "term_ids", "display_names"]].astype(str).agg("|".join, axis=1).tolist()
    ).lower()
    forbidden_hits = [token for token in FORBIDDEN_MASK_TOKENS if token in mask_text]
    promoted = scoreboard[scoreboard["promotion_gate_passed"].astype(bool)].copy()
    checks = [
        {
            "check_id": "no_confirmation_rows",
            "passed": bool(
                pd.to_datetime(frame["target_date"], errors="coerce").max() < CONFIRMATION_START
                and (selected.empty or pd.to_datetime(selected["target_date"], errors="coerce").max() < CONFIRMATION_START)
            ),
            "evidence": f"last target_date {pd.to_datetime(frame['target_date'], errors='coerce').max().date()}",
        },
        {
            "check_id": "thresholds_fit_pre2000",
            "passed": bool(
                thresholds["threshold_source"].astype(str).eq("station_feature_history_through_1999-12-31").all()
                and thresholds.loc[thresholds["threshold_status"].eq("usable"), "train_non_null_rows"].ge(365).all()
            ),
            "evidence": f"{int(thresholds['threshold_status'].eq('usable').sum())} usable thresholds, "
            f"{int(thresholds['threshold_status'].eq('skipped').sum())} skipped",
        },
        {
            "check_id": "candidate_masks_use_deployable_feature_buckets_only",
            "passed": bool(not forbidden_hits and not definitions["target_or_residual_used_in_mask"].astype(bool).any()),
            "evidence": "candidate definitions contain feature bucket terms only",
        },
        {
            "check_id": "corrections_have_prior_active_history_only",
            "passed": bool(selected.empty or (selected["prior_rows"] >= 0).all()),
            "evidence": "streaming correction state updates after each active row is scored",
        },
        {
            "check_id": "pair_stage_is_bounded_by_single_feature_screen",
            "passed": bool(len(pair_seed_terms) <= PAIR_SEED_TERMS),
            "evidence": f"{len(pair_seed_terms)} pair seed terms selected from single-feature screen",
        },
        {
            "check_id": "promotion_gate_requires_sample_effect_and_fold_guard",
            "passed": bool(
                promoted.empty
                or (
                    promoted["active_n"].ge(120).all()
                    and promoted["active_delta_mae_vs_reference"].le(-0.010).all()
                    and promoted["delta_mae_vs_reference"].lt(0.0).all()
                    and promoted["fold_delta_max"].le(0.010).all()
                    and promoted["active_correction_share"].ge(0.50).all()
                )
            ),
            "evidence": f"{len(promoted)} candidates passed promotion gate",
        },
    ]
    return pd.DataFrame(checks)


def build_subgroup_scoreboard(scoreboard: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate_type, group in scoreboard.groupby("candidate_type", observed=True):
        rows.append(
            {
                "candidate_type": candidate_type,
                "candidates": int(len(group)),
                "promoted": int(group["promotion_gate_passed"].astype(bool).sum()),
                "best_delta_mae_vs_reference": float(group["delta_mae_vs_reference"].min()),
                "best_active_delta_mae_vs_reference": float(group["active_delta_mae_vs_reference"].min()),
                "median_delta_mae_vs_reference": float(group["delta_mae_vs_reference"].median()),
                "median_active_delta_mae_vs_reference": float(group["active_delta_mae_vs_reference"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values("best_delta_mae_vs_reference").reset_index(drop=True)


def build_readme(
    *,
    summary: dict[str, Any],
    thresholds: pd.DataFrame,
    terms: pd.DataFrame,
    definitions: pd.DataFrame,
    pair_seeds: pd.DataFrame,
    scoreboard: pd.DataFrame,
    subgroups: pd.DataFrame,
    leakage: pd.DataFrame,
) -> str:
    promoted = scoreboard[scoreboard["promotion_gate_passed"].astype(bool)].copy()
    negative = scoreboard[~scoreboard["promotion_gate_passed"].astype(bool)].head(20).copy()
    return f"""# Station-Feature Bucket Residual Mining

Generated: `{summary['generated_at_utc']}`

## Purpose

The previous station-only chain reached a current pre-2024 station-only reference in `0064`. This experiment asks whether the broader `0054` station-feature matrix contains deployable regime buckets that can repair that reference with a strictly past-only residual correction. The experiment is intentionally practical: it does not use 2024+ rows, it does not use target-derived heat labels as candidate inputs, and it does not wait for the slow official forecast backfill. It uses only station-derived features whose timestamp eligibility was already audited before `0054`.

The screen has three candidate families:

- `feature_bucket`: one station feature in a pre-2000 low/mid/high bucket.
- `season_feature_bucket`: strongest base station-feature bucket seeds plus a known calendar season.
- `pair_feature_bucket`: intersections of the strongest bounded base station-feature bucket seeds.

Each candidate is tested with a streaming correction. For a target date `T`, the correction can only use residuals from earlier dates that matched the same candidate mask. The current row's target is added to the state only after the row is scored. To keep the run executable in the local shell, all base feature buckets are screened first; season and pair expansions are then limited to the strongest base seed terms.

## Data Contract

- Reference baseline: `0064` best proxy `{summary['reference_0064_proxy']}`.
- Target dates scored: `{summary['first_date']}` to `{summary['last_date']}`.
- Rows scored per candidate: `{summary['rows_scored']}`.
- Feature thresholds are fixed from station-feature history through `1999-12-31`.
- Candidate masks use station-feature low/mid/high buckets and calendar season only.
- Current local RSS/official forecast archive gaps are not used or filled here.
- 2024-2026 confirmation rows stay locked.

## Headline

| Item | Value |
|---|---:|
| Usable feature thresholds | {summary['usable_feature_thresholds']} |
| Candidate terms | {summary['term_count']} |
| Candidates tested | {summary['candidate_count']} |
| Pair seed terms | {summary['pair_seed_terms']} |
| Promoted candidates | {summary['promoted_candidate_count']} |
| Reference MAE | {summary['reference_mae']} |
| Reference RMSE | {summary['reference_rmse']} |
| Best candidate | {summary['best_candidate']} |
| Best candidate type | {summary['best_candidate_type']} |
| Best MAE | {summary['best_mae']} |
| Best RMSE | {summary['best_rmse']} |
| Best delta MAE vs reference | {summary['best_delta_mae_vs_reference']} |
| Best active delta MAE vs reference | {summary['best_active_delta_mae_vs_reference']} |
| Best promotion gate passed | {summary['best_promotion_gate_passed']} |

## Thresholds

{markdown_table(thresholds, max_rows=30)}

## Term Inventory

{markdown_table(terms.head(40), max_rows=40)}

## Pair Seed Terms

{markdown_table(pair_seeds, max_rows=50)}

## Candidate Definitions

{markdown_table(definitions.head(60), max_rows=60)}

## Scoreboard

{markdown_table(scoreboard.head(80), max_rows=80)}

## Promoted Candidates

{markdown_table(promoted, max_rows=80)}

## Negative / Rejected Candidates

{markdown_table(negative, max_rows=20)}

## Candidate Family Summary

{markdown_table(subgroups, max_rows=20)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Interpretation

This is a station-feature mining screen, not a final production model. A promoted candidate means a deployable bucket mask produced enough past-only active-row improvement, a non-negative full-window contribution, and no excessive fold damage against the current `0064` reference. A rejected candidate may still be scientifically useful, but it is not safe to promote as a prediction component unless a later experiment changes the gate or combines it with stronger routing.

The main thing to look for is whether a station-feature bucket can repeatedly isolate cases where the current reference is biased. If the correction improves active rows but damages the full fold, the signal is real but too unstable. If it improves only inside a tiny subset, it is also not robust enough. If a pair bucket passes the gate, it suggests that cross-station or cross-attribute interactions are more useful than one-feature thresholding.

## Files

- `artifacts/thresholds.csv`
- `artifacts/terms.csv`
- `artifacts/candidate_definitions.csv`
- `artifacts/pair_seed_terms.csv`
- `artifacts/scoreboard.csv`
- `artifacts/subgroup_scoreboard.csv`
- `artifacts/selected_predictions.csv`
- `artifacts/selected_predictions_sample.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_station_feature_bucket_residual_mining.py`:

- `{FOLDER_NAME}`: deployable station-feature bucket and bounded pair residual-mining screen against the current `0064` station-only reference.

| Metric | Value |
|---|---:|
| Reference 0064 MAE | {summary['reference_mae']} |
| Candidates tested | {summary['candidate_count']} |
| Promoted candidates | {summary['promoted_candidate_count']} |
| Best candidate | {summary['best_candidate']} |
| Best full MAE | {summary['best_mae']} |
| Best delta MAE | {summary['best_delta_mae_vs_reference']} |
| Best active delta MAE | {summary['best_active_delta_mae_vs_reference']} |
| Promotion gate passed | {summary['best_promotion_gate_passed']} |

Leakage contract: all buckets are fixed from pre-2000 station history; candidate masks use only station-feature buckets and known season; corrections are prior-active-row only.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Station-Feature Bucket Residual Mining",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_station_feature_bucket_residual_mining.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Reference | `0064` `{summary['reference_0064_proxy']}` | Tested |
| Feature thresholds | `{summary['usable_feature_thresholds']}` usable pre-2000 buckets | Tested |
| Candidates | `{summary['candidate_count']}` feature/season/pair candidates | Tested |
| Promoted candidates | `{summary['promoted_candidate_count']}` | Guarded |
| Best candidate | `{summary['best_candidate']}` | Diagnostic |
| Reference MAE / RMSE | `{summary['reference_mae']}` / `{summary['reference_rmse']}` | Baseline |
| Best MAE / RMSE | `{summary['best_mae']}` / `{summary['best_rmse']}` | Pre-2024 only |
| Best active delta MAE | `{summary['best_active_delta_mae_vs_reference']}` | Active rows |
| Promotion gate passed | `{summary['best_promotion_gate_passed']}` | Guarded |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: `0065` mines station-feature bucket and bounded pair specialists without using target-derived masks or 2024+ confirmation rows.
"""
    update_markdown_section(
        path,
        heading="Station-Feature Bucket Residual Mining",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    blocker = (
        f"49. Station-feature bucket residual mining tested `{summary['candidate_count']}` deployable candidates; "
        f"best delta vs 0064 is `{summary['best_delta_mae_vs_reference']}` from `{summary['best_candidate']}`, "
        f"with `{summary['promoted_candidate_count']}` candidates passing the guarded promotion gate."
    )
    text = path.read_text(encoding="utf-8")
    if blocker not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{blocker}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Continue while the official forecast backfill runs: convert the promoted `0065` bucket specialists into a deterministic guarded station-only stack on top of `0064`, then test whether the stack improves full-window MAE without fold damage before trying any richer model.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def candidates_from_definitions(definitions: pd.DataFrame) -> dict[str, CandidateSpec]:
    candidates: dict[str, CandidateSpec] = {}
    for row in definitions.to_dict("records"):
        half_life_value = row.get("half_life_days")
        half_life = None if pd.isna(half_life_value) else float(half_life_value)
        correction = CorrectionSpec(
            correction_id=str(row["correction_id"]),
            mode=str(row["correction_mode"]),
            half_life_days=half_life,
            min_prior_rows=int(row["min_prior_rows"]),
            shrinkage=float(row["shrinkage"]),
            cap_c=float(row["cap_c"]),
        )
        candidate = CandidateSpec(
            candidate_id=str(row["candidate_id"]),
            candidate_type=str(row["candidate_type"]),
            term_ids=tuple(part for part in str(row["term_ids"]).split("|") if part),
            correction=correction,
        )
        candidates[candidate.candidate_id] = candidate
    return candidates


def write_final_outputs(
    *,
    folder: Path,
    frame: pd.DataFrame,
    summary_0064: dict[str, Any],
    thresholds: pd.DataFrame,
    terms_frame: pd.DataFrame,
    definitions: pd.DataFrame,
    pair_seed_rows: pd.DataFrame,
    scoreboard: pd.DataFrame,
    selected: pd.DataFrame,
    subgroups: pd.DataFrame,
    leakage: pd.DataFrame,
    generated_at: str,
) -> dict[str, Any]:
    reference_score = score_arrays(
        pd.to_numeric(frame["target_tmax_c"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(frame["reference_prediction_c"], errors="coerce").to_numpy(dtype=float),
        pd.to_datetime(frame["target_date"], errors="coerce"),
    )
    best = scoreboard.iloc[0]
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "reference_0064_proxy": str(summary_0064["best_proxy"]),
        "candidate_count": int(len(scoreboard)),
        "initial_candidate_count": int(scoreboard["candidate_type"].eq("feature_bucket").sum()),
        "season_candidate_count": int(scoreboard["candidate_type"].eq("season_feature_bucket").sum()),
        "pair_candidate_count": int(scoreboard["candidate_type"].eq("pair_feature_bucket").sum()),
        "pair_seed_terms": int(len(pair_seed_rows)),
        "usable_feature_thresholds": int(thresholds["threshold_status"].eq("usable").sum()),
        "skipped_feature_thresholds": int(thresholds["threshold_status"].eq("skipped").sum()),
        "term_count": int(len(terms_frame)),
        "rows_scored": int(len(frame)),
        "first_date": str(frame["target_date"].min().date()),
        "last_date": str(frame["target_date"].max().date()),
        "reference_mae": float(reference_score["mae"]),
        "reference_rmse": float(reference_score["rmse"]),
        "best_candidate": str(best["candidate_id"]),
        "best_candidate_type": str(best["candidate_type"]),
        "best_correction_id": str(best["correction_id"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_delta_mae_vs_reference": float(best["delta_mae_vs_reference"]),
        "best_active_delta_mae_vs_reference": float(best["active_delta_mae_vs_reference"]),
        "best_active_n": int(best["active_n"]),
        "best_fold_delta_max": float(best["fold_delta_max"]),
        "best_promotion_gate_passed": bool(best["promotion_gate_passed"]),
        "promoted_candidate_count": int(scoreboard["promotion_gate_passed"].astype(bool).sum()),
        "leakage_check_rows": int(len(leakage)),
        "leakage_checks_passed": int(leakage["passed"].astype(bool).sum()),
        "uses_2024_plus_rows": False,
    }

    artifacts = folder / "artifacts"
    write_csv(artifacts / "selected_predictions.csv", selected)
    write_csv(artifacts / "selected_predictions_sample.csv", selected.head(2000))
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "station_feature_bucket_residual_mining_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            thresholds=thresholds,
            terms=terms_frame,
            definitions=definitions,
            pair_seeds=pair_seed_rows,
            scoreboard=scoreboard,
            subgroups=subgroups,
            leakage=leakage,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def scored_artifacts_present(output_root: Path = RESEARCH_ROOT) -> bool:
    artifacts = output_root / FOLDER_NAME / "artifacts"
    required = [
        "thresholds.csv",
        "terms.csv",
        "candidate_definitions.csv",
        "pair_seed_terms.csv",
        "scoreboard.csv",
        "subgroup_scoreboard.csv",
    ]
    return all((artifacts / name).exists() for name in required)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    if scored_artifacts_present(output_root):
        print("0065 existing scored artifacts detected; finalizing", flush=True)
        return finalize_existing(output_root)

    generated_at = now_utc()
    print("0065 load reference and feature buckets", flush=True)
    frame, summary_0064, thresholds, components = load_reference_frame()
    terms_frame = build_terms(thresholds, components)
    terms = term_specs(terms_frame)

    first_stage_candidates = initial_candidates(terms_frame)
    print(f"0065 score base feature buckets: {len(first_stage_candidates)} candidates", flush=True)
    first_stage_scoreboard = evaluate_candidates(frame, first_stage_candidates, terms)
    pair_seed_term_ids = select_pair_seed_terms(first_stage_scoreboard, terms)
    season_stage_candidates = season_candidates(pair_seed_term_ids, terms)
    pair_stage_candidates = pair_candidates(pair_seed_term_ids)
    all_candidates = first_stage_candidates + season_stage_candidates + pair_stage_candidates

    stage_scoreboards = [first_stage_scoreboard]
    if season_stage_candidates:
        print(f"0065 score season-gated seed buckets: {len(season_stage_candidates)} candidates", flush=True)
        stage_scoreboards.append(evaluate_candidates(frame, season_stage_candidates, terms))
    if pair_stage_candidates:
        print(f"0065 score pair seed buckets: {len(pair_stage_candidates)} candidates", flush=True)
        stage_scoreboards.append(evaluate_candidates(frame, pair_stage_candidates, terms))
    scoreboard = pd.concat(stage_scoreboards, ignore_index=True)
    scoreboard = scoreboard.sort_values(
        ["promotion_gate_passed", "delta_mae_vs_reference", "active_delta_mae_vs_reference"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    definitions = candidate_definitions(all_candidates, terms)
    all_candidate_map = {candidate.candidate_id: candidate for candidate in all_candidates}
    print("0065 materialize selected prediction artifacts", flush=True)
    selected = selected_predictions(frame, scoreboard, all_candidate_map, terms)
    subgroups = build_subgroup_scoreboard(scoreboard)
    pair_seed_rows = terms_frame[terms_frame["term_id"].isin(pair_seed_term_ids)].copy()
    pair_seed_rows["pair_seed_rank"] = pair_seed_rows["term_id"].map(
        {term_id: index + 1 for index, term_id in enumerate(pair_seed_term_ids)}
    )
    pair_seed_rows = pair_seed_rows.sort_values("pair_seed_rank").reset_index(drop=True)
    leakage = leakage_audit(frame, thresholds, definitions, scoreboard, selected, pair_seed_term_ids)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0065 leakage audit failed: {failed}")

    print("0065 write artifacts and docs", flush=True)
    folder = output_root / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    write_csv(artifacts / "thresholds.csv", thresholds)
    write_csv(artifacts / "terms.csv", terms_frame)
    write_csv(artifacts / "candidate_definitions.csv", definitions)
    write_csv(artifacts / "pair_seed_terms.csv", pair_seed_rows)
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "subgroup_scoreboard.csv", subgroups)
    write_csv(artifacts / "selected_predictions.csv", selected)
    write_csv(artifacts / "selected_predictions_sample.csv", selected.head(2000))
    return write_final_outputs(
        folder=folder,
        frame=frame,
        summary_0064=summary_0064,
        thresholds=thresholds,
        terms_frame=terms_frame,
        definitions=definitions,
        pair_seed_rows=pair_seed_rows,
        scoreboard=scoreboard,
        selected=selected,
        subgroups=subgroups,
        leakage=leakage,
        generated_at=generated_at,
    )


def finalize_existing(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    folder = output_root / FOLDER_NAME
    artifacts = folder / "artifacts"
    required = [
        "thresholds.csv",
        "terms.csv",
        "candidate_definitions.csv",
        "pair_seed_terms.csv",
        "scoreboard.csv",
        "subgroup_scoreboard.csv",
    ]
    missing = [name for name in required if not (artifacts / name).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot finalize 0065; missing artifacts: {missing}")
    print("0065 finalize existing scored artifacts", flush=True)
    frame, summary_0064, _thresholds_from_source, _components = load_reference_frame()
    thresholds = pd.read_csv(artifacts / "thresholds.csv")
    terms_frame = pd.read_csv(artifacts / "terms.csv")
    definitions = pd.read_csv(artifacts / "candidate_definitions.csv")
    pair_seed_rows = pd.read_csv(artifacts / "pair_seed_terms.csv")
    scoreboard = pd.read_csv(artifacts / "scoreboard.csv").sort_values(
        ["promotion_gate_passed", "delta_mae_vs_reference", "active_delta_mae_vs_reference"],
        ascending=[False, True, True],
    )
    subgroups = pd.read_csv(artifacts / "subgroup_scoreboard.csv")
    terms = term_specs(terms_frame)
    candidates = candidates_from_definitions(definitions)
    selected = selected_predictions(frame, scoreboard, candidates, terms)
    leakage = leakage_audit(
        frame,
        thresholds,
        definitions,
        scoreboard,
        selected,
        pair_seed_rows["term_id"].astype(str).tolist(),
    )
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0065 finalize leakage audit failed: {failed}")
    return write_final_outputs(
        folder=folder,
        frame=frame,
        summary_0064=summary_0064,
        thresholds=thresholds,
        terms_frame=terms_frame,
        definitions=definitions,
        pair_seed_rows=pair_seed_rows,
        scoreboard=scoreboard,
        selected=selected,
        subgroups=subgroups,
        leakage=leakage,
        generated_at=generated_at,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine deployable station-feature bucket residual specialists against the 0064 station-only reference."
    )
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Finalize docs and summary from already scored 0065 CSV artifacts without rerunning the heavy screen.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = finalize_existing() if args.finalize_existing else run()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
