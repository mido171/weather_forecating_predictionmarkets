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
from scripts.run_hkg_t24_multi_signal_local_residual_lab import (  # noqa: E402
    LocalResidualSpec,
    build_base_feature_sets,
    build_multisignal_frame,
    past_only_local_predictions,
)

FOLDER_NAME = "0031_regime_gated_specialist_selector"
MIN_HISTORY = 80
TOP_BLEND_EXPERTS = 12


@dataclass(frozen=True)
class SpecialistSpec:
    feature_set: str
    active_regimes: tuple[str, ...]
    same_source: bool
    phase_conditioned: bool
    k_neighbors: int = 80


@dataclass(frozen=True)
class SelectorSpec:
    mode: str
    same_source: bool
    match_regime: bool
    min_history: int = MIN_HISTORY


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 150) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def finite_value(row: pd.Series, column: str, default: float = math.nan) -> float:
    if column not in row.index:
        return default
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else default


def flag_value(row: pd.Series, column: str) -> bool:
    return finite_value(row, column, 0.0) >= 0.5


def classify_primary_regime(row: pd.Series) -> str:
    forecast_max = finite_value(row, "forecast_max_c")
    dew_change = finite_value(row, "isd_dew_point_mean_c_change_1d")
    dew_spread = finite_value(row, "isd_temp_dewpoint_spread_mean_c")
    rh_min = finite_value(row, "rh_min_pct")
    pressure_change = finite_value(row, "isd_pressure_mean_hpa_change_1d")
    pressure_slope = finite_value(row, "pressure_plane_slope_magnitude_hpa_per_deg")
    wind_speed = finite_value(row, "isd_wind_speed_mean_mps")
    onshore = finite_value(row, "isd_onshore_easterly_proxy_mps")
    north_south = finite_value(row, "isd_north_south_temp_gradient_c")
    graph_variation = finite_value(row, "isd_graph_total_variation_c2")

    if flag_value(row, "text_any_rain") or flag_value(row, "text_showers") or flag_value(row, "text_thunder"):
        return "rain_cloud"
    if flag_value(row, "text_hot") or flag_value(row, "text_very_hot") or forecast_max >= 31.0:
        return "hot_sunny"
    if dew_change >= 1.5 or rh_min >= 78.0:
        return "moisture_surge"
    if dew_spread >= 8.0 or rh_min <= 58.0:
        return "dry_mixing"
    if wind_speed >= 4.5 or onshore >= 1.7:
        return "wind_marine"
    if abs(pressure_change) >= 2.5 or pressure_slope >= 1.7:
        return "pressure_advection"
    if abs(north_south) >= 2.5 or graph_variation >= 8.0:
        return "station_gradient"
    if flag_value(row, "text_cloud") or flag_value(row, "text_sunny_or_fine"):
        return "cloud_suppressed"
    return "default"


def add_regimes(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["primary_regime"] = out.apply(classify_primary_regime, axis=1)
    return out


def specialist_id(spec: SpecialistSpec) -> str:
    source = "same_source" if spec.same_source else "all_prior"
    phase = "phase" if spec.phase_conditioned else "all_phase"
    regimes = "_".join(spec.active_regimes)
    return slug(f"specialist_{spec.feature_set}_{regimes}_k{spec.k_neighbors}_{source}_{phase}")


def build_specialist_specs(feature_sets: dict[str, object]) -> list[SpecialistSpec]:
    templates = [
        ("official_text_weather", ("rain_cloud", "hot_sunny", "cloud_suppressed", "default")),
        ("official_text_weather", ("all",)),
        ("moisture_upper_air_heat", ("moisture_surge", "dry_mixing")),
        ("pressure_moisture_advection", ("pressure_advection", "moisture_surge")),
        ("station_wind_marine_network", ("wind_marine", "station_gradient")),
    ]
    specs: list[SpecialistSpec] = []
    for feature_set, regimes in templates:
        if feature_set not in feature_sets:
            continue
        for same_source in (False, True):
            specs.append(
                SpecialistSpec(
                    feature_set=feature_set,
                    active_regimes=regimes,
                    same_source=same_source,
                    phase_conditioned=False,
                )
            )
    return specs


def build_specialist_predictions(
    frame: pd.DataFrame,
    feature_sets: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    mapping_rows: list[dict[str, object]] = []
    for spec in build_specialist_specs(feature_sets):
        feature_set = feature_sets[spec.feature_set]
        local_spec = LocalResidualSpec(
            feature_set=spec.feature_set,
            features=feature_set.features,
            k_neighbors=spec.k_neighbors,
            same_source=spec.same_source,
            phase_conditioned=spec.phase_conditioned,
            min_history=160,
        )
        predictions = past_only_local_predictions(frame, local_spec)
        expert_id = specialist_id(spec)
        predictions["expert_id"] = expert_id
        predictions["active_regimes"] = ",".join(spec.active_regimes)
        candidate = score_prediction_frame(predictions.rename(columns={"candidate_prediction_c": "prediction"}), "prediction")
        official = score_prediction_frame(predictions.rename(columns={"official_raw": "official_prediction"}), "official_prediction")
        corrected = predictions["past_rows_used"] > 0
        score_rows.append(
            {
                "expert_id": expert_id,
                "feature_set": spec.feature_set,
                "active_regimes": ",".join(spec.active_regimes),
                "same_source": spec.same_source,
                "phase_conditioned": spec.phase_conditioned,
                "k_neighbors": spec.k_neighbors,
                "feature_count": len(feature_set.features),
                **candidate,
                "official_same_rows_mae": official["mae"],
                "official_same_rows_rmse": official["rmse"],
                "delta_vs_official_same_rows": float(candidate["mae"] - official["mae"]),
                "corrected_rows": int(corrected.sum()),
                "fallback_rows": int((~corrected).sum()),
                "mean_abs_correction_c": float(predictions.loc[corrected, "residual_correction_c"].abs().mean()) if corrected.any() else 0.0,
            }
        )
        mapping_rows.append(
            {
                "expert_id": expert_id,
                "feature_set": spec.feature_set,
                "active_regimes": ",".join(spec.active_regimes),
                "same_source": spec.same_source,
                "phase_conditioned": spec.phase_conditioned,
                "k_neighbors": spec.k_neighbors,
                "features": ",".join(feature_set.features),
            }
        )
        prediction_frames.append(predictions)
    scoreboard = pd.DataFrame(score_rows).sort_values(["delta_vs_official_same_rows", "mae"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    mapping = pd.DataFrame(mapping_rows)
    return scoreboard, predictions, mapping


def expert_active_for_regime(active_regimes: str, regime: str) -> bool:
    values = {value.strip() for value in str(active_regimes).split(",") if value.strip()}
    return "all" in values or regime in values


def selector_id(spec: SelectorSpec) -> str:
    regime = "regime_matched" if spec.match_regime else "all_experts"
    source = "same_source" if spec.same_source else "all_prior"
    return f"regime_selector_{spec.mode}_{regime}_{source}"


def prior_error_stats(
    frame: pd.DataFrame,
    *,
    expert: str,
    target: np.ndarray,
    prior_mask: np.ndarray,
) -> tuple[int, float]:
    values = pd.to_numeric(frame[expert], errors="coerce").to_numpy(dtype=float)
    valid = prior_mask & np.isfinite(values) & np.isfinite(target)
    if not valid.any():
        return 0, math.nan
    error = np.abs(values[valid] - target[valid])
    return int(valid.sum()), float(np.mean(error))


def past_only_regime_selector(
    frame: pd.DataFrame,
    expert_mapping: pd.DataFrame,
    spec: SelectorSpec,
) -> pd.DataFrame:
    ordered = frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True).copy()
    dates = pd.to_datetime(ordered["target_date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
    target = pd.to_numeric(ordered["target_tmax_c"], errors="coerce").to_numpy(dtype=float)
    official = pd.to_numeric(ordered["official_raw"], errors="coerce").to_numpy(dtype=float)
    sources = ordered["forecast_source_family"].astype(str).to_numpy()
    regimes = ordered["primary_regime"].astype(str).to_numpy()
    experts = [str(expert) for expert in expert_mapping["expert_id"].to_list() if expert in ordered.columns]
    active_by_expert = {
        str(row.expert_id): str(row.active_regimes)
        for row in expert_mapping.itertuples(index=False)
    }

    predictions: list[float] = []
    selected: list[str] = []
    eligible_counts: list[int] = []
    prior_rows_used: list[int] = []
    selected_prior_mae: list[float] = []
    for index, target_date in enumerate(dates):
        current_regime = regimes[index]
        candidate_experts = [
            expert
            for expert in experts
            if np.isfinite(pd.to_numeric(pd.Series([ordered.at[index, expert]]), errors="coerce").iloc[0])
            and (not spec.match_regime or expert_active_for_regime(active_by_expert[expert], current_regime))
        ]
        if not candidate_experts:
            predictions.append(float(official[index]) if np.isfinite(official[index]) else math.nan)
            selected.append("official_raw_fallback")
            eligible_counts.append(0)
            prior_rows_used.append(0)
            selected_prior_mae.append(math.nan)
            continue

        prior_mask_base = np.arange(len(ordered)) < int(np.searchsorted(dates, target_date, side="left"))
        if spec.same_source:
            prior_mask_base &= sources == sources[index]
        if spec.match_regime:
            prior_mask_base &= regimes == current_regime

        scored: list[tuple[str, int, float]] = []
        for expert in candidate_experts:
            n, mae = prior_error_stats(ordered, expert=expert, target=target, prior_mask=prior_mask_base)
            if n >= spec.min_history and np.isfinite(mae):
                scored.append((expert, n, mae))

        if not scored:
            predictions.append(float(official[index]) if np.isfinite(official[index]) else math.nan)
            selected.append("official_raw_fallback")
            eligible_counts.append(0)
            prior_rows_used.append(0)
            selected_prior_mae.append(math.nan)
            continue

        eligible_counts.append(len(scored))
        if spec.mode == "best":
            best = min(scored, key=lambda item: (item[2], item[0]))
            prediction = float(ordered.at[index, best[0]])
            predictions.append(prediction)
            selected.append(best[0])
            prior_rows_used.append(best[1])
            selected_prior_mae.append(best[2])
        elif spec.mode == "inverse_mae":
            weights = np.array([1.0 / max(item[2], 1e-6) for item in scored], dtype=float)
            values = np.array([float(ordered.at[index, item[0]]) for item in scored], dtype=float)
            weights = weights / weights.sum()
            predictions.append(float(np.dot(weights, values)))
            selected.append("inverse_mae_blend")
            prior_rows_used.append(int(sum(item[1] for item in scored)))
            selected_prior_mae.append(float(np.average([item[2] for item in scored], weights=weights)))
        else:
            raise ValueError(f"Unknown selector mode: {spec.mode}")

    out = ordered[["target_date", "forecast_source_family", "primary_regime", "target_tmax_c", "official_raw"]].copy()
    out["selector_prediction_c"] = predictions
    out["selected_expert"] = selected
    out["eligible_expert_count"] = eligible_counts
    out["prior_rows_used"] = prior_rows_used
    out["selected_prior_mae"] = selected_prior_mae
    out["mode"] = spec.mode
    out["same_source"] = spec.same_source
    out["match_regime"] = spec.match_regime
    return out


def build_selector_frame(
    frame: pd.DataFrame,
    specialist_predictions: pd.DataFrame,
    expert_mapping: pd.DataFrame,
) -> pd.DataFrame:
    base = frame[["target_date", "forecast_source_family", "primary_regime", "target_tmax_c", "forecast_max_c"]].copy()
    base["official_raw"] = pd.to_numeric(base["forecast_max_c"], errors="coerce")
    if specialist_predictions.empty:
        return base
    long = specialist_predictions[["target_date", "expert_id", "candidate_prediction_c"]].copy()
    allowed = set(expert_mapping["expert_id"].astype(str).to_list())
    long = long[long["expert_id"].isin(allowed)]
    wide = (
        long.pivot_table(index="target_date", columns="expert_id", values="candidate_prediction_c", aggfunc="last")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return base.merge(wide, on="target_date", how="left").sort_values("target_date").reset_index(drop=True)


def run_selectors(
    frame: pd.DataFrame,
    specialist_predictions: pd.DataFrame,
    expert_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selector_frame = build_selector_frame(frame, specialist_predictions, expert_mapping)
    specs = [
        SelectorSpec(mode=mode, same_source=same_source, match_regime=match_regime)
        for mode in ("best", "inverse_mae")
        for same_source in (False, True)
        for match_regime in (False, True)
    ]
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for spec in specs:
        predictions = past_only_regime_selector(selector_frame, expert_mapping, spec)
        candidate_id = selector_id(spec)
        predictions["candidate_id"] = candidate_id
        candidate = score_prediction_frame(predictions.rename(columns={"selector_prediction_c": "prediction"}), "prediction")
        official = score_prediction_frame(predictions.rename(columns={"official_raw": "official_prediction"}), "official_prediction")
        score_rows.append(
            {
                "candidate_id": candidate_id,
                "mode": spec.mode,
                "same_source": spec.same_source,
                "match_regime": spec.match_regime,
                **candidate,
                "official_same_rows_mae": official["mae"],
                "official_same_rows_rmse": official["rmse"],
                "delta_vs_official_same_rows": float(candidate["mae"] - official["mae"]),
                "fallback_rows": int(predictions["selected_expert"].eq("official_raw_fallback").sum()),
                "mean_eligible_expert_count": float(predictions["eligible_expert_count"].mean()),
            }
        )
        prediction_frames.append(predictions)
    scoreboard = pd.DataFrame(score_rows).sort_values(["delta_vs_official_same_rows", "mae"]).reset_index(drop=True)
    return scoreboard, pd.concat(prediction_frames, ignore_index=True)


def regime_counts(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["forecast_source_family", "primary_regime"], observed=True)
        .agg(rows=("target_date", "count"), first_date=("target_date", "min"), last_date=("target_date", "max"))
        .reset_index()
        .assign(
            first_date=lambda table: pd.to_datetime(table["first_date"]).dt.date.astype(str),
            last_date=lambda table: pd.to_datetime(table["last_date"]).dt.date.astype(str),
        )
        .sort_values(["forecast_source_family", "rows"], ascending=[True, False])
    )


def selection_counts(selector_predictions: pd.DataFrame) -> pd.DataFrame:
    if selector_predictions.empty:
        return pd.DataFrame()
    return (
        selector_predictions.groupby(["candidate_id", "primary_regime", "selected_expert"], observed=True, dropna=False)
        .agg(rows=("target_date", "count"))
        .reset_index()
        .sort_values(["candidate_id", "primary_regime", "rows"], ascending=[True, True, False])
    )


def best_prior_screen_rows() -> pd.DataFrame:
    paths = [
        (
            "0018_official_expert_blend",
            RESEARCH_ROOT / "0018_past_only_official_expert_blend_screen" / "artifacts" / "scoreboard.csv",
        ),
        (
            "0026_pressure_gradient_blend",
            RESEARCH_ROOT / "0026_pressure_gradient_experts" / "artifacts" / "blend_scoreboard.csv",
        ),
        (
            "0030_best_local",
            RESEARCH_ROOT / "0030_multi_signal_local_residual_lab" / "artifacts" / "local_scoreboard.csv",
        ),
        (
            "0030_best_blend",
            RESEARCH_ROOT / "0030_multi_signal_local_residual_lab" / "artifacts" / "blend_scoreboard.csv",
        ),
    ]
    rows: list[dict[str, object]] = []
    for family, path in paths:
        if not path.exists():
            continue
        table = pd.read_csv(path)
        if table.empty:
            continue
        best = table.sort_values(["mae", "rmse"]).iloc[0]
        rows.append(
            {
                "system": family,
                "candidate_id": str(best["candidate_id"] if "candidate_id" in best.index else best.get("expert_id", "")),
                "mae": float(best["mae"]),
                "rmse": float(best["rmse"]),
                "delta_vs_official": float(best.get("delta_vs_official_same_rows", math.nan)),
                "n": int(best.get("n", 0)),
                "first_date": str(best.get("first_date", "")),
                "last_date": str(best.get("last_date", "")),
            }
        )
    return pd.DataFrame(rows)


def baseline_comparison(frame: pd.DataFrame, selector_scoreboard: pd.DataFrame) -> pd.DataFrame:
    official = score_prediction_frame(frame.rename(columns={"official_raw": "prediction"}), "prediction")
    rows: list[dict[str, object]] = [
        {
            "system": "official_raw",
            "candidate_id": "official_raw",
            "mae": official["mae"],
            "rmse": official["rmse"],
            "delta_vs_official": 0.0,
            "n": official["n"],
            "first_date": official["first_date"],
            "last_date": official["last_date"],
        }
    ]
    prior = best_prior_screen_rows()
    if not prior.empty:
        rows.extend(prior.to_dict("records"))
    if not selector_scoreboard.empty:
        best = selector_scoreboard.iloc[0]
        rows.append(
            {
                "system": "0031_best_selector",
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


def write_outputs(
    *,
    frame: pd.DataFrame,
    specialist_scoreboard: pd.DataFrame,
    specialist_predictions: pd.DataFrame,
    expert_mapping: pd.DataFrame,
    selector_scoreboard: pd.DataFrame,
    selector_predictions: pd.DataFrame,
) -> dict[str, object]:
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    comparison = baseline_comparison(frame, selector_scoreboard)
    write_csv(artifacts / "regime_counts.csv", regime_counts(frame))
    write_csv(artifacts / "specialist_mapping.csv", expert_mapping)
    write_csv(artifacts / "specialist_scoreboard.csv", specialist_scoreboard)
    write_csv(artifacts / "specialist_predictions.csv", specialist_predictions)
    write_csv(artifacts / "selector_scoreboard.csv", selector_scoreboard)
    write_csv(artifacts / "selector_predictions.csv", selector_predictions)
    write_csv(artifacts / "selection_counts.csv", selection_counts(selector_predictions))
    write_csv(artifacts / "baseline_comparison.csv", comparison)

    best_selector = selector_scoreboard.iloc[0] if not selector_scoreboard.empty else None
    best_specialist = specialist_scoreboard.iloc[0] if not specialist_scoreboard.empty else None
    manifest = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "official_rows": int(len(frame)),
        "first_target_date": str(frame["target_date"].min().date()),
        "last_target_date": str(frame["target_date"].max().date()),
        "source_counts": {str(k): int(v) for k, v in frame["forecast_source_family"].value_counts().to_dict().items()},
        "regime_counts": {str(k): int(v) for k, v in frame["primary_regime"].value_counts().to_dict().items()},
        "specialists": int(len(expert_mapping)),
        "selector_candidates": int(len(selector_scoreboard)),
        "best_specialist": "" if best_specialist is None else str(best_specialist["expert_id"]),
        "best_specialist_mae": None if best_specialist is None else float(best_specialist["mae"]),
        "best_specialist_rmse": None if best_specialist is None else float(best_specialist["rmse"]),
        "best_specialist_delta_vs_official": None if best_specialist is None else float(best_specialist["delta_vs_official_same_rows"]),
        "best_selector": "" if best_selector is None else str(best_selector["candidate_id"]),
        "best_selector_mae": None if best_selector is None else float(best_selector["mae"]),
        "best_selector_rmse": None if best_selector is None else float(best_selector["rmse"]),
        "best_selector_delta_vs_official": None if best_selector is None else float(best_selector["delta_vs_official_same_rows"]),
        "confirmation_start": str(CONFIRMATION_START.date()),
    }
    write_json(RESEARCH_ROOT / "regime_gated_specialist_selector_manifest.json", manifest)
    write_readme(
        folder=folder,
        manifest=manifest,
        specialist_scoreboard=specialist_scoreboard,
        selector_scoreboard=selector_scoreboard,
        comparison=comparison,
        counts=regime_counts(frame),
        selections=selection_counts(selector_predictions),
    )
    update_master_index(manifest)
    return manifest


def write_readme(
    *,
    folder: Path,
    manifest: dict[str, object],
    specialist_scoreboard: pd.DataFrame,
    selector_scoreboard: pd.DataFrame,
    comparison: pd.DataFrame,
    counts: pd.DataFrame,
    selections: pd.DataFrame,
) -> None:
    best_specialist = specialist_scoreboard.iloc[0] if not specialist_scoreboard.empty else None
    best_selector = selector_scoreboard.iloc[0] if not selector_scoreboard.empty else None
    best_specialist_text = "No specialist was scoreable."
    if best_specialist is not None:
        best_specialist_text = (
            f"Best standalone specialist: `{best_specialist['expert_id']}` with MAE `{best_specialist['mae']:.4f}`, "
            f"RMSE `{best_specialist['rmse']:.4f}`, and official delta "
            f"`{best_specialist['delta_vs_official_same_rows']:.4f}`."
        )
    best_selector_text = "No selector was scoreable."
    if best_selector is not None:
        best_selector_text = (
            f"Best regime selector: `{best_selector['candidate_id']}` with MAE `{best_selector['mae']:.4f}`, "
            f"RMSE `{best_selector['rmse']:.4f}`, and official delta "
            f"`{best_selector['delta_vs_official_same_rows']:.4f}`."
        )

    readme = f"""# Regime-Gated Specialist Selector

Generated: `{manifest['generated_at_utc']}`

## Purpose

`0030` showed that putting every physical channel into one nearest-neighbour distance diluted the signal. This insight tests the next architecture: small specialist residual experts, then a leakage-safe selector that chooses among them by current regime and prior specialist performance.

The regimes are deterministic and use only pre-cutoff information:

- rain/cloud official wording;
- hot/sunny official wording or high official Tmax forecast;
- moisture surge;
- dry mixing;
- wind/marine influence;
- pressure/advection;
- station-gradient structure;
- default.

## Data Window

Rows used: `{manifest['official_rows']}` official forecast/target rows.

Date range: `{manifest['first_target_date']}` to `{manifest['last_target_date']}`.

Source counts: `{manifest['source_counts']}`.

Regime counts: `{manifest['regime_counts']}`.

The scored official archive remains non-contiguous and still excludes moving partial 2005+ detail pages.

## Leakage Contract

- All scored rows are earlier than `{CONFIRMATION_START.date()}`.
- Current regime classification uses only current pre-cutoff features, not the target label.
- Each specialist forecast is a prior-only local residual correction.
- Selector performance estimates use only strictly earlier target dates.
- Same-source selector candidates restrict performance history to the same forecast source family.
- Regime-matched selector candidates estimate performance only inside the same prior regime.
- No 2024+ target labels are loaded or scored.

## Main Results

{best_specialist_text}

{best_selector_text}

## Baseline Comparison

{markdown_table(comparison, max_rows=20)}

## Regime Counts

{markdown_table(counts, max_rows=20)}

## Specialist Scoreboard

{markdown_table(specialist_scoreboard.head(20), max_rows=20)}

## Selector Scoreboard

{markdown_table(selector_scoreboard, max_rows=20)}

## Selection Counts

{markdown_table(selections.head(40), max_rows=40)}

## Interpretation

This screen tests whether staged selection is better than a single broad similarity metric. If the regime-matched selector beats the best standalone specialist and prior champions, that suggests the architecture should be promoted. If the standalone official text/weather specialist still wins, then current deterministic regime gating is not yet granular enough and the next work should focus on better regime discovery or more continuous forecast history.
"""
    write_text(folder / "README.md", readme)


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Regime-Gated Specialist Selector\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_regime_gated_specialist_selector.py`:

- `{FOLDER_NAME}`: deterministic regime labels, small residual specialists, and fold-local specialist selection by prior regime/source performance.

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Specialists | {manifest['specialists']} |
| Selector candidates | {manifest['selector_candidates']} |
| Best specialist MAE | {manifest['best_specialist_mae']} |
| Best specialist RMSE | {manifest['best_specialist_rmse']} |
| Best specialist delta vs official | {manifest['best_specialist_delta_vs_official']} |
| Best selector MAE | {manifest['best_selector_mae']} |
| Best selector RMSE | {manifest['best_selector_rmse']} |
| Best selector delta vs official | {manifest['best_selector_delta_vs_official']} |

Leakage contract: all scored rows are before `{CONFIRMATION_START.date()}`; specialist predictions and selector weights/choices use strictly prior target dates.
"""
    write_text(index_path, text)


def run() -> dict[str, object]:
    frame = add_regimes(build_multisignal_frame())
    require_no_confirmation_dates(frame["target_date"], context="regime-gated specialist selector")
    feature_sets = build_base_feature_sets(frame)
    specialist_scoreboard, specialist_predictions, expert_mapping = build_specialist_predictions(frame, feature_sets)
    selector_scoreboard, selector_predictions = run_selectors(frame, specialist_predictions, expert_mapping)
    return write_outputs(
        frame=frame,
        specialist_scoreboard=specialist_scoreboard,
        specialist_predictions=specialist_predictions,
        expert_mapping=expert_mapping,
        selector_scoreboard=selector_scoreboard,
        selector_predictions=selector_predictions,
    )


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 regime-gated specialist selector.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
