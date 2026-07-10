from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

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
from scripts.run_hkg_t24_station_network_forecast_residual_interaction_mining import (  # noqa: E402
    LATE_EVAL_START,
)
from scripts.run_hkg_t24_station_network_forecast_stack import (  # noqa: E402
    FOLDER_NAME as STACK_0041_FOLDER,
)
from scripts.run_hkg_t24_station_network_forecast_stack import (  # noqa: E402
    MIN_BUCKET_HISTORY,
    MIN_GLOBAL_HISTORY,
    StackSpec,
    build_stack_frame,
    family_scoreboard,
    past_only_stack_predictions,
    score_stack_candidate,
    stack_feature_sets,
)

FOLDER_NAME = "0042_trust_router_sensitivity"
STACK_0041_MANIFEST = RESEARCH_ROOT / "station_network_forecast_stack_manifest.json"
STACK_0041_ARTIFACTS = RESEARCH_ROOT / STACK_0041_FOLDER / "artifacts"
SCREEN_STAGE = "stage1_targeted_sensitivity"
HISTORY_THRESHOLDS = ((80, 30), (120, 45), (160, 45), (240, 60), (360, 90), (500, 120))
PRIMARY_CONTEXTS = ("source_revision_action", "source_forecast_text")


@dataclass(frozen=True)
class SensitivitySpec:
    variant_kind: str
    comparison_group: str
    stack_spec: StackSpec


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 140) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sensitivity_candidate_id(spec: SensitivitySpec) -> str:
    stack = spec.stack_spec
    source = "same_source" if stack.same_source else "all_prior"
    threshold = f"g{stack.min_global_history}_b{stack.min_bucket_history}"
    return slug(
        f"sensitivity_0042_{spec.variant_kind}_{spec.comparison_group}_{stack.family_group}_"
        f"{stack.feature_set}_{stack.mode}_{source}_{threshold}"
    )


def family_variant_map(family_catalog: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    available = set(family_catalog["family_name"].astype(str))

    def present(*names: str) -> tuple[str, ...]:
        return tuple(name for name in names if name in available)

    variants = {
        "core": present("anchor_0038_c", "hard_0039_best_c", "smooth_0040_01", "smooth_0040_02"),
        "anchor_hard": present("anchor_0038_c", "hard_0039_best_c"),
        "anchor_smooth_late": present("anchor_0038_c", "smooth_0040_01"),
        "anchor_smooth_full": present("anchor_0038_c", "smooth_0040_02"),
        "anchor_hard_smooth_late": present("anchor_0038_c", "hard_0039_best_c", "smooth_0040_01"),
        "anchor_hard_smooth_full": present("anchor_0038_c", "hard_0039_best_c", "smooth_0040_02"),
        "anchor_smooth_pair": present("anchor_0038_c", "smooth_0040_01", "smooth_0040_02"),
        "expanded": present(
            "official_raw",
            "anchor_0038_c",
            "hard_0039_best_c",
            "smooth_0040_01",
            "smooth_0040_02",
            "smooth_0040_03",
            "smooth_0040_04",
        ),
    }
    return {name: families for name, families in variants.items() if "anchor_0038_c" in families and len(families) >= 2}


def add_spec(
    specs: list[SensitivitySpec],
    seen: set[tuple[object, ...]],
    *,
    variant_kind: str,
    comparison_group: str,
    feature_set: str,
    feature_names: tuple[str, ...],
    mode: str,
    same_source: bool,
    family_group: str,
    family_names: tuple[str, ...],
    min_global_history: int = MIN_GLOBAL_HISTORY,
    min_bucket_history: int = MIN_BUCKET_HISTORY,
) -> None:
    key = (
        variant_kind,
        comparison_group,
        feature_set,
        mode,
        same_source,
        family_group,
        family_names,
        min_global_history,
        min_bucket_history,
    )
    if key in seen:
        return
    seen.add(key)
    specs.append(
        SensitivitySpec(
            variant_kind=variant_kind,
            comparison_group=comparison_group,
            stack_spec=StackSpec(
                feature_set=feature_set,
                feature_names=feature_names,
                mode=mode,
                same_source=same_source,
                family_group=family_group,
                family_names=family_names,
                min_global_history=min_global_history,
                min_bucket_history=min_bucket_history,
            ),
        )
    )


def build_sensitivity_specs(family_catalog: pd.DataFrame, meta_catalog: pd.DataFrame) -> pd.DataFrame:
    feature_sets = stack_feature_sets(meta_catalog)
    family_variants = family_variant_map(family_catalog)
    specs: list[SensitivitySpec] = []
    seen: set[tuple[object, ...]] = set()
    core = family_variants["core"]

    for context in (
        "global",
        "source",
        "text_signal",
        "source_forecast_text",
        "source_revision_action",
        "correction_activity",
        "prediction_disagreement",
        "compact_all",
    ):
        if context in feature_sets:
            add_spec(
                specs,
                seen,
                variant_kind="context",
                comparison_group=context,
                feature_set=context,
                feature_names=feature_sets[context],
                mode="positive_lift",
                same_source=True,
                family_group="core",
                family_names=core,
            )

    for context in PRIMARY_CONTEXTS:
        if context not in feature_sets:
            continue
        for mode in ("best", "inverse_mae", "positive_lift", "anchor_lift_blend"):
            for same_source in (False, True):
                add_spec(
                    specs,
                    seen,
                    variant_kind="routing_mode",
                    comparison_group=f"{context}_{mode}_{'same_source' if same_source else 'all_prior'}",
                    feature_set=context,
                    feature_names=feature_sets[context],
                    mode=mode,
                    same_source=same_source,
                    family_group="core",
                    family_names=core,
                )

    for context in PRIMARY_CONTEXTS:
        if context not in feature_sets:
            continue
        for min_global, min_bucket in HISTORY_THRESHOLDS:
            add_spec(
                specs,
                seen,
                variant_kind="history_threshold",
                comparison_group=f"{context}_g{min_global}_b{min_bucket}",
                feature_set=context,
                feature_names=feature_sets[context],
                mode="positive_lift",
                same_source=True,
                family_group="core",
                family_names=core,
                min_global_history=min_global,
                min_bucket_history=min_bucket,
            )

    for context in PRIMARY_CONTEXTS:
        if context not in feature_sets:
            continue
        for variant_name, family_names in family_variants.items():
            add_spec(
                specs,
                seen,
                variant_kind="family_inclusion",
                comparison_group=variant_name,
                feature_set=context,
                feature_names=feature_sets[context],
                mode="positive_lift",
                same_source=True,
                family_group=variant_name,
                family_names=family_names,
            )

    rows = []
    for spec in specs:
        stack = spec.stack_spec
        rows.append(
            {
                "candidate_id": sensitivity_candidate_id(spec),
                "variant_kind": spec.variant_kind,
                "comparison_group": spec.comparison_group,
                "feature_set": stack.feature_set,
                "feature_names": ",".join(stack.feature_names),
                "mode": stack.mode,
                "same_source": stack.same_source,
                "family_group": stack.family_group,
                "family_names": ",".join(stack.family_names),
                "family_count": len(stack.family_names),
                "min_global_history": stack.min_global_history,
                "min_bucket_history": stack.min_bucket_history,
            }
        )
    return pd.DataFrame(rows)


def specs_from_catalog(spec_catalog: pd.DataFrame) -> list[SensitivitySpec]:
    specs: list[SensitivitySpec] = []
    for row in spec_catalog.itertuples(index=False):
        feature_names = tuple(str(row.feature_names).split(",")) if isinstance(row.feature_names, str) and row.feature_names else ()
        family_names = tuple(str(row.family_names).split(","))
        specs.append(
            SensitivitySpec(
                variant_kind=str(row.variant_kind),
                comparison_group=str(row.comparison_group),
                stack_spec=StackSpec(
                    feature_set=str(row.feature_set),
                    feature_names=feature_names,
                    mode=str(row.mode),
                    same_source=bool(row.same_source),
                    family_group=str(row.family_group),
                    family_names=family_names,
                    min_global_history=int(row.min_global_history),
                    min_bucket_history=int(row.min_bucket_history),
                ),
            )
        )
    return specs


def score_sensitivity_candidate(predictions: pd.DataFrame, spec: SensitivitySpec) -> dict[str, object]:
    score = score_stack_candidate(predictions, spec.stack_spec)
    score["candidate_id"] = sensitivity_candidate_id(spec)
    score["variant_kind"] = spec.variant_kind
    score["comparison_group"] = spec.comparison_group
    score["min_global_history"] = spec.stack_spec.min_global_history
    score["min_bucket_history"] = spec.stack_spec.min_bucket_history
    score["family_names"] = ",".join(spec.stack_spec.family_names)
    return score


def run_sensitivity_screen(frame: pd.DataFrame, spec_catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for spec in specs_from_catalog(spec_catalog):
        predictions = past_only_stack_predictions(frame, spec.stack_spec)
        candidate_id = sensitivity_candidate_id(spec)
        predictions["candidate_id"] = candidate_id
        predictions["variant_kind"] = spec.variant_kind
        predictions["comparison_group"] = spec.comparison_group
        predictions["min_global_history"] = spec.stack_spec.min_global_history
        predictions["min_bucket_history"] = spec.stack_spec.min_bucket_history
        predictions["family_names"] = ",".join(spec.stack_spec.family_names)
        score_rows.append(score_sensitivity_candidate(predictions, spec))
        prediction_frames.append(predictions)
    scoreboard = pd.DataFrame(score_rows)
    if not scoreboard.empty:
        scoreboard = scoreboard.sort_values(["late_eval_mae", "mae", "rmse"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return scoreboard, predictions


def segment_definitions(predictions: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    dates = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    sources = predictions["forecast_source_family"].astype(str)
    segments: list[tuple[str, pd.Series]] = [
        ("all_rows", pd.Series(True, index=predictions.index)),
        ("late_eval_actual", dates >= LATE_EVAL_START),
        ("press_archive", sources.eq("press_archive")),
        ("rss_archive", sources.eq("rss_archive")),
        ("press_2000_2004", dates < pd.Timestamp("2005-01-01")),
        ("rss_2021_2023", dates >= pd.Timestamp("2021-01-01")),
    ]
    years = sorted(int(year) for year in dates.dt.year.dropna().unique())
    for year in years:
        mask = dates.dt.year.eq(year)
        if int(mask.sum()) >= 120:
            segments.append((f"year_{year}", mask))
    return segments


def segment_scoreboard(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if predictions.empty:
        return pd.DataFrame()
    for candidate_id, candidate_rows in predictions.groupby("candidate_id", observed=True):
        for segment_name, mask in segment_definitions(candidate_rows):
            subset = candidate_rows[mask.to_numpy()].copy()
            if len(subset) < 30:
                continue
            score = score_prediction_frame(subset.rename(columns={"candidate_prediction_c": "prediction"}), "prediction")
            anchor = score_prediction_frame(subset.rename(columns={"anchor_0038_c": "prediction"}), "prediction")
            official = score_prediction_frame(subset.rename(columns={"official_raw": "prediction"}), "prediction")
            first = subset.iloc[0]
            rows.append(
                {
                    "candidate_id": str(candidate_id),
                    "segment": segment_name,
                    "variant_kind": str(first["variant_kind"]),
                    "comparison_group": str(first["comparison_group"]),
                    **score,
                    "anchor_mae": float(anchor["mae"]),
                    "official_mae": float(official["mae"]),
                    "delta_vs_anchor": float(score["mae"] - anchor["mae"]),
                    "delta_vs_official": float(score["mae"] - official["mae"]),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["segment", "mae", "rmse"]).reset_index(drop=True)


def robustness_summary(scoreboard: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    if scoreboard.empty:
        return pd.DataFrame()
    source_segments = segments[segments["segment"].isin(["press_archive", "rss_archive"])].copy()
    worst = (
        segments.groupby("candidate_id", observed=True)
        .agg(
            worst_segment_mae=("mae", "max"),
            worst_delta_vs_anchor=("delta_vs_anchor", "max"),
            segments_scored=("segment", "count"),
            segments_beating_anchor=("delta_vs_anchor", lambda s: int((s < 0).sum())),
        )
        .reset_index()
    )
    if not source_segments.empty:
        source = (
            source_segments.groupby("candidate_id", observed=True)
            .agg(source_mae_spread=("mae", lambda s: float(s.max() - s.min())), worst_source_mae=("mae", "max"))
            .reset_index()
        )
        worst = worst.merge(source, on="candidate_id", how="left")
    cols = [
        "candidate_id",
        "variant_kind",
        "comparison_group",
        "feature_set",
        "mode",
        "same_source",
        "family_group",
        "family_count",
        "min_global_history",
        "min_bucket_history",
        "mae",
        "rmse",
        "delta_vs_anchor",
        "late_eval_mae",
        "late_eval_delta_vs_anchor",
    ]
    return scoreboard[cols].merge(worst, on="candidate_id", how="left").sort_values(
        ["late_eval_mae", "mae", "worst_delta_vs_anchor"]
    )


def variant_leaders(scoreboard: pd.DataFrame) -> pd.DataFrame:
    if scoreboard.empty:
        return pd.DataFrame()
    rows = []
    for variant_kind, group in scoreboard.groupby("variant_kind", observed=True):
        best_late = group.sort_values(["late_eval_mae", "mae", "rmse"]).iloc[0]
        best_full = group.sort_values(["mae", "late_eval_mae", "rmse"]).iloc[0]
        for metric, row in (("best_late", best_late), ("best_full", best_full)):
            rows.append(
                {
                    "variant_kind": str(variant_kind),
                    "leader_type": metric,
                    "candidate_id": str(row["candidate_id"]),
                    "comparison_group": str(row["comparison_group"]),
                    "feature_set": str(row["feature_set"]),
                    "mode": str(row["mode"]),
                    "same_source": bool(row["same_source"]),
                    "family_group": str(row["family_group"]),
                    "mae": float(row["mae"]),
                    "rmse": float(row["rmse"]),
                    "delta_vs_anchor": float(row["delta_vs_anchor"]),
                    "late_eval_mae": float(row["late_eval_mae"]),
                    "late_eval_delta_vs_anchor": float(row["late_eval_delta_vs_anchor"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["leader_type", "late_eval_mae", "mae"]).reset_index(drop=True)


def baseline_comparison(
    *,
    scoreboard: pd.DataFrame,
    family_scores: pd.DataFrame,
    stack_0041_manifest: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in family_scores.itertuples(index=False):
        rows.append(
            {
                "system": f"family_{row.family_name}",
                "candidate_id": str(row.candidate_id),
                "mae": float(row.mae),
                "rmse": float(row.rmse),
                "late_eval_mae": float(row.late_eval_mae),
                "late_eval_rmse": float(row.late_eval_rmse),
            }
        )
    rows.extend(
        [
            {
                "system": "0041_best_late_stack",
                "candidate_id": str(stack_0041_manifest["best_late_candidate"]),
                "mae": float(stack_0041_manifest["best_late_candidate_full_mae"]),
                "rmse": math.nan,
                "late_eval_mae": float(stack_0041_manifest["best_late_eval_mae"]),
                "late_eval_rmse": math.nan,
            },
            {
                "system": "0041_best_full_stack",
                "candidate_id": str(stack_0041_manifest["best_full_candidate"]),
                "mae": float(stack_0041_manifest["best_full_mae"]),
                "rmse": math.nan,
                "late_eval_mae": float(stack_0041_manifest["best_full_candidate_late_eval_mae"]),
                "late_eval_rmse": math.nan,
            },
        ]
    )
    if not scoreboard.empty:
        best_late = scoreboard.iloc[0]
        best_full = scoreboard.sort_values(["mae", "late_eval_mae", "rmse"]).iloc[0]
        rows.extend(
            [
                {
                    "system": "0042_best_late_sensitivity",
                    "candidate_id": str(best_late["candidate_id"]),
                    "mae": float(best_late["mae"]),
                    "rmse": float(best_late["rmse"]),
                    "late_eval_mae": float(best_late["late_eval_mae"]),
                    "late_eval_rmse": float(best_late["late_eval_rmse"]),
                },
                {
                    "system": "0042_best_full_sensitivity",
                    "candidate_id": str(best_full["candidate_id"]),
                    "mae": float(best_full["mae"]),
                    "rmse": float(best_full["rmse"]),
                    "late_eval_mae": float(best_full["late_eval_mae"]),
                    "late_eval_rmse": float(best_full["late_eval_rmse"]),
                },
            ]
        )
    return pd.DataFrame(rows).sort_values(["late_eval_mae", "mae"], na_position="last").reset_index(drop=True)


def write_readme(
    *,
    folder: Path,
    manifest: dict[str, object],
    spec_catalog: pd.DataFrame,
    family_scores: pd.DataFrame,
    scoreboard: pd.DataFrame,
    leaders: pd.DataFrame,
    robustness: pd.DataFrame,
    segments: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    best_late = scoreboard.iloc[0] if not scoreboard.empty else None
    best_full = scoreboard.sort_values(["mae", "late_eval_mae", "rmse"]).iloc[0] if not scoreboard.empty else None
    best_text = "No scoreable sensitivity candidate was produced."
    if best_late is not None and best_full is not None:
        best_text = (
            f"Best actual late-window sensitivity candidate: `{best_late['candidate_id']}` with MAE "
            f"`{best_late['late_eval_mae']:.4f}`, RMSE `{best_late['late_eval_rmse']:.4f}`, "
            f"late delta vs anchor `{best_late['late_eval_delta_vs_anchor']:.4f}`, and full MAE "
            f"`{best_late['mae']:.4f}`.\n\n"
            f"Best full-window sensitivity candidate: `{best_full['candidate_id']}` with full MAE "
            f"`{best_full['mae']:.4f}`, RMSE `{best_full['rmse']:.4f}`, full delta vs anchor "
            f"`{best_full['delta_vs_anchor']:.4f}`, and actual late-window MAE "
            f"`{best_full['late_eval_mae']:.4f}`."
        )
    readme = f"""# Trust Router Sensitivity

Generated: `{manifest['generated_at_utc']}`

## Purpose

`0041` produced a real compound improvement by routing between the forecast-history anchor, the hard station-network residual correction, and smooth station-network residual specialists. The improvement was encouraging but not yet enough to trust blindly. This experiment asks whether that router survives the stress tests that matter before any later sealed confirmation: different family inclusion sets, different minimum history thresholds, different routing contexts, same-source versus all-prior evidence, source-specific behavior, and full-window versus actual late-window stability.

This is not a predictive modelling run and it does not touch 2024+ confirmation rows. The experiment uses already-audited candidate families and reruns only strict prior-only stack routers. Every decision for a target date is based on rows with `target_date < current target_date`; same-source variants additionally restrict routing evidence to the current forecast source family. Post-hoc segment tables are diagnostic only and are not fed back into the row-level predictions.

## Data Window

Rows used: `{manifest['official_rows']}` scored forecast rows.

Full date range: `{manifest['first_target_date']}` to `{manifest['last_target_date']}`.

Configured late evaluation start: `{manifest['late_eval_start']}`.

Actual late evaluation range: `{manifest['late_eval_first_target_date']}` to `{manifest['late_eval_last_target_date']}`.

Late evaluation rows: `{manifest['late_eval_rows']}`.

Source counts: `{manifest['source_counts']}`.

Screen stage: `{manifest['screen_stage']}`.

## Leakage Contract

- All scored rows are earlier than `{CONFIRMATION_START.date()}`.
- 2024+ confirmation rows are not loaded, scored, selected on, or used for routing.
- Each candidate uses the same prior-only router as 0041: `target_date < current target_date`.
- Same-source candidates isolate source-family history.
- History-threshold tests only change how much prior evidence is required; they do not use future labels.
- Family-inclusion tests remove or add candidate families but never use the target value to choose a family for the current row.
- Segment scoreboards are post-run diagnostics and are not used to generate row-level predictions.
- The stable scored archive is still non-contiguous, so late-window claims apply to the actual available 2021-2023 rows, not to a seamless 2018-2023 block.

## Main Result

{best_text}

## What Was Tested

The sensitivity grid has `{manifest['sensitivity_candidates']}` candidates. Context tests compare global, source-only, text-state, source-plus-text, source-plus-revision-action, correction-activity, prediction-disagreement, and compact combined states. Routing-mode tests compare best-family selection, inverse-MAE blending, positive-lift blending, and anchor-lift blending. History-threshold tests vary global and bucket evidence requirements from `{HISTORY_THRESHOLDS[0]}` through `{HISTORY_THRESHOLDS[-1]}`. Family-inclusion tests remove the hard residual family, remove one smooth champion at a time, use only hard residuals, use only smooth specialists, and add the broader selected smooth set.

The point is not just to find a slightly smaller MAE. The point is to learn whether the current 0041 router is stable enough to promote into a hardened system, or whether its gain is dependent on one fragile context, one history threshold, or one source family. A robust result should keep beating the 0038 anchor and the 0040 single-family champions across both the full scored frame and the actual late RSS-era rows.

## Baseline Comparison

{markdown_table(comparison, max_rows=30)}

## Family Scores

{markdown_table(family_scores, max_rows=20)}

## Sensitivity Leaders

{markdown_table(leaders, max_rows=40)}

## Robustness Summary

{markdown_table(robustness.head(50), max_rows=50)}

## Segment Scoreboard

{markdown_table(segments.head(80), max_rows=80)}

## Sensitivity Spec Catalog

{markdown_table(spec_catalog, max_rows=80)}

## Full Sensitivity Scoreboard

{markdown_table(scoreboard.head(80), max_rows=80)}

## Interpretation

The most useful interpretation is comparative. If the best 0042 candidate matches or beats 0041 while several nearby candidates also perform well, then the stack signal is broad and likely worth hardening further. If only one exact candidate wins and close variants collapse, the result is fragile and should not be treated as deployable. If removing the hard residual family or either smooth champion barely hurts, then the stack is over-complex and should be simplified. If source-specific segments disagree sharply, the next router should be source-specific rather than global. If stricter history thresholds improve stability, the current router may be benefiting from too little prior evidence early in the archive.

This experiment still cannot solve the biggest structural limitation: the scored forecast archive is non-contiguous. We have deep station history back to 1949+, but the current official forecast-scored frame used by these stack experiments is still 2000-2004 plus 2021-2023. That means these tests are leakage-safe but not a substitute for promoting the remaining 2005-2026 forecast archive into a continuous scored export. Until that happens, tiny changes under roughly a few thousandths of a degree should guide research direction rather than be treated as final production truth.
"""
    write_text(folder / "README.md", readme)


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Trust Router Sensitivity\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_trust_router_sensitivity.py`:

- `{FOLDER_NAME}`: targeted hardening of the 0041 prior-only stack across family inclusion, routing contexts, history thresholds, source behavior, and full-vs-late stability.

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Sensitivity candidates | {manifest['sensitivity_candidates']} |
| Best late sensitivity MAE | {manifest['best_late_eval_mae']} |
| Best late sensitivity delta vs anchor | {manifest['best_late_eval_delta_vs_anchor']} |
| Best full sensitivity MAE | {manifest['best_full_mae']} |
| Best full sensitivity delta vs anchor | {manifest['best_full_delta_vs_anchor']} |

Leakage contract: all candidate routing decisions use only earlier target dates; all scored rows are before `{CONFIRMATION_START.date()}`.
"""
    write_text(index_path, text)


def update_milestones(manifest: dict[str, object]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Milestones\n"
    section_marker = "\n## Trust Router Sensitivity\n"
    blockers_marker = "\n## Current Blockers And Gaps\n"
    next_marker = "\n## Exact Next Recommended Codex Task\n"
    if section_marker in existing:
        base, rest = existing.split(section_marker, 1)
        suffix = f"{blockers_marker}{rest.split(blockers_marker, 1)[1]}" if blockers_marker in rest else ""
    elif blockers_marker in existing:
        base, rest = existing.split(blockers_marker, 1)
        suffix = f"{blockers_marker}{rest}"
    else:
        base = existing.rstrip()
        suffix = ""

    section = f"""{base.rstrip()}
{section_marker}
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_trust_router_sensitivity.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Rows / candidates | Strongest current finding | Status |
|---|---:|---|---|
| Trust-router sensitivity | `{manifest['official_rows']}` rows; `{manifest['sensitivity_candidates']}` sensitivity candidates | Best late candidate `{manifest['best_late_candidate']}`: actual late MAE `{manifest['best_late_eval_mae']}`, late delta vs anchor `{manifest['best_late_eval_delta_vs_anchor']}`, full MAE `{manifest['best_late_candidate_full_mae']}`. Best full candidate `{manifest['best_full_candidate']}`: full MAE `{manifest['best_full_mae']}`, full delta vs anchor `{manifest['best_full_delta_vs_anchor']}`, actual late MAE `{manifest['best_full_candidate_late_eval_mae']}` | Audited |
| Stress dimensions | family inclusion, routing mode, history thresholds, routing context, source/time segments | Leaders and robustness tables are in the 0042 artifacts | Documented |
| Leakage guards | date-strict prior masks and zero 2024+ rows | Same-date cross-source rows are excluded from routing evidence; 2024+ remains locked | Guarded |

Interpretation: `0042` hardens the 0041 stack by checking whether the gain survives sensible perturbations. A robust improvement here strengthens the case for a simplified production-style router; fragility points to the next ablation rather than sealed confirmation.
"""
    blocker = (
        f"31. Trust-router sensitivity produced best full MAE `{manifest['best_full_mae']}` and best actual "
        f"late-window MAE `{manifest['best_late_eval_mae']}` on `{manifest['sensitivity_candidates']}` bounded "
        "candidates. The result remains limited by the non-contiguous scored forecast archive; 2024+ confirmation "
        "stays locked until explicitly commanded."
    )
    if blockers_marker in suffix and blocker not in suffix:
        before_next, after_next = suffix.split(next_marker, 1) if next_marker in suffix else (suffix, "")
        before_next = before_next.rstrip() + f"\n{blocker}\n"
        next_task = f"""{next_marker}

Implement `0043_router_simplification_or_archive_refresh_decision`: use the 0042 robustness tables to choose either a simpler source-specific trust router for the current non-contiguous frame, or prioritize promoting the remaining 2005-2026 forecast archive into a continuous scored export. Keep 2024+ locked.
"""
        suffix = before_next + "\n" + next_task if after_next else before_next
    section += suffix
    write_text(path, section)


def write_outputs(
    *,
    frame: pd.DataFrame,
    family_scores: pd.DataFrame,
    spec_catalog: pd.DataFrame,
    scoreboard: pd.DataFrame,
    predictions: pd.DataFrame,
    stack_0041_manifest: dict[str, object],
) -> dict[str, object]:
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    segments = segment_scoreboard(predictions)
    robust = robustness_summary(scoreboard, segments)
    leaders = variant_leaders(scoreboard)
    comparison = baseline_comparison(
        scoreboard=scoreboard,
        family_scores=family_scores,
        stack_0041_manifest=stack_0041_manifest,
    )
    write_csv(artifacts / "sensitivity_spec_catalog.csv", spec_catalog)
    write_csv(artifacts / "sensitivity_scoreboard.csv", scoreboard)
    write_csv(artifacts / "sensitivity_predictions.csv", predictions)
    write_csv(artifacts / "segment_scoreboard.csv", segments)
    write_csv(artifacts / "robustness_summary.csv", robust)
    write_csv(artifacts / "variant_leaders.csv", leaders)
    write_csv(artifacts / "family_scoreboard.csv", family_scores)
    write_csv(artifacts / "baseline_comparison.csv", comparison)
    top_ids = scoreboard.head(10)["candidate_id"].to_list() if not scoreboard.empty else []
    write_csv(
        artifacts / "top_sensitivity_predictions.csv",
        predictions[predictions["candidate_id"].isin(top_ids)].copy() if top_ids else predictions.head(0),
    )

    best_late = scoreboard.iloc[0] if not scoreboard.empty else None
    best_full = scoreboard.sort_values(["mae", "late_eval_mae", "rmse"]).iloc[0] if not scoreboard.empty else None
    late_mask = pd.to_datetime(frame["target_date"]) >= LATE_EVAL_START
    late_frame = frame[late_mask].copy()
    manifest = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "screen_stage": SCREEN_STAGE,
        "official_rows": int(len(frame)),
        "first_target_date": str(frame["target_date"].min().date()),
        "last_target_date": str(frame["target_date"].max().date()),
        "late_eval_start": str(LATE_EVAL_START.date()),
        "late_eval_first_target_date": "" if late_frame.empty else str(late_frame["target_date"].min().date()),
        "late_eval_last_target_date": "" if late_frame.empty else str(late_frame["target_date"].max().date()),
        "late_eval_rows": int(late_mask.sum()),
        "source_counts": {str(k): int(v) for k, v in frame["forecast_source_family"].value_counts().to_dict().items()},
        "sensitivity_candidates": int(len(scoreboard)),
        "best_late_candidate": "" if best_late is None else str(best_late["candidate_id"]),
        "best_late_candidate_full_mae": None if best_late is None else float(best_late["mae"]),
        "best_late_eval_mae": None if best_late is None else float(best_late["late_eval_mae"]),
        "best_late_eval_delta_vs_anchor": None if best_late is None else float(best_late["late_eval_delta_vs_anchor"]),
        "best_full_candidate": "" if best_full is None else str(best_full["candidate_id"]),
        "best_full_mae": None if best_full is None else float(best_full["mae"]),
        "best_full_delta_vs_anchor": None if best_full is None else float(best_full["delta_vs_anchor"]),
        "best_full_candidate_late_eval_mae": None if best_full is None else float(best_full["late_eval_mae"]),
        "anchor_full_mae": float(family_scores.loc[family_scores["family_name"].eq("anchor_0038_c"), "mae"].iloc[0]),
        "anchor_late_eval_mae": float(family_scores.loc[family_scores["family_name"].eq("anchor_0038_c"), "late_eval_mae"].iloc[0]),
        "confirmation_start": str(CONFIRMATION_START.date()),
    }
    write_json(RESEARCH_ROOT / "trust_router_sensitivity_manifest.json", manifest)
    write_readme(
        folder=folder,
        manifest=manifest,
        spec_catalog=spec_catalog,
        family_scores=family_scores,
        scoreboard=scoreboard,
        leaders=leaders,
        robustness=robust,
        segments=segments,
        comparison=comparison,
    )
    update_master_index(manifest)
    update_milestones(manifest)
    return manifest


def run() -> dict[str, object]:
    stack_0041_manifest = load_json(STACK_0041_MANIFEST)
    frame, family_catalog, meta_catalog = build_stack_frame()
    require_no_confirmation_dates(frame["target_date"], context="0042 sensitivity frame")
    spec_catalog = build_sensitivity_specs(family_catalog, meta_catalog)
    family_scores = family_scoreboard(frame, family_catalog)
    scoreboard, predictions = run_sensitivity_screen(frame, spec_catalog)
    require_no_confirmation_dates(predictions["target_date"], context="0042 sensitivity predictions")
    return write_outputs(
        frame=frame,
        family_scores=family_scores,
        spec_catalog=spec_catalog,
        scoreboard=scoreboard,
        predictions=predictions,
        stack_0041_manifest=stack_0041_manifest,
    )


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 trust-router sensitivity checks.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
