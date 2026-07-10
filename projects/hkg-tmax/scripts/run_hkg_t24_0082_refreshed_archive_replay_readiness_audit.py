from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_beastmode_signal_discovery import (
    CONFIRMATION_START,
    DATASETS_ROOT,
    RESEARCH_ROOT,
    markdown_table,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_forecast_archive_continuous_scored_export import (
    SCORED_EXPORT_MANIFEST_PATH,
    contiguous_date_segments,
    gap_rows_from_segments,
)
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import update_markdown_section

FOLDER_NAME = "0082_refreshed_archive_replay_readiness_audit"
OFFICIAL_SCORED_PATH = (
    DATASETS_ROOT / "05_hko_historical_rss_forecasts" / "hko_official_t15_scored_pre2024.parquet"
)


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: str
    path: Path
    role: str
    replay_required: bool


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def artifact_specs() -> list[ArtifactSpec]:
    return [
        ArtifactSpec(
            artifact_id="0042_trust_router_sensitivity",
            path=RESEARCH_ROOT
            / "0042_trust_router_sensitivity"
            / "artifacts"
            / "top_sensitivity_predictions.csv",
            role="older official-family router sensitivity artifact; direct dependency of 0049",
            replay_required=True,
        ),
        ArtifactSpec(
            artifact_id="0048_gated_residual_specialist_screen",
            path=RESEARCH_ROOT
            / "0048_gated_residual_specialist_screen"
            / "artifacts"
            / "sample_candidate_predictions.csv",
            role="official residual specialist screen using the scored official archive",
            replay_required=True,
        ),
        ArtifactSpec(
            artifact_id="0049_router_gate_stack_screen",
            path=RESEARCH_ROOT
            / "0049_router_gate_stack_screen"
            / "artifacts"
            / "top_predictions.csv",
            role="stack screen combining 0042 router predictions and 0048 specialist predictions",
            replay_required=True,
        ),
        ArtifactSpec(
            artifact_id="0067_station_official_family_router_common_frame",
            path=RESEARCH_ROOT
            / "0067_station_official_family_router"
            / "artifacts"
            / "common_frame.csv",
            role="later router common frame used by downstream prior-calibrated screens",
            replay_required=True,
        ),
        ArtifactSpec(
            artifact_id="0067_station_official_family_router_top_predictions",
            path=RESEARCH_ROOT
            / "0067_station_official_family_router"
            / "artifacts"
            / "top_predictions.csv",
            role="later router top-prediction artifact used by downstream prior-calibrated screens",
            replay_required=True,
        ),
        ArtifactSpec(
            artifact_id="0078_prior_only_residual_specialists",
            path=RESEARCH_ROOT
            / "0078_prior_only_residual_specialists"
            / "artifacts"
            / "top_predictions.csv",
            role="previous deployable research screen and direct 0080/0081 baseline",
            replay_required=True,
        ),
        ArtifactSpec(
            artifact_id="0081_rss_gate_stability_stress",
            path=RESEARCH_ROOT
            / "0081_rss_gate_stability_stress"
            / "artifacts"
            / "top_predictions.csv",
            role="current deployable research screen",
            replay_required=True,
        ),
    ]


def date_series(frame: pd.DataFrame, column: str = "target_date") -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(frame[column], errors="coerce").dt.normalize().dropna()


def date_set(frame: pd.DataFrame, column: str = "target_date") -> set[pd.Timestamp]:
    return set(date_series(frame, column).drop_duplicates())


def date_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(pd.Timestamp(value).date().isoformat())


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported artifact extension: {path}")


def source_col_for(frame: pd.DataFrame) -> str | None:
    for candidate in ("forecast_source_family", "source_id", "source"):
        if candidate in frame.columns:
            return candidate
    return None


def unique_date_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "target_date" not in frame.columns:
        return pd.DataFrame(columns=["target_date", "forecast_source_family"])
    source_col = source_col_for(frame)
    out = pd.DataFrame({"target_date": date_series(frame)})
    if source_col is not None:
        source_values = frame.loc[out.index, source_col].astype(str).to_numpy()
        out["forecast_source_family"] = source_values
    else:
        out["forecast_source_family"] = ""
    return (
        out.dropna(subset=["target_date"])
        .sort_values(["target_date", "forecast_source_family"])
        .drop_duplicates(["target_date", "forecast_source_family"])
        .reset_index(drop=True)
    )


def coverage_row(
    *,
    artifact_id: str,
    frame: pd.DataFrame,
    official_dates: set[pd.Timestamp],
    path: Path,
    role: str,
    replay_required: bool,
) -> dict[str, object]:
    dates = date_series(frame)
    unique_dates = set(dates.drop_duplicates())
    confirmation_rows = int(dates.ge(CONFIRMATION_START).sum())
    official_intersection = official_dates.intersection(unique_dates)
    missing_official = official_dates.difference(unique_dates)
    extra_dates = unique_dates.difference(official_dates)
    if confirmation_rows:
        status = "leakage_failed"
    elif not unique_dates:
        status = "no_target_dates"
    elif missing_official:
        status = "stale_partial_frame"
    elif extra_dates:
        status = "covers_official_plus_extra_dates"
    else:
        status = "covers_refreshed_official_frame"
    first_date = date_text(dates.min()) if not dates.empty else ""
    last_date = date_text(dates.max()) if not dates.empty else ""
    coverage_ratio = len(official_intersection) / len(official_dates) if official_dates else math.nan
    return {
        "artifact_id": artifact_id,
        "path": str(path),
        "role": role,
        "replay_required": replay_required,
        "rows": int(len(frame)),
        "unique_target_days": int(len(unique_dates)),
        "first_target_date": first_date,
        "last_target_date": last_date,
        "official_days_covered": int(len(official_intersection)),
        "official_days_missing": int(len(missing_official)),
        "artifact_days_not_in_official_export": int(len(extra_dates)),
        "official_coverage_ratio": float(coverage_ratio) if not math.isnan(coverage_ratio) else None,
        "confirmation_rows": confirmation_rows,
        "replay_status": status,
    }


def missing_official_dates(official: pd.DataFrame, artifact_frame: pd.DataFrame) -> pd.DataFrame:
    artifact_dates = date_set(artifact_frame)
    official_unique = unique_date_frame(official)
    out = official_unique[~official_unique["target_date"].isin(artifact_dates)].copy()
    out["target_date"] = out["target_date"].dt.date.astype(str)
    return out.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True)


def source_coverage(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    if frame.empty or "target_date" not in frame.columns:
        return pd.DataFrame(
            columns=[
                "artifact_id",
                "forecast_source_family",
                "rows",
                "unique_target_days",
                "first_target_date",
                "last_target_date",
            ]
        )
    source_col = source_col_for(frame)
    working = frame.copy()
    if source_col is None:
        working["forecast_source_family"] = "unknown"
        source_col = "forecast_source_family"
    rows: list[dict[str, object]] = []
    for source, group in working.groupby(source_col, observed=True, dropna=False):
        dates = date_series(group)
        rows.append(
            {
                "artifact_id": label,
                "forecast_source_family": str(source),
                "rows": int(len(group)),
                "unique_target_days": int(dates.nunique()),
                "first_target_date": date_text(dates.min()) if not dates.empty else "",
                "last_target_date": date_text(dates.max()) if not dates.empty else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["artifact_id", "forecast_source_family"]).reset_index(drop=True)


def build_leakage_audit(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for artifact_id, frame in frames.items():
        dates = date_series(frame)
        confirmation_rows = int(dates.ge(CONFIRMATION_START).sum()) if not dates.empty else 0
        rows.append(
            {
                "artifact_id": artifact_id,
                "rows": int(len(frame)),
                "unique_target_days": int(dates.nunique()) if not dates.empty else 0,
                "max_target_date": date_text(dates.max()) if not dates.empty else "",
                "confirmation_start": CONFIRMATION_START.date().isoformat(),
                "confirmation_rows": confirmation_rows,
                "status": "PASS" if confirmation_rows == 0 else "FAIL",
            }
        )
    return pd.DataFrame(rows).sort_values("artifact_id").reset_index(drop=True)


def dependency_gap_rows(coverage: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in coverage.itertuples(index=False):
        if row.replay_status == "covers_refreshed_official_frame":
            blocker = ""
            action = "ready for replay on refreshed official frame"
        elif row.replay_status == "stale_partial_frame":
            blocker = (
                f"missing {row.official_days_missing} official target days from the refreshed export; "
                f"artifact covers {row.unique_target_days} unique days"
            )
            action = "regenerate this dependency before claiming refreshed-frame champion performance"
        elif row.replay_status == "leakage_failed":
            blocker = "contains 2024+ confirmation target dates"
            action = "stop and repair leakage before any scoring"
        else:
            blocker = f"status is {row.replay_status}"
            action = "inspect artifact before replay"
        rows.append(
            {
                "artifact_id": row.artifact_id,
                "replay_status": row.replay_status,
                "required_for_0081_replay": bool(row.replay_required),
                "blocker": blocker,
                "required_action": action,
            }
        )
    return pd.DataFrame(rows)


def official_segments_and_gaps(official: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    segments = contiguous_date_segments(date_series(official))
    gaps = gap_rows_from_segments(segments)
    return segments, gaps


def summary_from_outputs(
    *,
    generated_at: str,
    official: pd.DataFrame,
    coverage: pd.DataFrame,
    leakage: pd.DataFrame,
    missing_for_current: pd.DataFrame,
    segments: pd.DataFrame,
    gaps: pd.DataFrame,
) -> dict[str, object]:
    official_dates = date_series(official)
    current = coverage[coverage["artifact_id"].eq("0081_rss_gate_stability_stress")]
    current_row = current.iloc[0].to_dict() if not current.empty else {}
    required = coverage[coverage["replay_required"].astype(bool)].copy()
    stale_required = required[required["replay_status"].ne("covers_refreshed_official_frame")]
    leak_failures = leakage[leakage["status"].eq("FAIL")]
    if not leak_failures.empty:
        replay_status = "blocked_by_confirmation_leakage"
    elif not stale_required.empty:
        replay_status = "blocked_by_stale_downstream_router_frame"
    else:
        replay_status = "ready_to_replay_current_chain"
    largest_gap_days = int(gaps["missing_days"].max()) if not gaps.empty else 0
    largest_gap = (
        gaps.sort_values("missing_days", ascending=False).iloc[0].to_dict() if not gaps.empty else {}
    )
    return {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "official_scored_path": str(OFFICIAL_SCORED_PATH),
        "official_manifest_path": str(SCORED_EXPORT_MANIFEST_PATH),
        "official_rows": int(len(official)),
        "official_unique_target_days": int(official_dates.nunique()),
        "official_first_target_date": date_text(official_dates.min()) if not official_dates.empty else "",
        "official_last_target_date": date_text(official_dates.max()) if not official_dates.empty else "",
        "official_segment_count": int(len(segments)),
        "official_largest_gap_days": largest_gap_days,
        "official_largest_gap": largest_gap,
        "current_champion_unique_target_days": int(current_row.get("unique_target_days", 0) or 0),
        "current_champion_official_days_missing": int(current_row.get("official_days_missing", 0) or 0),
        "newly_available_official_days_not_in_current_champion": int(len(missing_for_current)),
        "replay_status": replay_status,
        "stale_required_dependency_count": int(len(stale_required)),
        "leakage_failures": int(len(leak_failures)),
        "uses_2024_plus_rows": bool(len(leak_failures) > 0),
        "next_recommended_task": (
            "Run 0083 as an expanded-frame official-anchor replay: either regenerate the 0038-0042 router "
            "chain against the refreshed 0044 export, or build a focused replacement benchmark that scores "
            "0078/0081-style rules on every refreshed non-2024 official target date without opening 2024+."
        ),
    }


def build_readme(
    *,
    generated_at: str,
    summary: dict[str, object],
    coverage: pd.DataFrame,
    source_table: pd.DataFrame,
    dependency_gaps: pd.DataFrame,
    segments: pd.DataFrame,
    gaps: pd.DataFrame,
    missing_for_current: pd.DataFrame,
    leakage: pd.DataFrame,
) -> str:
    largest_missing_preview = missing_for_current.head(25)
    segment_text = "; ".join(
        f"{row.first_date} to {row.last_date} ({row.observed_days} days)"
        for row in segments.itertuples(index=False)
    )
    if not segment_text:
        segment_text = "no usable segments"
    return f"""# 0082 Refreshed Archive Replay Readiness Audit

Generated: `{generated_at}`

## Purpose

The HKO official forecast backfill is still moving. The local scored export has already advanced beyond the old champion experiment frame, but the champion chain cannot be assumed to have used those rows. This audit answers one narrow question: can the current `0081` champion chain be honestly re-scored on the refreshed official forecast export right now, without regenerating its upstream dependencies?

The answer is **no**. The refreshed official scored export has `{summary['official_unique_target_days']}` unique pre-2024 target days from `{summary['official_first_target_date']}` to `{summary['official_last_target_date']}`, while the current `0081` champion artifact still has `{summary['current_champion_unique_target_days']}` unique target days. There are `{summary['newly_available_official_days_not_in_current_champion']}` official target days now present in the refreshed export that are not represented in the current champion predictions. Therefore the correct action is not to claim a better or worse champion yet. The correct action is to regenerate or replace the stale router chain and then score the refreshed frame.

## Main Status

| Field | Value |
|---|---|
| Replay status | `{summary['replay_status']}` |
| Official scored rows | `{summary['official_rows']}` |
| Official unique target days | `{summary['official_unique_target_days']}` |
| Official date range | `{summary['official_first_target_date']}` to `{summary['official_last_target_date']}` |
| Current champion unique target days | `{summary['current_champion_unique_target_days']}` |
| Official days missing from current champion | `{summary['current_champion_official_days_missing']}` |
| Newly available official days not in current champion | `{summary['newly_available_official_days_not_in_current_champion']}` |
| Stale required dependency count | `{summary['stale_required_dependency_count']}` |
| 2024+ leakage failures | `{summary['leakage_failures']}` |

## Artifact Coverage

{markdown_table(coverage, max_rows=20)}

## Source Coverage

{markdown_table(source_table, max_rows=30)}

## Dependency Gap Audit

{markdown_table(dependency_gaps, max_rows=20)}

## Refreshed Official Segments

{markdown_table(segments, max_rows=20)}

## Remaining Official Gaps

{markdown_table(gaps.sort_values('missing_days', ascending=False) if not gaps.empty else gaps, max_rows=20)}

## Example Official Dates Missing From Current Champion

{markdown_table(largest_missing_preview, max_rows=25)}

## Leakage Audit

{markdown_table(leakage, max_rows=20)}

## Interpretation

This is an important but unglamorous result. The data backfill has made progress: the refreshed official export now has these usable pre-2024 segments: {segment_text}. But the deployed research chain is still constrained by older artifacts, especially the router/sensitivity family that was built before those newly parsed press rows existed. If we simply reran the last stage or quoted the current `0081` number as though it covered the expanded archive, the result would be misleading.

The leakage-safe conclusion is:

1. The refreshed official export is usable as an input.
2. The current champion score remains valid only for the old frame it actually predicted.
3. The current champion score is not yet a refreshed-frame result.
4. The next experiment must replay or replace the stale official-router dependency chain on the expanded pre-2024 frame.

## Exact Next Recommended Task

{summary['next_recommended_task']}
"""


def update_milestones(summary: dict[str, object], coverage: pd.DataFrame) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    required = coverage[coverage["replay_required"].astype(bool)].copy()
    stale = required[required["replay_status"].ne("covers_refreshed_official_frame")]
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0082_refreshed_archive_replay_readiness_audit.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Current evidence | Status |
|---|---|---|
| Refreshed official scored export | `{summary['official_unique_target_days']}` unique target days, `{summary['official_first_target_date']}` to `{summary['official_last_target_date']}` | Usable input |
| Current `0081` champion frame | `{summary['current_champion_unique_target_days']}` unique target days | Still old-frame only |
| Official days missing from current champion | `{summary['current_champion_official_days_missing']}` | Replay required |
| Required stale dependencies | `{len(stale)}` | `{summary['replay_status']}` |
| Leakage audit | `{summary['leakage_failures']}` 2024+ failures | PASS if zero |

Interpretation: the forecast archive backfill has advanced the local scored official export, but the current `0081` champion cannot yet be treated as an expanded-frame result. Older router artifacts still restrict the downstream chain to a partial frame. The next task is to regenerate or replace that stale official-router chain on the refreshed pre-2024 export, while keeping 2024+ sealed.
"""
    update_markdown_section(
        path,
        heading="0082 Refreshed Forecast Archive Replay Readiness Audit",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    next_section = """
Implement `0083_expanded_frame_official_anchor_replay`: use the current refreshed `0044` official scored export as input, keep 2024+ sealed, and either regenerate the stale `0038`-through-`0042` official-router chain or build a focused replacement benchmark that evaluates the current champion family on all available refreshed pre-2024 official target dates. Do not resume Polymarket, production, or 2024+ confirmation scoring.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_section)


def run() -> dict[str, object]:
    generated_at = now_utc()
    if not OFFICIAL_SCORED_PATH.exists():
        raise FileNotFoundError(f"Missing refreshed official scored export: {OFFICIAL_SCORED_PATH}")

    official = pd.read_parquet(OFFICIAL_SCORED_PATH)
    official_dates = date_set(official)
    frames: dict[str, pd.DataFrame] = {"0044_refreshed_official_scored_export": official}
    coverage_rows: list[dict[str, object]] = []
    source_frames = [source_coverage(official, label="0044_refreshed_official_scored_export")]

    for spec in artifact_specs():
        if not spec.path.exists():
            empty = pd.DataFrame()
            frames[spec.artifact_id] = empty
            coverage_rows.append(
                {
                    "artifact_id": spec.artifact_id,
                    "path": str(spec.path),
                    "role": spec.role,
                    "replay_required": spec.replay_required,
                    "rows": 0,
                    "unique_target_days": 0,
                    "first_target_date": "",
                    "last_target_date": "",
                    "official_days_covered": 0,
                    "official_days_missing": int(len(official_dates)),
                    "artifact_days_not_in_official_export": 0,
                    "official_coverage_ratio": 0.0,
                    "confirmation_rows": 0,
                    "replay_status": "missing_artifact",
                }
            )
            continue
        frame = read_table(spec.path)
        frames[spec.artifact_id] = frame
        coverage_rows.append(
            coverage_row(
                artifact_id=spec.artifact_id,
                frame=frame,
                official_dates=official_dates,
                path=spec.path,
                role=spec.role,
                replay_required=spec.replay_required,
            )
        )
        source_frames.append(source_coverage(frame, label=spec.artifact_id))

    coverage = pd.DataFrame(coverage_rows).sort_values("artifact_id").reset_index(drop=True)
    source_table = pd.concat(source_frames, ignore_index=True) if source_frames else pd.DataFrame()
    dependency_gaps = dependency_gap_rows(coverage)
    leakage = build_leakage_audit(frames)
    current_frame = frames.get("0081_rss_gate_stability_stress", pd.DataFrame())
    missing_for_current = missing_official_dates(official, current_frame)
    segments, gaps = official_segments_and_gaps(official)
    summary = summary_from_outputs(
        generated_at=generated_at,
        official=official,
        coverage=coverage,
        leakage=leakage,
        missing_for_current=missing_for_current,
        segments=segments,
        gaps=gaps,
    )

    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    write_csv(artifacts / "coverage_comparison.csv", coverage)
    write_csv(artifacts / "source_coverage.csv", source_table)
    write_csv(artifacts / "dependency_gap_audit.csv", dependency_gaps)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_csv(artifacts / "official_segments.csv", segments)
    write_csv(artifacts / "official_gaps.csv", gaps)
    write_csv(artifacts / "missing_current.csv", missing_for_current)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "refreshed_archive_replay_readiness_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            generated_at=generated_at,
            summary=summary,
            coverage=coverage,
            source_table=source_table,
            dependency_gaps=dependency_gaps,
            segments=segments,
            gaps=gaps,
            missing_for_current=missing_for_current,
            leakage=leakage,
        ),
    )
    update_milestones(summary, coverage)
    if bool(summary["uses_2024_plus_rows"]):
        raise RuntimeError("0082 found 2024+ confirmation rows in at least one required frame")
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Audit whether refreshed official forecast export can replay the current HKG Tmax champion chain."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
