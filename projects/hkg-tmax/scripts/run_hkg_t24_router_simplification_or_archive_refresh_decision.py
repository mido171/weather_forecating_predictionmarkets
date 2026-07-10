from __future__ import annotations

import argparse
import json
import re
import sys
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
    write_csv,
    write_json,
    write_text,
)

FOLDER_NAME = "0043_router_simplification_or_archive_refresh_decision"
SENSITIVITY_FOLDER = RESEARCH_ROOT / "0042_trust_router_sensitivity"
SENSITIVITY_ARTIFACTS = SENSITIVITY_FOLDER / "artifacts"
SENSITIVITY_MANIFEST = RESEARCH_ROOT / "trust_router_sensitivity_manifest.json"
STACK_0041_MANIFEST = RESEARCH_ROOT / "station_network_forecast_stack_manifest.json"
PRESS_GAP_MANIFEST = RESEARCH_ROOT / "press_archive_raw_detail_gap_audit_manifest.json"
PRESS_GAP_ARTIFACTS = RESEARCH_ROOT / "0024_press_gap_audit" / "artifacts"
SCREEN_STAGE = "stage1_decision_gate"
MIN_SIGNIFICANT_MAE_GAIN_C = 0.0020
MIN_CONTINUITY_RATIO_FOR_ROUTER_FIRST = 0.70
MIN_SCORED_ROWS_FOR_ROUTER_FIRST = 5000


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 140) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def inclusive_calendar_days(first_date: object, last_date: object) -> int:
    first = pd.to_datetime(first_date, errors="raise").normalize()
    last = pd.to_datetime(last_date, errors="raise").normalize()
    if last < first:
        return 0
    return int((last - first).days) + 1


def unique_target_day_coverage(predictions: pd.DataFrame) -> dict[str, object]:
    dates = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(dates, context="0043 decision target-date audit")
    unique_dates = dates.dropna().drop_duplicates().sort_values()
    if unique_dates.empty:
        return {
            "first_target_date": "",
            "last_target_date": "",
            "unique_target_days": 0,
            "expected_calendar_days": 0,
            "continuity_ratio": 0.0,
            "missing_calendar_days": 0,
        }
    expected = inclusive_calendar_days(unique_dates.min(), unique_dates.max())
    unique_count = int(len(unique_dates))
    return {
        "first_target_date": str(unique_dates.min().date()),
        "last_target_date": str(unique_dates.max().date()),
        "unique_target_days": unique_count,
        "expected_calendar_days": expected,
        "continuity_ratio": float(unique_count / expected) if expected else 0.0,
        "missing_calendar_days": int(expected - unique_count),
    }


def press_archive_gap_summary(gap_manifest: dict[str, object]) -> dict[str, object]:
    raw_years = [int(year) for year in gap_manifest.get("raw_detail_years", [])]
    zero_raw_years = [int(year) for year in gap_manifest.get("zero_raw_detail_years", [])]
    first_year = int(gap_manifest.get("candidate_first_year") or 2000)
    last_year = int(gap_manifest.get("candidate_last_year") or 2026)
    candidate_year_count = max(0, last_year - first_year + 1)
    raw_year_count = len(set(raw_years))
    return {
        "candidate_first_year": first_year,
        "candidate_last_year": last_year,
        "candidate_year_count": candidate_year_count,
        "raw_detail_year_count": raw_year_count,
        "raw_detail_year_coverage_ratio": float(raw_year_count / candidate_year_count) if candidate_year_count else 0.0,
        "zero_raw_detail_year_count": len(set(zero_raw_years)),
        "zero_raw_detail_years": ",".join(str(year) for year in zero_raw_years),
        "first_scoreable_target_date": str(gap_manifest.get("first_scoreable_target_date", "")),
        "last_scoreable_target_date": str(gap_manifest.get("last_scoreable_target_date", "")),
        "scoreable_forecast_day_rows": int(gap_manifest.get("scoreable_forecast_day_rows") or 0),
    }


def best_balanced_router(robustness: pd.DataFrame) -> pd.Series:
    candidates = robustness.copy()
    candidates["segments_scored"] = pd.to_numeric(candidates["segments_scored"], errors="coerce")
    candidates["segments_beating_anchor"] = pd.to_numeric(candidates["segments_beating_anchor"], errors="coerce")
    candidates["worst_delta_vs_anchor"] = pd.to_numeric(candidates["worst_delta_vs_anchor"], errors="coerce")
    stable = candidates[
        (candidates["segments_scored"] > 0)
        & (candidates["segments_beating_anchor"] == candidates["segments_scored"])
        & (candidates["worst_delta_vs_anchor"] < 0.0)
    ].copy()
    if stable.empty:
        stable = candidates.copy()
    stable["mae"] = pd.to_numeric(stable["mae"], errors="coerce")
    stable["late_eval_mae"] = pd.to_numeric(stable["late_eval_mae"], errors="coerce")
    stable["source_mae_spread"] = pd.to_numeric(stable.get("source_mae_spread"), errors="coerce")
    return stable.sort_values(["mae", "late_eval_mae", "source_mae_spread"], na_position="last").iloc[0]


def decision_branch(
    *,
    continuity_ratio: float,
    scored_rows: int,
    robust_full_mae_gain_vs_0041: float,
    robust_late_mae_gain_vs_0041_best_late: float,
    robust_candidate_beats_all_segments: bool,
) -> tuple[str, str]:
    low_continuity = continuity_ratio < MIN_CONTINUITY_RATIO_FOR_ROUTER_FIRST
    small_full_gain = robust_full_mae_gain_vs_0041 < MIN_SIGNIFICANT_MAE_GAIN_C
    small_or_negative_late_gain = robust_late_mae_gain_vs_0041_best_late < MIN_SIGNIFICANT_MAE_GAIN_C
    too_few_rows = scored_rows < MIN_SCORED_ROWS_FOR_ROUTER_FIRST
    if low_continuity and too_few_rows and small_full_gain and small_or_negative_late_gain:
        return (
            "archive_refresh_first_freeze_router_benchmark",
            "Router signal is stable but marginal on a small non-contiguous scored frame; expanding the forecast archive is the highest-leverage next move.",
        )
    if robust_candidate_beats_all_segments and not low_continuity and not too_few_rows:
        return (
            "simplify_router_first",
            "The robust router has enough continuity and segment stability to justify simplification before archive work.",
        )
    return (
        "archive_refresh_first_freeze_router_benchmark",
        "The router is useful as a benchmark, but current continuity/row-count evidence is not strong enough for router-first hardening.",
    )


def build_decision_artifacts() -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sensitivity_manifest = load_json(SENSITIVITY_MANIFEST)
    stack_manifest = load_json(STACK_0041_MANIFEST)
    press_gap_manifest = load_json(PRESS_GAP_MANIFEST)
    robustness = pd.read_csv(SENSITIVITY_ARTIFACTS / "robustness_summary.csv")
    scoreboard = pd.read_csv(SENSITIVITY_ARTIFACTS / "sensitivity_scoreboard.csv")
    predictions = pd.read_csv(SENSITIVITY_ARTIFACTS / "sensitivity_predictions.csv", usecols=["target_date"])

    target_coverage = unique_target_day_coverage(predictions)
    press_gap = press_archive_gap_summary(press_gap_manifest)
    balanced = best_balanced_router(robustness)
    best_late = scoreboard.sort_values(["late_eval_mae", "mae", "rmse"]).iloc[0]

    robust_full_gain_vs_0041 = float(stack_manifest["best_full_mae"]) - float(balanced["mae"])
    robust_late_gain_vs_0041_best_late = float(stack_manifest["best_late_eval_mae"]) - float(balanced["late_eval_mae"])
    robust_candidate_beats_all_segments = bool(
        int(balanced["segments_scored"]) > 0
        and int(balanced["segments_beating_anchor"]) == int(balanced["segments_scored"])
        and float(balanced["worst_delta_vs_anchor"]) < 0.0
    )
    decision, reason = decision_branch(
        continuity_ratio=float(target_coverage["continuity_ratio"]),
        scored_rows=int(sensitivity_manifest["official_rows"]),
        robust_full_mae_gain_vs_0041=robust_full_gain_vs_0041,
        robust_late_mae_gain_vs_0041_best_late=robust_late_gain_vs_0041_best_late,
        robust_candidate_beats_all_segments=robust_candidate_beats_all_segments,
    )

    decision_summary = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "screen_stage": SCREEN_STAGE,
        "decision": decision,
        "decision_reason": reason,
        "recommended_next_task": "0044_promote_2005_2026_forecast_archive_to_continuous_scored_export"
        if decision.startswith("archive_refresh")
        else "0044_simplify_source_specific_router",
        "router_freeze_candidate": str(balanced["candidate_id"]),
        "router_freeze_full_mae": float(balanced["mae"]),
        "router_freeze_late_eval_mae": float(balanced["late_eval_mae"]),
        "router_freeze_rmse": float(balanced["rmse"]),
        "router_segments_scored": int(balanced["segments_scored"]),
        "router_segments_beating_anchor": int(balanced["segments_beating_anchor"]),
        "router_worst_delta_vs_anchor": float(balanced["worst_delta_vs_anchor"]),
        "best_late_candidate": str(best_late["candidate_id"]),
        "best_late_candidate_full_mae": float(best_late["mae"]),
        "best_late_candidate_late_eval_mae": float(best_late["late_eval_mae"]),
        "best_late_candidate_segments_note": "not selected as benchmark because full-window MAE is weaker",
        "0041_best_full_mae": float(stack_manifest["best_full_mae"]),
        "0041_best_late_eval_mae": float(stack_manifest["best_late_eval_mae"]),
        "robust_full_mae_gain_vs_0041_c": robust_full_gain_vs_0041,
        "robust_late_mae_gain_vs_0041_best_late_c": robust_late_gain_vs_0041_best_late,
        "official_rows": int(sensitivity_manifest["official_rows"]),
        "first_target_date": str(sensitivity_manifest["first_target_date"]),
        "last_target_date": str(sensitivity_manifest["last_target_date"]),
        "target_unique_days": int(target_coverage["unique_target_days"]),
        "target_expected_calendar_days": int(target_coverage["expected_calendar_days"]),
        "target_continuity_ratio": float(target_coverage["continuity_ratio"]),
        "target_missing_calendar_days": int(target_coverage["missing_calendar_days"]),
        "press_candidate_first_year": press_gap["candidate_first_year"],
        "press_candidate_last_year": press_gap["candidate_last_year"],
        "press_raw_detail_year_coverage_ratio": float(press_gap["raw_detail_year_coverage_ratio"]),
        "press_zero_raw_detail_year_count": int(press_gap["zero_raw_detail_year_count"]),
        "press_first_scoreable_target_date": press_gap["first_scoreable_target_date"],
        "press_last_scoreable_target_date": press_gap["last_scoreable_target_date"],
        "confirmation_start": str(CONFIRMATION_START.date()),
    }

    continuity_rows = [
        {"metric": key, "value": value}
        for key, value in {
            **{f"target_{key}": value for key, value in target_coverage.items()},
            **{f"press_{key}": value for key, value in press_gap.items()},
        }.items()
    ]
    continuity = pd.DataFrame(continuity_rows)

    router_cols = [
        "candidate_id",
        "variant_kind",
        "comparison_group",
        "feature_set",
        "mode",
        "same_source",
        "family_group",
        "mae",
        "rmse",
        "late_eval_mae",
        "late_eval_delta_vs_anchor",
        "segments_scored",
        "segments_beating_anchor",
        "source_mae_spread",
        "worst_delta_vs_anchor",
    ]
    router_shortlist = robustness[router_cols].copy()
    router_shortlist["all_segments_beat_anchor"] = (
        router_shortlist["segments_scored"].astype(int) == router_shortlist["segments_beating_anchor"].astype(int)
    ) & (pd.to_numeric(router_shortlist["worst_delta_vs_anchor"], errors="coerce") < 0.0)
    router_shortlist = router_shortlist.sort_values(
        ["all_segments_beat_anchor", "mae", "late_eval_mae"],
        ascending=[False, True, True],
    ).head(25)

    factors = pd.DataFrame(
        [
            {
                "factor": "robust_router_beats_anchor_in_all_segments",
                "value": robust_candidate_beats_all_segments,
                "threshold_or_rule": "required for any router freeze candidate",
                "disposition": "pass" if robust_candidate_beats_all_segments else "fail",
            },
            {
                "factor": "robust_full_mae_gain_vs_0041",
                "value": robust_full_gain_vs_0041,
                "threshold_or_rule": f">= {MIN_SIGNIFICANT_MAE_GAIN_C}",
                "disposition": "pass" if robust_full_gain_vs_0041 >= MIN_SIGNIFICANT_MAE_GAIN_C else "weak",
            },
            {
                "factor": "robust_late_mae_gain_vs_0041_best_late",
                "value": robust_late_gain_vs_0041_best_late,
                "threshold_or_rule": f">= {MIN_SIGNIFICANT_MAE_GAIN_C}",
                "disposition": "pass" if robust_late_gain_vs_0041_best_late >= MIN_SIGNIFICANT_MAE_GAIN_C else "weak_or_negative",
            },
            {
                "factor": "scored_frame_continuity_ratio",
                "value": target_coverage["continuity_ratio"],
                "threshold_or_rule": f">= {MIN_CONTINUITY_RATIO_FOR_ROUTER_FIRST}",
                "disposition": "pass"
                if float(target_coverage["continuity_ratio"]) >= MIN_CONTINUITY_RATIO_FOR_ROUTER_FIRST
                else "fail",
            },
            {
                "factor": "scored_rows",
                "value": int(sensitivity_manifest["official_rows"]),
                "threshold_or_rule": f">= {MIN_SCORED_ROWS_FOR_ROUTER_FIRST}",
                "disposition": "pass"
                if int(sensitivity_manifest["official_rows"]) >= MIN_SCORED_ROWS_FOR_ROUTER_FIRST
                else "fail",
            },
            {
                "factor": "press_archive_raw_year_coverage",
                "value": press_gap["raw_detail_year_coverage_ratio"],
                "threshold_or_rule": "high enough to support continuous 2005-2026 scored export",
                "disposition": "fail" if float(press_gap["raw_detail_year_coverage_ratio"]) < 0.70 else "pass",
            },
        ]
    )
    return decision_summary, continuity, router_shortlist, factors


def write_readme(
    *,
    folder: Path,
    decision: dict[str, object],
    continuity: pd.DataFrame,
    router_shortlist: pd.DataFrame,
    factors: pd.DataFrame,
) -> None:
    readme = f"""# Router Simplification Or Archive Refresh Decision

Generated: `{decision['generated_at_utc']}`

## Purpose

`0042` proved that the current prior-only trust router has real signal, but the broader objective is not to optimize a tiny non-contiguous analysis frame. The user goal is an elite HKG Tmax system that combines the deep 1949+ historical station/weather data with the 2000-2026 operational forecast archive. This decision gate asks the practical next question: should we spend the next engineering effort simplifying the current router, or should we prioritize promoting the remaining forecast archive into a continuous scored export?

## Decision

Decision: `{decision['decision']}`.

Recommended next task: `{decision['recommended_next_task']}`.

Reason: {decision['decision_reason']}

The current router should be frozen only as a benchmark, not treated as the main path toward the 0.45 MAE goal. The best robust router candidate is `{decision['router_freeze_candidate']}` with full MAE `{decision['router_freeze_full_mae']}`, RMSE `{decision['router_freeze_rmse']}`, and actual late-window MAE `{decision['router_freeze_late_eval_mae']}`. It beats the anchor in `{decision['router_segments_beating_anchor']}` of `{decision['router_segments_scored']}` scored diagnostic segments, so it is real signal. But its full-window improvement over the 0041 full champion is only `{decision['robust_full_mae_gain_vs_0041_c']}` C, and its late-window value does not beat the 0041 best-late champion by a meaningful amount.

## Why Archive Refresh Wins

The scored forecast frame used by the stack experiments has `{decision['official_rows']}` rows from `{decision['first_target_date']}` to `{decision['last_target_date']}`. Across that calendar span, only `{decision['target_unique_days']}` unique target days are present out of `{decision['target_expected_calendar_days']}` possible days, a continuity ratio of `{decision['target_continuity_ratio']}`. That means the router is being optimized on a narrow bridge between 2000-2004 press rows and 2021-2023 RSS rows, not on a continuous 2000-2023 operational forecast history.

The press archive audit is the stronger blocker: candidate coverage exists from `{decision['press_candidate_first_year']}` to `{decision['press_candidate_last_year']}`, but raw detail/scored export coverage is still materially concentrated in early years. The audit records `{decision['press_zero_raw_detail_year_count']}` zero-raw-detail years and scoreable press targets only through `{decision['press_last_scoreable_target_date']}`. Until 2005-2026 is promoted into a continuous scored export, deeper router tweaks are more likely to overfit archive shape than discover the path to 0.45 MAE.

## Leakage Contract

- No 2024+ confirmation rows are loaded or scored.
- This is a decision/reporting gate; it does not train a model.
- It reads 0042 scored development artifacts only.
- The router freeze candidate is selected from prior-only 0042 candidates already audited for `target_date < current target_date` routing.
- Segment and archive continuity metrics are diagnostic and are not fed back into any row-level forecast.

## Decision Factors

{markdown_table(factors, max_rows=20)}

## Archive Continuity Assessment

{markdown_table(continuity, max_rows=80)}

## Router Shortlist

{markdown_table(router_shortlist, max_rows=25)}

## Interpretation

This is the point where chasing another few ten-thousandths of MAE on the current 2,670-row non-contiguous frame becomes lower value than fixing the data shape. The best robust router is worth preserving as a benchmark because it beats the anchor in every scored diagnostic segment. It should be rerun after the forecast archive is continuous. The next engineering effort should make the 2005-2026 forecast archive scoreable, then rerun the official anchor, station-network residual screens, and trust-router stack over the expanded frame.
"""
    write_text(folder / "README.md", readme)


def update_master_index(decision: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Router Simplification Or Archive Refresh Decision\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{decision['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_router_simplification_or_archive_refresh_decision.py`:

- `{FOLDER_NAME}`: decision gate using 0042 trust-router robustness and forecast-archive continuity evidence.

| Metric | Value |
|---|---:|
| Decision | {decision['decision']} |
| Recommended next task | {decision['recommended_next_task']} |
| Router freeze full MAE | {decision['router_freeze_full_mae']} |
| Router freeze actual late MAE | {decision['router_freeze_late_eval_mae']} |
| Target continuity ratio | {decision['target_continuity_ratio']} |
| Press raw detail year coverage ratio | {decision['press_raw_detail_year_coverage_ratio']} |

Leakage contract: this is a no-training decision gate; all evidence remains before `{CONFIRMATION_START.date()}`.
"""
    write_text(index_path, text)


def update_milestones(decision: dict[str, object]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Milestones\n"
    section_marker = "\n## Router Simplification Or Archive Refresh Decision\n"
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
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_router_simplification_or_archive_refresh_decision.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Decision | Status |
|---|---|---|---|
| Router freeze benchmark | `{decision['router_freeze_candidate']}` full MAE `{decision['router_freeze_full_mae']}`, actual late MAE `{decision['router_freeze_late_eval_mae']}`, beats anchor in `{decision['router_segments_beating_anchor']}` / `{decision['router_segments_scored']}` segments | Freeze as benchmark, not as final path | Audited |
| Forecast archive continuity | target continuity ratio `{decision['target_continuity_ratio']}`; press raw detail year coverage `{decision['press_raw_detail_year_coverage_ratio']}`; scoreable press targets through `{decision['press_last_scoreable_target_date']}` | Prioritize archive refresh | Blocker |
| Final branch | Decision `{decision['decision']}` | Next task `{decision['recommended_next_task']}` | Selected |

Interpretation: `0043` chooses archive refresh over more router micro-optimization. The current router signal is real, but the scored forecast frame is too non-contiguous and too small to justify deeper trust-router tuning before promoting 2005-2026 forecast history into a continuous scored export.
"""
    blocker = (
        f"32. Router-vs-archive decision selected `{decision['decision']}`. The robust router benchmark is "
        f"`{decision['router_freeze_candidate']}` with full MAE `{decision['router_freeze_full_mae']}` and actual late "
        f"MAE `{decision['router_freeze_late_eval_mae']}`, but target continuity is only "
        f"`{decision['target_continuity_ratio']}` and press archive raw/scored continuity remains the dominant blocker."
    )
    if blockers_marker in suffix and blocker not in suffix:
        before_next, after_next = suffix.split(next_marker, 1) if next_marker in suffix else (suffix, "")
        before_next = before_next.rstrip() + f"\n{blocker}\n"
        next_task = f"""{next_marker}

Implement `0044_promote_2005_2026_forecast_archive_to_continuous_scored_export`: parse/promote the remaining HKO press/RSS forecast archive into a continuous, point-in-time scored forecast table, verify issue/target dates and no 2024+ confirmation access, then rerun the 0038-0042 anchor/residual/router stack on the expanded frame.
"""
        suffix = before_next + "\n" + next_task if after_next else before_next
    section += suffix
    write_text(path, section)


def write_outputs(
    *,
    decision: dict[str, object],
    continuity: pd.DataFrame,
    router_shortlist: pd.DataFrame,
    factors: pd.DataFrame,
) -> dict[str, object]:
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    write_csv(artifacts / "factors.csv", factors)
    write_csv(artifacts / "continuity.csv", continuity)
    write_csv(artifacts / "router_shortlist.csv", router_shortlist)
    write_csv(artifacts / "decision.csv", pd.DataFrame([decision]))
    write_json(RESEARCH_ROOT / "router_simplification_or_archive_refresh_decision_manifest.json", decision)
    write_json(artifacts / "decision.json", decision)
    write_readme(folder=folder, decision=decision, continuity=continuity, router_shortlist=router_shortlist, factors=factors)
    update_master_index(decision)
    update_milestones(decision)
    return decision


def run() -> dict[str, object]:
    decision, continuity, router_shortlist, factors = build_decision_artifacts()
    return write_outputs(decision=decision, continuity=continuity, router_shortlist=router_shortlist, factors=factors)


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 router-vs-archive decision gate.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
