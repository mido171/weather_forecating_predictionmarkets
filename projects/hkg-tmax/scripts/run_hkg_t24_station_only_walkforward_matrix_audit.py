from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    EVAL_END,
    EVAL_START,
    TRAIN_END,
    safe_corr,
    update_markdown_section,
)
from scripts.run_hkg_t24_station_contribution_atlas import (  # noqa: E402
    load_station_day_features,
    load_target,
)
from scripts.run_hkg_t24_station_regime_interaction_atlas import (  # noqa: E402
    build_feature_frame,
    station_token,
    unique_feature_id,
)

FOLDER_NAME = "0054_station_only_walkforward_matrix_audit"
INPUT_AUDIT_PATH = (
    RESEARCH_ROOT
    / "0053_candidate_timestamp_eligibility_audit"
    / "artifacts"
    / "candidate_timestamp_audit.csv"
)
STATION_ID_PATTERN = r"\d{5,6}-\d{5}"
PAIR_PATTERN = re.compile(rf"^(?P<a>{STATION_ID_PATTERN}) minus (?P<b>{STATION_ID_PATTERN}) (?P<attribute>.+)$")
SINGLE_PATTERN = re.compile(rf"^(?P<station>{STATION_ID_PATTERN}) (?P<feature>.+)$")
FORBIDDEN_FEATURE_TOKENS = (
    "target_tmax",
    "target_anomaly",
    "official_error",
    "official_abs_error",
    "forecast_max",
    "mae",
    "rmse",
    "residual",
)


@dataclass(frozen=True)
class ComponentSpec:
    source_family: str
    station_id: str
    station_a: str
    station_b: str
    station_ids: str
    source_attribute: str
    raw_feature_name: str
    display_name: str


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 150) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_component_text(text: str) -> ComponentSpec:
    clean = " ".join(str(text).strip().split())
    pair = PAIR_PATTERN.match(clean)
    if pair:
        station_a = pair.group("a")
        station_b = pair.group("b")
        attribute = pair.group("attribute")
        return ComponentSpec(
            source_family="station_pair_spread",
            station_id="",
            station_a=station_a,
            station_b=station_b,
            station_ids=f"{station_a},{station_b}",
            source_attribute=attribute,
            raw_feature_name=attribute,
            display_name=f"{station_a} minus {station_b} {attribute}",
        )
    single = SINGLE_PATTERN.match(clean)
    if not single:
        raise ValueError(f"Could not parse station component: {text!r}")
    station_id = single.group("station")
    feature = single.group("feature")
    if "__" in feature:
        source_attribute = feature.split("__", 1)[0]
        source_family = "station_trajectory"
    else:
        source_attribute = feature
        source_family = "station_attribute"
    return ComponentSpec(
        source_family=source_family,
        station_id=station_id,
        station_a="",
        station_b="",
        station_ids=station_id,
        source_attribute=source_attribute,
        raw_feature_name=feature,
        display_name=f"{station_id} {feature}",
    )


def component_feature_base(spec: ComponentSpec) -> str:
    if spec.source_family == "station_trajectory":
        return f"traj_{station_token(spec.station_id)}_{spec.raw_feature_name}"
    if spec.source_family == "station_attribute":
        return f"stat_{station_token(spec.station_id)}_{spec.raw_feature_name}"
    return f"pair_{spec.raw_feature_name}_{station_token(spec.station_a)}_minus_{station_token(spec.station_b)}"


def normalize_component_text(part: str, candidate: dict[str, object]) -> str:
    clean = " ".join(str(part).strip().split())
    if PAIR_PATTERN.match(clean) or SINGLE_PATTERN.match(clean):
        return clean
    candidate_name = " ".join(str(candidate.get("candidate_name", "")).strip().split())
    station_ids = re.findall(STATION_ID_PATTERN, candidate_name)
    if len(station_ids) == 1 and clean and candidate_name.endswith(clean):
        return f"{station_ids[0]} {clean}"
    return clean


def load_allowed_candidates(path: Path = INPUT_AUDIT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing 0053 candidate audit file: {path}")
    frame = pd.read_csv(path)
    allowed = frame[frame["allowed_for_future_walkforward"].map(truthy)].copy()
    if allowed.empty:
        raise ValueError("No 0053 candidates are allowed for future walk-forward use")
    return allowed.sort_values("audit_priority_score", ascending=False, na_position="last").reset_index(drop=True)


def build_component_catalog(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    used_ids: set[str] = set()
    spec_to_id: dict[tuple[str, str, str, str, str], str] = {}
    component_rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    for candidate in candidates.to_dict("records"):
        raw_parts = [part.strip() for part in str(candidate["deployable_feature_text"]).split(";") if part.strip()]
        if not raw_parts:
            raw_parts = [str(candidate["candidate_name"])]
        component_ids: list[str] = []
        for part_index, part in enumerate(raw_parts, start=1):
            spec = parse_component_text(normalize_component_text(part, candidate))
            key = (
                spec.source_family,
                spec.station_id,
                spec.station_a,
                spec.station_b,
                spec.raw_feature_name,
            )
            if key not in spec_to_id:
                feature_id = unique_feature_id(component_feature_base(spec), used_ids)
                spec_to_id[key] = feature_id
                component_rows.append(
                    {
                        "selected_rank": len(component_rows) + 1,
                        "feature_id": feature_id,
                        "source_family": spec.source_family,
                        "station_id": spec.station_id,
                        "station_a": spec.station_a,
                        "station_b": spec.station_b,
                        "station_ids": spec.station_ids,
                        "source_attribute": spec.source_attribute,
                        "transform": "parsed_from_0053_allowed_candidate",
                        "raw_feature_name": spec.raw_feature_name,
                        "display_name": spec.display_name,
                    }
                )
            component_id = spec_to_id[key]
            component_ids.append(component_id)
            mapping_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "candidate_type": candidate["candidate_type"],
                    "candidate_name": candidate["candidate_name"],
                    "component_order": part_index,
                    "component_feature_id": component_id,
                    "component_display_name": spec.display_name,
                    "candidate_primary_score": candidate.get("primary_score", math.nan),
                    "candidate_official_error_score": candidate.get("official_error_score", math.nan),
                    "candidate_audit_priority_score": candidate.get("audit_priority_score", math.nan),
                }
            )
        mapping_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_type": candidate["candidate_type"],
                "candidate_name": candidate["candidate_name"],
                "component_order": 0,
                "component_feature_id": ",".join(component_ids),
                "component_display_name": " + ".join(component_ids),
                "candidate_primary_score": candidate.get("primary_score", math.nan),
                "candidate_official_error_score": candidate.get("official_error_score", math.nan),
                "candidate_audit_priority_score": candidate.get("audit_priority_score", math.nan),
            }
        )
    return pd.DataFrame(component_rows), pd.DataFrame(mapping_rows)


def deployable_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in {"target_date", "target_tmax_c", "past_doy_count", "past_doy_mean_tmax_c", "target_anomaly_vs_past_doy_c"}
    ]


def assert_no_forbidden_feature_columns(columns: list[str]) -> None:
    offenders = [
        column
        for column in columns
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if offenders:
        raise ValueError(f"Forbidden leakage-like feature columns: {offenders[:10]}")


def build_deployable_matrix(feature_frame: pd.DataFrame) -> pd.DataFrame:
    feature_columns = deployable_feature_columns(feature_frame)
    assert_no_forbidden_feature_columns(feature_columns)
    out = feature_frame[["target_date", *feature_columns]].copy()
    out["source_local_date_rule"] = "target_date_minus_1"
    out["source_cutoff_hkt"] = (
        (pd.to_datetime(out["target_date"], errors="coerce").dt.normalize() - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
        + " 15:00:00+08:00"
    )
    return out


def feature_coverage_catalog(feature_frame: pd.DataFrame, component_catalog: pd.DataFrame) -> pd.DataFrame:
    feature_columns = deployable_feature_columns(feature_frame)
    train_mask = feature_frame["target_date"].le(TRAIN_END)
    eval_mask = feature_frame["target_date"].ge(EVAL_START) & feature_frame["target_date"].le(EVAL_END)
    meta = component_catalog.set_index("feature_id").to_dict("index")
    rows: list[dict[str, object]] = []
    for feature in feature_columns:
        values = pd.to_numeric(feature_frame[feature], errors="coerce")
        n_train, corr_train = safe_corr(
            values[train_mask],
            feature_frame.loc[train_mask, "target_anomaly_vs_past_doy_c"],
            min_rows=365,
        )
        n_eval, corr_eval = safe_corr(
            values[eval_mask],
            feature_frame.loc[eval_mask, "target_anomaly_vs_past_doy_c"],
            min_rows=365,
        )
        present = feature_frame.loc[values.notna(), ["target_date"]]
        row_meta = meta.get(feature, {})
        rows.append(
            {
                "feature_id": feature,
                "source_family": row_meta.get("source_family", ""),
                "station_ids": row_meta.get("station_ids", ""),
                "raw_feature_name": row_meta.get("raw_feature_name", ""),
                "non_null_rows": int(values.notna().sum()),
                "coverage_fraction": float(values.notna().mean()),
                "first_target_date": str(present["target_date"].min().date()) if not present.empty else "",
                "last_target_date": str(present["target_date"].max().date()) if not present.empty else "",
                "n_train_pre2000_corr": n_train,
                "corr_train_pre2000_target_anomaly": corr_train,
                "abs_corr_train_pre2000_target_anomaly": abs(corr_train) if math.isfinite(corr_train) else math.nan,
                "n_eval_2000_2023_corr": n_eval,
                "corr_eval_2000_2023_target_anomaly": corr_eval,
                "abs_corr_eval_2000_2023_target_anomaly": abs(corr_eval) if math.isfinite(corr_eval) else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["abs_corr_eval_2000_2023_target_anomaly", "coverage_fraction"],
        ascending=[False, False],
        na_position="last",
    )


def leakage_audit(
    *,
    candidates: pd.DataFrame,
    component_catalog: pd.DataFrame,
    deployable_matrix: pd.DataFrame,
    feature_coverage: pd.DataFrame,
) -> pd.DataFrame:
    feature_columns = [column for column in deployable_matrix.columns if column not in {"target_date", "source_local_date_rule", "source_cutoff_hkt"}]
    checks = [
        {
            "check_id": "no_confirmation_rows",
            "passed": bool(pd.to_datetime(deployable_matrix["target_date"], errors="coerce").max() < CONFIRMATION_START),
            "evidence": f"last target_date {pd.to_datetime(deployable_matrix['target_date'], errors='coerce').max().date()}",
        },
        {
            "check_id": "all_candidates_allowed_by_0053",
            "passed": bool(candidates["allowed_for_future_walkforward"].map(truthy).all()),
            "evidence": f"{len(candidates)} input candidates all marked allowed",
        },
        {
            "check_id": "no_target_or_residual_feature_columns",
            "passed": not any(any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS) for column in feature_columns),
            "evidence": f"{len(feature_columns)} deployable feature columns scanned",
        },
        {
            "check_id": "feature_matrix_excludes_labels",
            "passed": not {"target_tmax_c", "target_anomaly_vs_past_doy_c", "past_doy_mean_tmax_c"}.intersection(deployable_matrix.columns),
            "evidence": "deployable parquet contains target_date, feature columns, and source timing rule columns only",
        },
        {
            "check_id": "source_cutoff_rule_attached",
            "passed": bool(deployable_matrix["source_local_date_rule"].eq("target_date_minus_1").all()),
            "evidence": "all rows carry source_local_date_rule=target_date_minus_1",
        },
        {
            "check_id": "component_catalog_station_only",
            "passed": bool(component_catalog["source_family"].isin(["station_attribute", "station_trajectory", "station_pair_spread"]).all()),
            "evidence": ",".join(sorted(component_catalog["source_family"].unique())),
        },
        {
            "check_id": "features_have_eval_coverage",
            "passed": bool((feature_coverage["n_eval_2000_2023_corr"] >= 365).all()),
            "evidence": f"minimum eval rows {int(feature_coverage['n_eval_2000_2023_corr'].min()) if not feature_coverage.empty else 0}",
        },
    ]
    return pd.DataFrame(checks)


def build_readme(
    *,
    summary: dict[str, Any],
    leakage: pd.DataFrame,
    coverage: pd.DataFrame,
    component_catalog: pd.DataFrame,
    candidate_mapping: pd.DataFrame,
) -> str:
    coverage_display = coverage[
        [
            "feature_id",
            "source_family",
            "station_ids",
            "coverage_fraction",
            "first_target_date",
            "last_target_date",
            "abs_corr_eval_2000_2023_target_anomaly",
        ]
    ].head(80) if not coverage.empty else pd.DataFrame()
    component_display = component_catalog.head(80) if not component_catalog.empty else pd.DataFrame()
    mapping_display = candidate_mapping[candidate_mapping["component_order"].ne(0)].head(80) if not candidate_mapping.empty else pd.DataFrame()
    return f"""# Station-Only Walk-Forward Matrix Audit

Generated: `{summary['generated_at_utc']}`

## Purpose

`0053` proved that the immediate leakage-safe candidate pool is station-derived. This folder converts those allowed candidates into a station-only deployable feature matrix and runs leakage checks on the matrix. This is not a model run and not a final MAE claim.

## Matrix Scope

| Item | Value |
|---|---:|
| Input 0053 allowed candidates | {summary['allowed_candidate_rows']} |
| Unique station component features | {summary['component_feature_rows']} |
| Matrix rows | {summary['matrix_rows']} |
| Matrix feature columns | {summary['matrix_feature_columns']} |
| First target date | {summary['first_target_date']} |
| Last target date | {summary['last_target_date']} |
| Uses 2024+ rows | {summary['uses_2024_plus_rows']} |

## Leakage Contract

- The deployable matrix excludes target labels, target anomalies, official forecast errors, residuals, MAE, and RMSE.
- Every feature is derived from station data only.
- Every row uses target date `T` with source local date `T-1` and latest station observation before `15:00 HKT`.
- Trajectory features use station-local past-only lag/rolling formulas.
- Correlation columns in this report are audit diagnostics only and are not included in the deployable matrix file.

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Strongest Feature Coverage/Correlation Audit

{markdown_table(coverage_display, max_rows=80)}

## Component Feature Catalog

{markdown_table(component_display, max_rows=80)}

## Candidate-To-Component Mapping

{markdown_table(mapping_display, max_rows=80)}

## Interpretation

The station-only pool is now concrete and testable: future walk-forward experiments can consume the deployable feature matrix, but any fitted transformation, scaling, thresholding, routing, or model training still has to happen inside each OOF fold. This preserves the reliability requirement while keeping the high-information station signals discovered in `0047`, `0050`, and `0051`.

## Files

- `artifacts/features.parquet`
- `artifacts/features_sample.csv`
- `artifacts/components.csv`
- `artifacts/candidate_components.csv`
- `artifacts/feature_coverage.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_station_only_walkforward_matrix_audit.py`:

- `{FOLDER_NAME}`: station-only deployable feature matrix and leakage audit from `0053` allowed candidates.

| Metric | Value |
|---|---:|
| Allowed candidates consumed | {summary['allowed_candidate_rows']} |
| Component features | {summary['component_feature_rows']} |
| Matrix rows | {summary['matrix_rows']} |
| Matrix feature columns | {summary['matrix_feature_columns']} |

Leakage contract: deployable matrix has no labels/residuals and carries only station-derived pre-cutoff features.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Station-Only Walk-Forward Matrix Audit",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_station_only_walkforward_matrix_audit.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Station-only matrix | `{summary['matrix_rows']}` rows x `{summary['matrix_feature_columns']}` deployable feature columns | Built |
| Allowed candidates consumed | `{summary['allowed_candidate_rows']}` from `0053` | Complete |
| Unique component features | `{summary['component_feature_rows']}` | Audited |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |
| Confirmation rows | `{summary['uses_2024_plus_rows']}` | Locked out |

Interpretation: `0054` produces the first concrete station-only deployable matrix from the allowed candidate pool. It still does not train a model; future OOF work must fit transformations inside each fold.
"""
    update_markdown_section(
        path,
        heading="Station-Only Walk-Forward Matrix Audit",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    blocker = (
        f"38. Station-only walk-forward matrix built with `{summary['matrix_feature_columns']}` deployable feature columns "
        f"across `{summary['matrix_rows']}` rows, consuming `{summary['allowed_candidate_rows']}` allowed candidates. "
        "No labels/residuals are present in the deployable matrix."
    )
    text = path.read_text(encoding="utf-8")
    if blocker not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{blocker}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Use the `0054` station-only matrix to run fold-local leakage tests for candidate transformations and only then start a bounded walk-forward station-only benchmark. Keep 2024+ locked and do not include upper-air/HKO daily candidates until their timestamp proofs are attached.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    folder = output_root / FOLDER_NAME
    artifacts = folder / "artifacts"
    candidates = load_allowed_candidates()
    component_catalog, candidate_mapping = build_component_catalog(candidates)
    target = load_target()
    station_frame = load_station_day_features(target)
    feature_frame, enriched_catalog = build_feature_frame(station_frame, component_catalog)
    require_no_confirmation_dates(feature_frame["target_date"], context="0054 feature frame")
    deployable_matrix = build_deployable_matrix(feature_frame)
    feature_coverage = feature_coverage_catalog(feature_frame, enriched_catalog)
    leakage = leakage_audit(
        candidates=candidates,
        component_catalog=enriched_catalog,
        deployable_matrix=deployable_matrix,
        feature_coverage=feature_coverage,
    )
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0054 leakage audit failed: {failed}")

    deployable_matrix_path = artifacts / "features.parquet"
    deployable_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    deployable_matrix.to_parquet(deployable_matrix_path, index=False)
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "allowed_candidate_rows": int(len(candidates)),
        "component_feature_rows": int(len(enriched_catalog)),
        "candidate_component_mapping_rows": int(len(candidate_mapping)),
        "matrix_rows": int(len(deployable_matrix)),
        "matrix_feature_columns": int(len(deployable_feature_columns(feature_frame))),
        "first_target_date": str(pd.to_datetime(deployable_matrix["target_date"]).min().date()),
        "last_target_date": str(pd.to_datetime(deployable_matrix["target_date"]).max().date()),
        "leakage_check_rows": int(len(leakage)),
        "leakage_checks_passed": int(leakage["passed"].astype(bool).sum()),
        "training_threshold_end": str(TRAIN_END.date()),
        "evaluation_start": str(EVAL_START.date()),
        "evaluation_end": str(EVAL_END.date()),
        "confirmation_start": str(CONFIRMATION_START.date()),
        "uses_2024_plus_rows": False,
        "deployable_matrix_path": str(deployable_matrix_path),
    }

    write_csv(artifacts / "features_sample.csv", deployable_matrix.head(500))
    write_csv(artifacts / "components.csv", enriched_catalog)
    write_csv(artifacts / "candidate_components.csv", candidate_mapping)
    write_csv(artifacts / "feature_coverage.csv", feature_coverage)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "station_only_walkforward_matrix_audit_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            leakage=leakage,
            coverage=feature_coverage,
            component_catalog=enriched_catalog,
            candidate_mapping=candidate_mapping,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Build station-only walk-forward matrix and leakage audit.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
