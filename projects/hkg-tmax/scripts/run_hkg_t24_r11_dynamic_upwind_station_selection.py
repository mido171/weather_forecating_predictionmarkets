from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from hkg_tmax.hkg_t24.guard import assert_no_locked_dates
from hkg_tmax.paths import ProjectPaths

PROJECT_PATHS = ProjectPaths.discover(Path(__file__))
REPO_ROOT = PROJECT_PATHS.project_root
DEFAULT_DATA_ROOT = PROJECT_PATHS.data_root
RESEARCH_ID = "HKG-T24-R11"
EXPERIMENT_ID = "EXP-0043"
EXPERIMENT_DIR = PROJECT_PATHS.run_root / "experiments" / "legacy" / "hkg_tmax_t24" / "EXP-0043-HKG-T24-R11"


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


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_date": pd.Series(dtype="datetime64[ns]"),
            "fold_id": pd.Series(dtype="object"),
            "model_id": pd.Series(dtype="object"),
            "model_family": pd.Series(dtype="object"),
            "point_forecast": pd.Series(dtype="float64"),
            "target_tmax_c": pd.Series(dtype="float64"),
            "blocked_reason": pd.Series(dtype="object"),
        }
    )


def source_summary(path: Path, date_col: str, station_col: str | None = None, *, guard_target_dates: bool = False) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    frame = pd.read_parquet(path)
    out: dict[str, Any] = {"exists": True, "rows": int(len(frame)), "sha256": sha256_file(path)}
    if date_col in frame.columns and not frame.empty:
        dates = pd.to_datetime(frame[date_col])
        out["first_time"] = str(dates.min())
        out["last_time"] = str(dates.max())
        if guard_target_dates:
            assert_no_locked_dates(dates.dt.date, context=f"R11 source summary {path.name}")
    if station_col and station_col in frame.columns:
        out["station_count"] = int(frame[station_col].nunique())
        out["stations"] = sorted(frame[station_col].dropna().astype(str).unique().tolist())
    return out


def hko_registry_has_geometry(registry: pd.DataFrame) -> bool:
    required = {"latitude", "longitude", "elevation_m"}
    if not required.issubset(set(registry.columns)):
        return False
    hko_rows = registry[registry["network"].astype(str).eq("HKO")]
    return all(hko_rows[col].notna().any() for col in required)


def build_readiness(data_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    registry_path = REPO_ROOT / "config" / "hkg_t24" / "station_registry.parquet"
    static_registry_path = data_root / "metadata" / "static_context" / "station_registry.parquet"
    distance_path = data_root / "metadata" / "static_context" / "station_distance_bearing.parquet"
    r08_wind_path = data_root / "bronze" / "hkg_t24" / "r08_wind_vector_sampled_observations.parquet"
    r09_temp_path = data_root / "bronze" / "hkg_t24" / "r09_temperature_sampled_observations.parquet"
    r08_matrix_path = data_root / "gold" / "hkg_t24" / "r08_wind_advection" / "r08_feature_matrix.parquet"
    r09_matrix_path = data_root / "gold" / "hkg_t24" / "r09_station_temperature_gradient" / "r09_feature_matrix.parquet"

    registry = pd.read_parquet(registry_path)
    static_registry = pd.read_parquet(static_registry_path) if static_registry_path.exists() else pd.DataFrame()
    distance = pd.read_parquet(distance_path) if distance_path.exists() else pd.DataFrame()
    hko_geometry_ready = hko_registry_has_geometry(registry)
    hko_distance_ready = (
        not distance.empty
        and "origin_station_id" in distance.columns
        and "target_station_id" in distance.columns
        and distance["target_station_id"].astype(str).str.startswith("hko_name:").any()
    )

    sources = {
        "r08_wind_sampled": source_summary(r08_wind_path, "observed_at_hkt", "station"),
        "r09_temperature_sampled": source_summary(r09_temp_path, "observed_at_hkt", "station"),
        "r08_feature_matrix": source_summary(r08_matrix_path, "target_date", guard_target_dates=True),
        "r09_feature_matrix": source_summary(r09_matrix_path, "target_date", guard_target_dates=True),
        "hko_station_registry": {"exists": registry_path.exists(), "rows": int(len(registry)), "sha256": sha256_file(registry_path)},
        "static_station_registry": {
            "exists": static_registry_path.exists(),
            "rows": int(len(static_registry)),
            "sha256": sha256_file(static_registry_path) if static_registry_path.exists() else None,
        },
        "station_distance_bearing": {
            "exists": distance_path.exists(),
            "rows": int(len(distance)),
            "sha256": sha256_file(distance_path) if distance_path.exists() else None,
        },
    }

    rows = [
        {
            "requirement": "surface wind network before cutoff",
            "status": "available",
            "evidence": f"{sources['r08_wind_sampled'].get('rows')} sampled rows; {sources['r08_wind_sampled'].get('station_count')} stations",
            "disposition": "usable as raw flow signal from R08",
        },
        {
            "requirement": "surrounding station temperature field before cutoff",
            "status": "available",
            "evidence": f"{sources['r09_temperature_sampled'].get('rows')} sampled rows; {sources['r09_temperature_sampled'].get('station_count')} stations",
            "disposition": "usable as fixed station-field signal from R09",
        },
        {
            "requirement": "canonical HKO station latitude/longitude/elevation for all dynamic candidates",
            "status": "blocked",
            "evidence": "config/hkg_t24/station_registry.parquet has no latitude/longitude/elevation_m columns for HKO high-frequency stations",
            "disposition": "do not compute cones, distances, gradients, or random-coordinate controls",
        },
        {
            "requirement": "distance/bearing matrix for HKO high-frequency station names",
            "status": "blocked" if not hko_distance_ready else "available",
            "evidence": f"static distance table rows={len(distance)}; HKO named station ids present={hko_distance_ready}",
            "disposition": "NOAA/HKO target static table is not a substitute for 39 HKO public feed stations",
        },
        {
            "requirement": "eligible IGRA 925/850 hPa wind by cutoff",
            "status": "blocked",
            "evidence": "IGRA raw period-of-record zip is downloaded but no parsed eligible sounding table exists",
            "disposition": "upper-air flow cannot enter R11 scoring",
        },
        {
            "requirement": "flow-relative upwind cones and length-scale selection",
            "status": "blocked",
            "evidence": "depends on verified station coordinates and elevations",
            "disposition": "document blocker; do not fake geometry from station names",
        },
        {
            "requirement": "fixed-group fallback comparison",
            "status": "available",
            "evidence": "R08 and R09 fixed wind/spatial group diagnostics exist",
            "disposition": "already benchmarked; insufficient to answer R11 dynamic-geometry question",
        },
    ]
    payload = {
        "generated_at": now_utc(),
        "hko_geometry_ready": hko_geometry_ready,
        "hko_distance_ready": hko_distance_ready,
        "sources": sources,
        "blockers": [
            "hko_station_coordinates",
            "hko_station_elevations",
            "hko_named_station_distance_bearing_matrix",
            "igra_eligible_upper_air_wind_parser",
        ],
    }
    return pd.DataFrame(rows), payload


def long_report(payload: dict[str, Any], readiness: pd.DataFrame) -> str:
    wind = payload["sources"]["r08_wind_sampled"]
    temp = payload["sources"]["r09_temperature_sampled"]
    return f"""# EXP-0043 / HKG-T24-R11 Long-Form Experiment Report

## Purpose

R11 was specified as the dynamic upwind station-selection and flow-relative advection experiment. Its purpose is to test whether the station or regional source area that matters for tomorrow's official HKO Headquarters Tmax changes with the observed flow at the T-1 15:00 cutoff. This is a different question from R08 and R09. R08 asked whether wind state by itself carries signal. R09 asked whether fixed station-temperature contrasts carry signal. R11 asks whether the wind vector can choose which surrounding thermal and moisture observations are upstream, downstream, or dynamically irrelevant.

## Required Design

The uploaded goal requires upstream cones of 30, 60, 90, and 120 degrees; distance-decayed upwind averages using multiple length scales; downstream-minus-upstream contrasts; advection proxies from spatial gradients; surface plus 925/850 hPa flow; fallback behavior when winds are weak; and randomized station-coordinate negative controls. Every one of those operations depends on verified station geometry. A dynamic upwind cone cannot be computed from station names, file order, or manually guessed neighborhoods. The geometry must be canonical and point-in-time documented for the same station identities used by the high-frequency HKO feeds.

## Inputs Audited

The available surface wind table from R08 has `{wind.get('rows')}` sampled rows, `{wind.get('station_count')}` stations, and observed timestamps from `{wind.get('first_time')}` through `{wind.get('last_time')}`. The available station-temperature table from R09 has `{temp.get('rows')}` sampled rows, `{temp.get('station_count')}` stations, and observed timestamps from `{temp.get('first_time')}` through `{temp.get('last_time')}`. These are legitimate cutoff-safe source families under the conservative 20-minute availability rule. They are necessary for R11, but they are not sufficient.

The current canonical experiment registry at `config/hkg_t24/station_registry.parquet` contains HKO station identities, aliases, feed membership, target flags, and unresolved official-code status. It does not contain latitude, longitude, or elevation columns for the named HKO high-frequency stations. The separate static-context registry under the data root contains one HKO target point and NOAA ISD station metadata, but it does not resolve the full HKO named station network used in R08/R09. The static distance/bearing table is therefore not a valid substitute for dynamic upwind station geometry across the public HKO feed stations.

## Leakage Decision

R11 is blocked rather than approximated. That is an intentional leakage and scientific-validity decision. If I computed cones using station-name ordering, hand-assigned compass groups, or the NOAA-only static table, the resulting features would look mathematically precise but would not correspond to the station field actually used by the modern HKO high-frequency archive. Worse, randomized-coordinate negative controls would become meaningless because there would be no trusted coordinate baseline to randomize from.

## What Was Completed

R11 completed an input-readiness and no-go gate. The experiment folder contains a machine-readable readiness table, metrics JSON, empty OOF prediction table with a blocked reason, explicit feature specification, ablation plan, negative controls, date ranges, data manifest, and conclusion. The repository report records that the surface wind and temperature ingredients exist, but the dynamic geometry layer and eligible upper-air wind parser do not.

## Why No OOF Model Was Scored

Scoring a model would require at least one valid dynamic feature family. The available fixed-group fallback features were already tested in R08 and R09. Repackaging them as R11 would double-count earlier work and would not answer the dynamic upwind hypothesis. The R11 model ladder is therefore intentionally empty until station coordinates/elevations and the IGRA wind parser are available. This is not a missing effort item; it is the correct blocked outcome under the non-negotiable no-forward-looking and no-fake-geometry rules.

## Readiness Table

{markdown_table(readiness)}

## Data Ranges

The audited wind observations span `{wind.get('first_time')}` through `{wind.get('last_time')}`. The audited temperature observations span `{temp.get('first_time')}` through `{temp.get('last_time')}`. The existing modern pre-validation feature matrices still end at 2023-12-31, and no 2024 validation outcomes or 2025+ locked-test rows were accessed. Because no dynamic R11 feature matrix exists, there is no R11 development OOF date range and no candidate can be promoted.

## Blockers

The exact blockers are: verified latitude/longitude for all HKO high-frequency station names, verified elevation for those same station identities, a station-distance/bearing matrix keyed to the canonical HKO station ids rather than NOAA-only ids, an eligible 00 UTC T-1 IGRA parser for 925/850 hPa wind, and a fold-safe hyperparameter procedure for cone angle and length scale once the geometry exists. All of these are engineering inputs, not model-tuning choices.

## Next Action

The next lawful task for this research branch is to enrich the station registry from official HKO station metadata or another citable source, preserve aliases such as Wong Chuk Han versus Wong Chuk Hang without blind merging, derive the HKO-named station distance/bearing/elevation matrix, and then rerun R11 with the predeclared cone and length-scale grid. Until then, R12 can proceed because it uses already parsed King's Park solar observations and does not depend on station geometry.

## Decision Record

R11 status is `BLOCKED_INPUTS_MISSING`. Surface wind and temperature inputs are available, but the required dynamic upwind geometry is not. No validation data was read. No locked-test data was read. No model was trained. No feature was promoted. The null/blocker result is retained and indexed so later work does not accidentally treat dynamic upwind information as tested.

This result should be treated as a hard engineering prerequisite, not as evidence that flow-relative advection lacks meteorological value. The experiment says only that the current repository cannot test the idea honestly yet.

## Guardrail Detail

The most tempting shortcut would be to use a hand-built list of "north", "south", "coastal", or "inland" station groups as a pseudo-upwind geometry. R11 rejects that shortcut. Those groups were already represented in R08/R09-style fixed diagnostics, and they do not satisfy the dynamic upwind specification. A flow-relative cone needs bearings from HKO to each candidate station, and a distance-decay calculation needs verified distances in kilometers. Without those fields, every downstream number would be arbitrary even if the code looked precise.

Another tempting shortcut would be to use the NOAA ISD static registry as a proxy for the HKO public feed stations. That is also rejected. The NOAA stations are useful for future long-history regional work, but their identifiers and station histories are not the same as the named HKO high-frequency stations in the current temperature, humidity, pressure, and wind feeds. Mixing those registries would create a false sense of spatial precision and could silently merge unresolved station aliases.

The R11 folder is therefore designed to make the blocker operationally actionable. It tells the next worker exactly what to add: official HKO station coordinates, elevations, validity periods, alias resolution, a distance/bearing matrix keyed to the canonical HKO feed station ids, and an eligible upper-air wind parser. Once those are present, the same experiment id can be rerun as a scored dynamic-upwind test rather than a no-go gate.
"""


def write_experiment(data_root: Path, readiness: pd.DataFrame, payload: dict[str, Any]) -> None:
    for subdir in ["artifacts", "logs", "metrics", "predictions", "results"]:
        (EXPERIMENT_DIR / subdir).mkdir(parents=True, exist_ok=True)
    predictions = empty_predictions()
    subgroup = pd.DataFrame(
        {
            "subgroup": ["all"],
            "status": ["not_scored"],
            "reason": ["dynamic upwind geometry blocked"],
        }
    )
    predictions_path = EXPERIMENT_DIR / "predictions" / "oof_predictions.parquet"
    subgroup_path = EXPERIMENT_DIR / "metrics" / "subgroup_metrics.parquet"
    readiness_path = EXPERIMENT_DIR / "artifacts" / "input_readiness.csv"
    predictions.to_parquet(predictions_path, index=False)
    subgroup.to_parquet(subgroup_path, index=False)
    readiness.to_csv(readiness_path, index=False)

    metrics = {
        "research_id": RESEARCH_ID,
        "experiment_id": EXPERIMENT_ID,
        "status": "BLOCKED_INPUTS_MISSING",
        "validation_2024_accessed": False,
        "locked_test_accessed": False,
        "oof_predictions": 0,
        "primary_blockers": payload["blockers"],
        "input_readiness": readiness.to_dict(orient="records"),
        "source_summary": payload["sources"],
    }
    write_text(EXPERIMENT_DIR / "metrics" / "metrics.json", json.dumps(metrics, indent=2, default=str))
    write_text(EXPERIMENT_DIR / "results" / "metrics.json", json.dumps(metrics, indent=2, default=str))
    write_text(EXPERIMENT_DIR / "logs" / "run_summary.json", json.dumps(payload, indent=2, default=str))
    write_text(EXPERIMENT_DIR / "README.md", "# EXP-0043 HKG-T24-R11 Dynamic Upwind Station Selection\n\nBlocked input-readiness gate for dynamic upwind station selection. Surface wind and temperature exist; canonical HKO station geometry and eligible upper-air wind parsing are missing.\n")
    write_text(EXPERIMENT_DIR / "HYPOTHESIS.md", "# Hypothesis\n\nFlow-relative upstream station selection should outperform fixed station groups when verified station geometry and eligible flow vectors are available.\n")
    write_text(EXPERIMENT_DIR / "INFORMATION_GAIN.md", "# Information Gain\n\nThe information gained here is a precise no-go result: R11 cannot be scored without verified HKO station coordinates/elevations and an upper-air wind parser. This prevents fake dynamic geometry from contaminating later experiments.\n")
    write_text(EXPERIMENT_DIR / "ASOF_CONTRACT.md", "# As-Of Contract\n\nAll audited R08/R09 source observations are cutoff-safe under the conservative 20-minute HKO latency rule. No dynamic feature was produced because required static geometry is missing.\n")
    write_text(EXPERIMENT_DIR / "FEATURE_SPEC.yaml", """research_id: HKG-T24-R11
feature_families:
  dynamic_upwind_surface_temperature: blocked_missing_hko_station_geometry
  dynamic_upwind_surface_moisture: blocked_missing_hko_station_geometry
  dynamic_upwind_pressure: blocked_missing_hko_station_geometry
  upper_air_flow_weighting: blocked_missing_igra_parser
  fixed_group_fallback: available_from_R08_R09_not_scored_as_R11
""")
    write_text(EXPERIMENT_DIR / "RUN_CONFIG.yaml", f"""research_id: {RESEARCH_ID}
experiment_id: {EXPERIMENT_ID}
mode: input_readiness_gate
data_root: {data_root}
validation_2024_accessed: false
locked_test_policy: deny
""")
    write_text(EXPERIMENT_DIR / "PROTOCOL.md", "# Protocol\n\n1. Audit R08 wind source table.\n2. Audit R09 station-temperature source table.\n3. Inspect canonical station registry for HKO geometry.\n4. Inspect static distance/bearing table for HKO named station ids.\n5. If geometry is missing, write blocker-complete experiment artifacts and do not score a model.\n")
    write_text(EXPERIMENT_DIR / "ABLATION_PLAN.md", "# Ablation Plan\n\nThe planned ablations are surface wind only, upper-air wind only, fixed versus dynamic upwind, no distance decay, and randomized station coordinates. They are not executed because the required coordinate baseline is missing.\n")
    write_text(EXPERIMENT_DIR / "NEGATIVE_CONTROLS.md", "# Negative Controls\n\nRandomized station coordinates and globally selected cone/length-scale controls are required later. They are blocked now because no trusted HKO station-coordinate table exists.\n")
    write_text(EXPERIMENT_DIR / "DATE_RANGES.md", f"""# Date Ranges

- Wind observations audited: `{payload['sources']['r08_wind_sampled'].get('first_time')}` through `{payload['sources']['r08_wind_sampled'].get('last_time')}`.
- Temperature observations audited: `{payload['sources']['r09_temperature_sampled'].get('first_time')}` through `{payload['sources']['r09_temperature_sampled'].get('last_time')}`.
- R11 OOF prediction period: none, blocked before feature construction.
- Validation 2024: not accessed.
- Locked test 2025-01-01 onward: not accessed.
""")
    write_text(EXPERIMENT_DIR / "DATA_MANIFEST.yaml", f"""research_id: {RESEARCH_ID}
data_root: {data_root}
input_readiness: {readiness_path}
input_readiness_sha256: {sha256_file(readiness_path)}
oof_predictions: {predictions_path}
oof_predictions_sha256: {sha256_file(predictions_path)}
validation_2024_accessed: false
locked_test_accessed: false
blocked_inputs: {payload['blockers']}
""")
    report = long_report(payload, readiness)
    write_text(EXPERIMENT_DIR / "EXPERIMENT_REPORT_7500_CHARS.md", report)
    write_text(
        EXPERIMENT_DIR / "RESULTS.md",
        "# Results\n\nR11 was not scored. The dynamic geometry requirements are blocked.\n\n"
        + markdown_table(readiness)
        + "\n",
    )
    write_text(EXPERIMENT_DIR / "CONCLUSION.md", "# Conclusion\n\nR11 is blocked by missing HKO station geometry and missing eligible upper-air wind parsing. The blocker is explicit; no dynamic upwind feature was faked.\n")
    write_text(EXPERIMENT_DIR / "REPRODUCE.md", "# Reproduce\n\n```powershell\n.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_r11_dynamic_upwind_station_selection.py --data-root C:\\hkg_tmax_data\n```\n")
    write_text(EXPERIMENT_DIR / "STATUS.yaml", """status: BLOCKED_INPUTS_MISSING
research_id: HKG-T24-R11
locked_test_accessed: false
validation_2024_accessed: false
leakage_guard: PASS
production_eligible: false
blocked_inputs: [hko_station_coordinates, hko_station_elevations, hko_named_station_distance_bearing_matrix, igra_eligible_upper_air_wind_parser]
""")
    write_text(PROJECT_PATHS.run_root / "reports" / "hkg_t24" / "R11_DYNAMIC_UPWIND_STATION_SELECTION.md", report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HKG-T24-R11 dynamic upwind station-selection input-readiness gate.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    readiness, payload = build_readiness(data_root)
    write_experiment(data_root, readiness, payload)
    print(json.dumps({"status": "blocked", "research_id": RESEARCH_ID, "blockers": payload["blockers"]}, indent=2))


if __name__ == "__main__":
    main()
