#!/usr/bin/env python3
"""Build a column- and station-level census of HKG T+24 datasets.

The script profiles exact schemas for all supported tables. It can sample large
tables or fully scan files below a configurable size. It never reads target
values from 2024+ to compute target-dependent summaries; generic schema and
coverage inventory may still enumerate files containing later rows.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from _common import utc_now, write_csv, write_json

try:
    import pandas as pd
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "build_data_census.py requires pandas and numpy in the project environment"
    ) from exc

SUPPORTED = (".csv", ".csv.gz", ".parquet", ".json", ".jsonl", ".ndjson", ".feather")
STATION_HINTS = (
    "station_id","station","stn","usaf_wban","usaf","wban","site_id","site",
)
TIME_HINTS = (
    "date","time","timestamp","valid","issue","publish","available","retrieval",
    "cycle","target_date","observation",
)
QUALITY_HINTS = ("quality","qc","flag","status")
SENTINELS = {-9999, -999.9, 9999, 999.9, 99999, -99999}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True, type=Path)
    p.add_argument("--datasets-root", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--mode", choices=["metadata","sample","full"], default="sample")
    p.add_argument("--sample-rows", type=int, default=100000)
    p.add_argument("--max-full-file-mb", type=float, default=256.0)
    p.add_argument("--include-derived-experiment-tables", action="store_true")
    return p.parse_args()


def compound_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".csv.gz"):
        return ".csv.gz"
    return path.suffix.lower()


def identify_column(columns: list[str], hints: tuple[str, ...]) -> list[str]:
    ranked: list[str] = []
    for column in columns:
        low = column.lower()
        if low in hints or any(hint in low for hint in hints):
            ranked.append(column)
    return ranked


def read_table(path: Path, mode: str, sample_rows: int, max_full_bytes: int) -> tuple["pd.DataFrame", int | None, str, dict]:
    suffix = compound_suffix(path)
    size = path.stat().st_size
    actual_mode = mode
    if mode == "full" and size > max_full_bytes:
        actual_mode = "sample"
    metadata: dict[str, Any] = {}

    if suffix in {".csv", ".csv.gz"}:
        if actual_mode == "metadata":
            frame = pd.read_csv(path, nrows=0)
            row_count = None
        elif actual_mode == "sample":
            frame = pd.read_csv(path, nrows=sample_rows, low_memory=False)
            row_count = None
        else:
            frame = pd.read_csv(path, low_memory=False)
            row_count = len(frame)
        return frame, row_count, actual_mode, metadata

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
            parquet = pq.ParquetFile(path)
            row_count = parquet.metadata.num_rows
            metadata["row_groups"] = parquet.metadata.num_row_groups
            metadata["schema"] = str(parquet.schema)
            if actual_mode == "metadata":
                frame = pd.DataFrame({name: pd.Series(dtype="object") for name in parquet.schema.names})
            elif actual_mode == "sample":
                batches = parquet.iter_batches(batch_size=min(sample_rows, 65536))
                pieces = []
                seen = 0
                for batch in batches:
                    piece = batch.to_pandas()
                    pieces.append(piece)
                    seen += len(piece)
                    if seen >= sample_rows:
                        break
                frame = pd.concat(pieces, ignore_index=True).head(sample_rows) if pieces else pd.DataFrame()
            else:
                frame = pd.read_parquet(path)
            return frame, row_count, actual_mode, metadata
        except ImportError:
            if actual_mode == "metadata":
                raise RuntimeError("pyarrow required to inspect parquet metadata")
            frame = pd.read_parquet(path)
            return frame.head(sample_rows) if actual_mode == "sample" else frame, len(frame), actual_mode, metadata

    if suffix == ".feather":
        frame = pd.read_feather(path)
        row_count = len(frame)
        if actual_mode == "metadata":
            frame = frame.iloc[:0]
        elif actual_mode == "sample":
            frame = frame.head(sample_rows)
        return frame, row_count, actual_mode, metadata

    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            frame = pd.json_normalize(value)
        elif isinstance(value, dict):
            for key in ("data","rows","records","items"):
                if isinstance(value.get(key), list):
                    frame = pd.json_normalize(value[key])
                    metadata["record_container"] = key
                    break
            else:
                frame = pd.json_normalize([value])
        else:
            raise ValueError("JSON root is not object/list")
        row_count = len(frame)
        if actual_mode == "metadata":
            frame = frame.iloc[:0]
        elif actual_mode == "sample":
            frame = frame.head(sample_rows)
        return frame, row_count, actual_mode, metadata

    if suffix in {".jsonl", ".ndjson"}:
        frame = pd.read_json(path, lines=True, nrows=None if actual_mode == "full" else sample_rows)
        row_count = len(frame) if actual_mode == "full" else None
        if actual_mode == "metadata":
            frame = frame.iloc[:0]
        return frame, row_count, actual_mode, metadata

    raise ValueError(f"Unsupported format: {suffix}")


def safe_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (float,)):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


def profile_column(series: "pd.Series") -> dict[str, Any]:
    out: dict[str, Any] = {
        "dtype": str(series.dtype),
        "sample_count": int(len(series)),
        "non_null_count": int(series.notna().sum()),
        "null_rate": float(series.isna().mean()) if len(series) else None,
        "unique_count_sample": int(series.nunique(dropna=True)) if len(series) else 0,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "p01": None,
        "p10": None,
        "p50": None,
        "p90": None,
        "p99": None,
        "top_values": None,
        "sentinel_count": 0,
        "constant_sample": False,
    }
    non_null = series.dropna()
    if non_null.empty:
        return out
    if pd.api.types.is_numeric_dtype(non_null):
        numeric = pd.to_numeric(non_null, errors="coerce").dropna()
        if not numeric.empty:
            out.update({
                "min": safe_value(numeric.min()),
                "max": safe_value(numeric.max()),
                "mean": safe_value(numeric.mean()),
                "std": safe_value(numeric.std(ddof=0)),
                "p01": safe_value(numeric.quantile(0.01)),
                "p10": safe_value(numeric.quantile(0.10)),
                "p50": safe_value(numeric.quantile(0.50)),
                "p90": safe_value(numeric.quantile(0.90)),
                "p99": safe_value(numeric.quantile(0.99)),
                "sentinel_count": int(numeric.isin(SENTINELS).sum()),
                "constant_sample": bool(numeric.nunique() <= 1),
            })
    else:
        values = non_null.astype(str)
        counts = values.value_counts().head(10)
        out["min"] = safe_value(values.min()) if len(values) else None
        out["max"] = safe_value(values.max()) if len(values) else None
        out["top_values"] = json.dumps(
            [{"value": str(idx), "count": int(count)} for idx, count in counts.items()],
            ensure_ascii=False,
        )
        out["constant_sample"] = bool(values.nunique() <= 1)
    return out


def classify_time_column(column: str) -> str:
    low = column.lower()
    if "target_date" in low:
        return "TARGET_LOCAL_DATE"
    if "available" in low:
        return "AVAILABLE_AT"
    if "publish" in low or "issue" in low or "cycle" in low:
        return "ISSUE_OR_PUBLICATION"
    if "retriev" in low or "download" in low:
        return "RETRIEVAL"
    if "valid" in low:
        return "VALID_TIME"
    if "process" in low or "created" in low or "updated" in low:
        return "PROCESSING"
    if "date" in low or "time" in low or "timestamp" in low:
        return "UNKNOWN_TIME"
    return "NOT_TIME"


def infer_source_family(path: Path, datasets_root: Path) -> str:
    rel = path.relative_to(datasets_root)
    return rel.parts[0] if rel.parts else "ROOT"


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    datasets = (args.datasets_root or repo / "data" / "datasets").resolve()
    output = (args.output_dir or repo / ".hkg_t24_research" / "census").resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not datasets.is_dir():
        raise FileNotFoundError(f"Datasets root missing: {datasets}")

    files = [
        p for p in datasets.rglob("*")
        if p.is_file() and any(p.name.lower().endswith(suffix) for suffix in SUPPORTED)
    ]
    if args.include_derived_experiment_tables and (repo / "experiments").is_dir():
        files.extend(
            p for p in (repo / "experiments").rglob("*")
            if p.is_file() and any(p.name.lower().endswith(suffix) for suffix in SUPPORTED)
        )
    max_full_bytes = int(args.max_full_file_mb * 1024 * 1024)

    tables: list[dict] = []
    attributes: list[dict] = []
    timestamp_rows: list[dict] = []
    station_rows: list[dict] = []
    station_coverage: list[dict] = []
    quality: list[dict] = []
    unreadable: list[dict] = []
    station_ids: dict[str, dict] = {}

    for path in sorted(set(files)):
        try:
            frame, row_count, profile_scope, metadata = read_table(
                path, args.mode, args.sample_rows, max_full_bytes
            )
            columns = [str(c) for c in frame.columns]
            rel = path.relative_to(repo).as_posix()
            family = infer_source_family(path, datasets) if datasets in path.parents else "DERIVED_EXPERIMENT"
            station_cols = identify_column(columns, STATION_HINTS)
            time_cols = identify_column(columns, TIME_HINTS)
            quality_cols = identify_column(columns, QUALITY_HINTS)
            duplicate_rows_sample = int(frame.duplicated().sum()) if len(frame) else 0
            tables.append({
                "source_family": family,
                "relative_path": rel,
                "format": compound_suffix(path),
                "size_bytes": path.stat().st_size,
                "row_count_exact": row_count,
                "profile_rows": len(frame),
                "profile_scope": profile_scope,
                "column_count": len(columns),
                "columns_json": json.dumps(columns),
                "station_columns": "|".join(station_cols),
                "time_columns": "|".join(time_cols),
                "quality_columns": "|".join(quality_cols),
                "duplicate_rows_in_profile": duplicate_rows_sample,
                "metadata_json": json.dumps(metadata, default=str),
            })
            if duplicate_rows_sample:
                quality.append({
                    "source_family": family,
                    "relative_path": rel,
                    "severity": "WARNING",
                    "finding": "DUPLICATE_ROWS_IN_PROFILE",
                    "count": duplicate_rows_sample,
                    "details": "",
                })

            for column in columns:
                series = frame[column] if column in frame else pd.Series(dtype="object")
                profile = profile_column(series)
                low = column.lower()
                target_derived = any(x in low for x in ("target","actual_tmax","residual","error","label"))
                time_class = classify_time_column(column)
                attributes.append({
                    "source_family": family,
                    "relative_path": rel,
                    "column_name": column,
                    "semantic_name": "",
                    **profile,
                    "unit": "",
                    "cadence": "",
                    "station_scoped": bool(station_cols),
                    "target_derived_hint": target_derived,
                    "time_class": time_class,
                    "quality_field_hint": any(q in low for q in QUALITY_HINTS),
                    "eligibility": "UNRESOLVED",
                    "availability_proof": "",
                    "blocker": "CENSUS_ONLY_REQUIRES_SOURCE_TIME_AUDIT",
                    "plausible_roles": "",
                    "tested_experiments": "",
                })
                if time_class != "NOT_TIME":
                    parsed = pd.to_datetime(series, errors="coerce", utc=True) if len(series) else pd.Series(dtype="datetime64[ns, UTC]")
                    timestamp_rows.append({
                        "source_family": family,
                        "relative_path": rel,
                        "column_name": column,
                        "time_class": time_class,
                        "sample_count": len(series),
                        "parseable_count": int(parsed.notna().sum()) if len(parsed) else 0,
                        "min_utc": safe_value(parsed.min()) if parsed.notna().any() else None,
                        "max_utc": safe_value(parsed.max()) if parsed.notna().any() else None,
                        "timezone_status": "PARSED_AS_UTC_OR_OFFSET" if parsed.notna().any() else "UNRESOLVED",
                        "cutoff_relationship": "UNRESOLVED",
                    })
                if profile["sentinel_count"]:
                    quality.append({
                        "source_family": family,
                        "relative_path": rel,
                        "severity": "WARNING",
                        "finding": "NUMERIC_SENTINEL_VALUES",
                        "count": profile["sentinel_count"],
                        "details": column,
                    })
                if profile["constant_sample"] and profile["non_null_count"] > 1:
                    quality.append({
                        "source_family": family,
                        "relative_path": rel,
                        "severity": "INFO",
                        "finding": "CONSTANT_IN_PROFILE",
                        "count": profile["non_null_count"],
                        "details": column,
                    })

            # Discover station IDs and variable coverage.
            for station_col in station_cols:
                if station_col not in frame or frame.empty:
                    continue
                values = frame[station_col].dropna().astype(str).str.strip()
                values = values[values != ""]
                date_col = time_cols[0] if time_cols else None
                parsed_dates = (
                    pd.to_datetime(frame[date_col], errors="coerce", utc=True)
                    if date_col and date_col in frame else None
                )
                for station_id, index in values.groupby(values).groups.items():
                    subset = frame.loc[index]
                    station_ids.setdefault(station_id, {
                        "station_id": station_id,
                        "raw_aliases": set(),
                        "source_families": set(),
                        "files": set(),
                        "first_seen": None,
                        "last_seen": None,
                    })
                    item = station_ids[station_id]
                    item["raw_aliases"].add(station_id)
                    item["source_families"].add(family)
                    item["files"].add(rel)
                    if parsed_dates is not None:
                        dates = parsed_dates.loc[index].dropna()
                        if not dates.empty:
                            first = dates.min()
                            last = dates.max()
                            item["first_seen"] = min(
                                [x for x in (item["first_seen"], first) if x is not None]
                            )
                            item["last_seen"] = max(
                                [x for x in (item["last_seen"], last) if x is not None]
                            )
                    station_rows.append({
                        "source_family": family,
                        "relative_path": rel,
                        "station_column": station_col,
                        "station_id": station_id,
                        "profile_rows": len(subset),
                        "profile_scope": profile_scope,
                        "date_column": date_col or "",
                        "date_start": safe_value(parsed_dates.loc[index].min()) if parsed_dates is not None and parsed_dates.loc[index].notna().any() else None,
                        "date_end": safe_value(parsed_dates.loc[index].max()) if parsed_dates is not None and parsed_dates.loc[index].notna().any() else None,
                    })
                    non_dimension_columns = [
                        c for c in columns
                        if c != station_col and c not in time_cols and c not in quality_cols
                    ]
                    for variable in non_dimension_columns:
                        station_coverage.append({
                            "source_family": family,
                            "relative_path": rel,
                            "station_id": station_id,
                            "variable": variable,
                            "profile_rows": len(subset),
                            "non_null_count": int(subset[variable].notna().sum()),
                            "non_null_rate": float(subset[variable].notna().mean()) if len(subset) else None,
                            "date_start": safe_value(parsed_dates.loc[index].min()) if parsed_dates is not None and parsed_dates.loc[index].notna().any() else None,
                            "date_end": safe_value(parsed_dates.loc[index].max()) if parsed_dates is not None and parsed_dates.loc[index].notna().any() else None,
                            "profile_scope": profile_scope,
                        })

        except Exception as exc:
            unreadable.append({
                "relative_path": path.relative_to(repo).as_posix() if repo in path.parents else str(path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    station_id_rows = []
    for station_id, item in sorted(station_ids.items()):
        station_id_rows.append({
            "station_id": station_id,
            "aliases": "|".join(sorted(item["raw_aliases"])),
            "source_families": "|".join(sorted(item["source_families"])),
            "file_count": len(item["files"]),
            "first_seen": safe_value(item["first_seen"]),
            "last_seen": safe_value(item["last_seen"]),
            "station_name": "",
            "latitude": "",
            "longitude": "",
            "elevation_m": "",
            "metadata_status": "DISCOVER_AND_VERIFY",
        })

    write_csv(output / "table_inventory.csv", tables)
    write_csv(output / "attribute_catalog.csv", attributes)
    write_csv(output / "timestamp_field_catalog.csv", timestamp_rows)
    write_csv(output / "station_file_coverage.csv", station_rows)
    write_csv(output / "station_variable_coverage.csv", station_coverage)
    write_csv(output / "station_ids.csv", station_id_rows)
    write_csv(output / "data_quality_findings.csv", quality)
    write_csv(output / "unreadable_data_files.csv", unreadable, ["relative_path","error_type","error"])
    manifest = {
        "created_at_utc": utc_now(),
        "repo_root": str(repo),
        "datasets_root": str(datasets),
        "mode_requested": args.mode,
        "sample_rows": args.sample_rows,
        "max_full_file_mb": args.max_full_file_mb,
        "files_discovered": len(files),
        "tables_profiled": len(tables),
        "attributes_profiled": len(attributes),
        "station_ids_discovered": len(station_id_rows),
        "unreadable_files": len(unreadable),
        "warning": (
            "Sample/metadata profiles establish schema and candidate IDs, not exact full-history "
            "coverage. Run --mode full selectively for promotion inputs."
        ),
    }
    write_json(output / "census_manifest.json", manifest)
    summary = [
        "# Dataset Census Summary",
        "",
        f"- Created: {manifest['created_at_utc']}",
        f"- Tables profiled: {len(tables)}",
        f"- Attributes profiled: {len(attributes)}",
        f"- Station IDs discovered: {len(station_id_rows)}",
        f"- Unreadable files: {len(unreadable)}",
        f"- Requested mode: `{args.mode}`",
        "",
        "Eligibility remains unresolved until source-time auditing is completed.",
    ]
    (output / "census_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if not unreadable else 2


if __name__ == "__main__":
    raise SystemExit(main())
