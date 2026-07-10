#!/usr/bin/env python3
"""Create a GPT-Pro friendly attribute/value profile for data/datasets.

The output is a structured text file. It profiles every Parquet/CSV table under
``data/datasets`` and registers ZIP payloads as non-row payloads.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS_ROOT = REPO_ROOT / "data" / "datasets"
DEFAULT_OUTPUT = DEFAULT_DATASETS_ROOT / "DATASET_ATTRIBUTE_VALUE_PROFILE_FOR_GPT_PRO.txt"
DEFAULT_JSON_OUTPUT = DEFAULT_DATASETS_ROOT / "DATASET_ATTRIBUTE_VALUE_PROFILE_FOR_GPT_PRO.json"

NOMINAL_VALUE_LIMIT = 1000
UNIQUE_TRACK_LIMIT = 1001
EXAMPLE_LIMIT = 100
BATCH_SIZE = 100_000
METADATA_TIME_MARKERS = (
    "retrieved",
    "downloaded",
    "ingested",
    "exported",
    "generated",
    "attempted",
)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, TypeError, ValueError):
            return str(value)
    return value


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, bool) and missing


def normalized_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def csv_join(values: Iterable[str]) -> str:
    from io import StringIO

    handle = StringIO()
    writer = csv.writer(handle, lineterminator="")
    writer.writerow(list(values))
    return handle.getvalue()


def looks_like_hash(name: str, samples: list[str]) -> bool:
    lowered = name.lower()
    if "sha" in lowered or "hash" in lowered:
        return True
    return bool(samples) and all(len(v) in {32, 40, 64} and all(c in "0123456789abcdef" for c in v.lower()) for v in samples[:20])


def looks_like_url(name: str, samples: list[str]) -> bool:
    lowered = name.lower()
    return "url" in lowered or any(v.startswith(("http://", "https://")) for v in samples[:20])


def looks_like_path(name: str, samples: list[str]) -> bool:
    lowered = name.lower()
    return "path" in lowered or any(("\\" in v or "/" in v) and "." in v for v in samples[:20])


def looks_like_identifier(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith("_id") or lowered in {
        "id",
        "station_id",
        "source_id",
        "bulletin_id",
        "retrieval_id",
        "model_id",
        "jma_code",
        "hko_code",
        "station_code",
    }


def looks_like_datetime_name(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered.endswith("_at")
        or lowered.endswith("_at_utc")
        or lowered.endswith("_at_hkt")
        or lowered.endswith("_utc")
        or lowered.endswith("_hkt")
        or lowered.endswith("_time")
        or lowered.endswith("_time_utc")
        or lowered.endswith("_time_hkt")
        or lowered.endswith("_timestamp")
        or lowered in {"valid_at_utc", "valid_at_hkt", "observed_at_utc", "observed_at_hkt"}
    )


def looks_like_date_name(name: str) -> bool:
    lowered = name.lower()
    return lowered == "date" or lowered.endswith("_date") or lowered in {"local_date", "target_date", "forecast_date", "cycle_date"}


def is_metadata_time_name(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in METADATA_TIME_MARKERS)


def looks_like_temporal_name(name: str) -> bool:
    return looks_like_datetime_name(name) or looks_like_date_name(name)


@dataclass
class AttributeProfile:
    name: str
    source_dtype: str
    total_rows: int = 0
    non_null_count: int = 0
    null_count: int = 0
    numeric_count: int = 0
    numeric_sum: float = 0.0
    numeric_min: float | None = None
    numeric_max: float | None = None
    boolean_counts: Counter[str] = field(default_factory=Counter)
    datetime_count: int = 0
    datetime_min: str | None = None
    datetime_max: str | None = None
    text_count: int = 0
    total_text_length: int = 0
    max_text_length: int = 0
    blank_text_count: int = 0
    newline_text_count: int = 0
    text_samples: list[str] = field(default_factory=list)
    unique_values: set[str] = field(default_factory=set)
    unique_overflow: bool = False
    value_counts: Counter[str] = field(default_factory=Counter)
    observed_kinds: set[str] = field(default_factory=set)

    def update(self, series: pd.Series) -> None:
        self.total_rows += int(len(series))
        non_null = series[~series.map(is_missing)]
        self.non_null_count += int(len(non_null))
        self.null_count += int(len(series) - len(non_null))
        if non_null.empty:
            return

        if pd.api.types.is_bool_dtype(non_null):
            self.observed_kinds.add("boolean")
            for value, count in non_null.value_counts(dropna=True).items():
                self.boolean_counts[str(bool(value)).lower()] += int(count)
            return

        if pd.api.types.is_numeric_dtype(non_null):
            self.observed_kinds.add("numeric")
            numeric = pd.to_numeric(non_null, errors="coerce").dropna()
            if numeric.empty:
                return
            self.numeric_count += int(len(numeric))
            self.numeric_sum += float(numeric.sum())
            minimum = float(numeric.min())
            maximum = float(numeric.max())
            self.numeric_min = minimum if self.numeric_min is None else min(self.numeric_min, minimum)
            self.numeric_max = maximum if self.numeric_max is None else max(self.numeric_max, maximum)
            return

        if pd.api.types.is_datetime64_any_dtype(non_null):
            self.observed_kinds.add("datetime")
            self._update_datetime(pd.to_datetime(non_null, errors="coerce", utc=True))
            return

        self.observed_kinds.add("text")
        text_values = [normalized_text(value) for value in non_null.tolist()]
        self._update_text(text_values)

        if looks_like_temporal_name(self.name):
            parse_candidates = [value for value in text_values if value.strip()]
            parsed = pd.to_datetime(parse_candidates, errors="coerce", utc=True, format="mixed")
            if parsed.notna().any():
                self._update_datetime(parsed)

    def _update_datetime(self, parsed: Any) -> None:
        parsed_series = pd.Series(parsed).dropna()
        if parsed_series.empty:
            return
        self.datetime_count += int(len(parsed_series))
        minimum = parsed_series.min().isoformat()
        maximum = parsed_series.max().isoformat()
        self.datetime_min = minimum if self.datetime_min is None else min(self.datetime_min, minimum)
        self.datetime_max = maximum if self.datetime_max is None else max(self.datetime_max, maximum)

    def _update_text(self, text_values: list[str]) -> None:
        self.text_count += len(text_values)
        for text in text_values:
            self.total_text_length += len(text)
            self.max_text_length = max(self.max_text_length, len(text))
            if text.strip() == "":
                self.blank_text_count += 1
            if "\n" in text or "\r" in text:
                self.newline_text_count += 1
            if len(self.text_samples) < EXAMPLE_LIMIT and text not in self.text_samples:
                self.text_samples.append(text)
            if not self.unique_overflow:
                self.unique_values.add(text)
                self.value_counts[text] += 1
                if len(self.unique_values) > UNIQUE_TRACK_LIMIT:
                    self.unique_overflow = True
                    self.unique_values.clear()
                    self.value_counts.clear()

    def semantic_class(self) -> tuple[str, str]:
        samples = self.text_samples
        if self.non_null_count == 0:
            return "empty_or_all_null", "no non-null values exist"
        if "numeric" in self.observed_kinds and self.numeric_count == self.non_null_count:
            return "numeric", "all non-null values are numeric"
        if "boolean" in self.observed_kinds and sum(self.boolean_counts.values()) == self.non_null_count:
            return "boolean", "all non-null values are boolean"
        non_blank_text_count = self.non_null_count - self.blank_text_count
        if (
            self.datetime_count > 0
            and self.datetime_count == non_blank_text_count
            and looks_like_temporal_name(self.name)
        ):
            return "datetime_or_date", "column name and values parse as date/time"
        if looks_like_hash(self.name, samples):
            return "hash_identifier", "name or values look like hashes"
        if looks_like_url(self.name, samples):
            return "url", "name or values look like URLs"
        if looks_like_path(self.name, samples):
            return "path_or_file_reference", "name or values look like filesystem/object paths"
        if looks_like_identifier(self.name):
            return "identifier_or_code", "column name is an identifier/code"
        avg_len = self.total_text_length / self.text_count if self.text_count else 0.0
        if self.newline_text_count > 0 or avg_len >= 80 or self.name.lower().endswith("_text"):
            return "free_natural_text", "long text, newline text, or *_text column"
        if not self.unique_overflow and len(self.unique_values) <= NOMINAL_VALUE_LIMIT:
            return "nominal_text", "bounded distinct text values"
        return "high_cardinality_text", "too many distinct values to treat as nominal"

    def to_lines(self) -> list[str]:
        semantic_class, reason = self.semantic_class()
        unique_count = f">{UNIQUE_TRACK_LIMIT}" if self.unique_overflow else str(len(self.unique_values))
        numeric_mean = self.numeric_sum / self.numeric_count if self.numeric_count else None
        avg_text_length = self.total_text_length / self.text_count if self.text_count else None

        lines = [
            "BEGIN_ATTRIBUTE",
            f"name: {json.dumps(self.name, ensure_ascii=False)}",
            f"source_dtype: {json.dumps(self.source_dtype, ensure_ascii=False)}",
            f"semantic_class: {semantic_class}",
            f"classification_reason: {json.dumps(reason, ensure_ascii=False)}",
            f"total_rows_seen: {self.total_rows}",
            f"non_null_count: {self.non_null_count}",
            f"null_count: {self.null_count}",
            f"null_pct: {round(self.null_count / self.total_rows, 6) if self.total_rows else None}",
            f"unique_count: {unique_count}",
        ]

        if semantic_class == "numeric":
            lines.extend(
                [
                    f"numeric_min: {json.dumps(json_value(self.numeric_min))}",
                    f"numeric_max: {json.dumps(json_value(self.numeric_max))}",
                    f"numeric_avg: {json.dumps(json_value(numeric_mean))}",
                    f"numeric_mean: {json.dumps(json_value(numeric_mean))}",
                ],
            )
        elif semantic_class == "boolean":
            lines.append(f"boolean_value_counts_json: {json.dumps(dict(self.boolean_counts), ensure_ascii=False, sort_keys=True)}")
        elif semantic_class == "datetime_or_date":
            lines.extend(
                [
                    f"datetime_min: {json.dumps(self.datetime_min, ensure_ascii=False)}",
                    f"datetime_max: {json.dumps(self.datetime_max, ensure_ascii=False)}",
                ],
            )

        if semantic_class == "nominal_text":
            nominal_values = sorted(self.unique_values)
            lines.append(f"nominal_values_count: {len(nominal_values)}")
            lines.append(f"nominal_values_json: {json.dumps(nominal_values, ensure_ascii=False)}")
            lines.append(
                "nominal_value_counts_json: "
                + json.dumps(dict(self.value_counts.most_common()), ensure_ascii=False),
            )
        elif semantic_class == "free_natural_text":
            lines.append(f"text_avg_length: {json.dumps(json_value(avg_text_length))}")
            lines.append(f"text_max_length: {self.max_text_length}")
            lines.append(f"free_text_examples_count: {len(self.text_samples[:EXAMPLE_LIMIT])}")
            lines.append(f"free_text_examples_csv: {json.dumps(csv_join(self.text_samples[:EXAMPLE_LIMIT]), ensure_ascii=False)}")
        elif semantic_class in {"identifier_or_code", "hash_identifier", "url", "path_or_file_reference", "high_cardinality_text"}:
            lines.append(f"examples_count: {len(self.text_samples[:EXAMPLE_LIMIT])}")
            lines.append(f"examples_csv: {json.dumps(csv_join(self.text_samples[:EXAMPLE_LIMIT]), ensure_ascii=False)}")

        lines.append("END_ATTRIBUTE")
        return lines

    def to_json_object(self, *, dataset_id: str, source_file: str, file_type: str) -> dict[str, Any]:
        semantic_class, reason = self.semantic_class()
        unique_count: int | str = f">{UNIQUE_TRACK_LIMIT}" if self.unique_overflow else len(self.unique_values)
        numeric_mean = self.numeric_sum / self.numeric_count if self.numeric_count else None
        avg_text_length = self.total_text_length / self.text_count if self.text_count else None
        payload: dict[str, Any] = {
            "dataset_id": dataset_id,
            "source_file": source_file,
            "file_type": file_type,
            "name": self.name,
            "source_dtype": self.source_dtype,
            "semantic_class": semantic_class,
            "classification_reason": reason,
            "total_rows_seen": self.total_rows,
            "non_null_count": self.non_null_count,
            "null_count": self.null_count,
            "null_pct": round(self.null_count / self.total_rows, 6) if self.total_rows else None,
            "unique_count": unique_count,
        }

        if semantic_class == "numeric":
            payload["numeric"] = {
                "min": json_value(self.numeric_min),
                "max": json_value(self.numeric_max),
                "avg": json_value(numeric_mean),
                "mean": json_value(numeric_mean),
            }
        elif semantic_class == "boolean":
            payload["boolean_value_counts"] = dict(self.boolean_counts)
        elif semantic_class == "datetime_or_date":
            payload["datetime"] = {
                "min": self.datetime_min,
                "max": self.datetime_max,
            }

        if semantic_class == "nominal_text":
            payload["nominal_values"] = sorted(self.unique_values)
            payload["nominal_value_counts"] = dict(self.value_counts.most_common())
        elif semantic_class == "free_natural_text":
            payload["text"] = {
                "avg_length": json_value(avg_text_length),
                "max_length": self.max_text_length,
                "examples": self.text_samples[:EXAMPLE_LIMIT],
            }
        elif semantic_class in {
            "identifier_or_code",
            "hash_identifier",
            "url",
            "path_or_file_reference",
            "high_cardinality_text",
        }:
            payload["examples"] = self.text_samples[:EXAMPLE_LIMIT]

        return payload


@dataclass
class TableProfile:
    dataset_id: str
    source_file: str
    file_type: str
    row_count: int
    byte_size: int
    columns: dict[str, AttributeProfile]


def date_range_for_attribute(profile: TableProfile, attribute: AttributeProfile) -> dict[str, Any] | None:
    semantic_class, _ = attribute.semantic_class()
    if semantic_class != "datetime_or_date" or attribute.datetime_min is None or attribute.datetime_max is None:
        return None
    range_kind = "metadata_timestamp" if is_metadata_time_name(attribute.name) else "data_temporal_coverage"
    return {
        "dataset_id": profile.dataset_id,
        "source_file": profile.source_file,
        "file_type": profile.file_type,
        "attribute": attribute.name,
        "range_kind": range_kind,
        "min": attribute.datetime_min,
        "max": attribute.datetime_max,
        "non_null_count": attribute.non_null_count,
        "parsed_datetime_count": attribute.datetime_count,
    }


def date_ranges_for_table(profile: TableProfile, *, range_kind: str | None = None) -> list[dict[str, Any]]:
    return [
        date_range
        for attribute in profile.columns.values()
        if (date_range := date_range_for_attribute(profile, attribute)) is not None
        and (range_kind is None or date_range["range_kind"] == range_kind)
    ]


def summarize_data_range(date_ranges: list[dict[str, Any]], *, empty_basis: str) -> dict[str, Any]:
    if not date_ranges:
        return {
            "min": None,
            "max": None,
            "basis": empty_basis,
            "date_attribute_count": 0,
            "date_attributes": [],
        }
    return {
        "min": min(date_range["min"] for date_range in date_ranges),
        "max": max(date_range["max"] for date_range in date_ranges),
        "basis": "min/max across attributes classified as datetime_or_date",
        "date_attribute_count": len(date_ranges),
        "date_attributes": date_ranges,
    }


def parquet_column_types(path: Path) -> dict[str, str]:
    schema = pq.ParquetFile(path).schema_arrow
    return {field.name: str(field.type) for field in schema}


def csv_column_types(path: Path) -> dict[str, str]:
    try:
        frame = pd.read_csv(path, nrows=1000)
    except pd.errors.EmptyDataError:
        return {}
    return {str(column): str(dtype) for column, dtype in frame.dtypes.items()}


def parquet_chunks(path: Path) -> Iterator[pd.DataFrame]:
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
        yield batch.to_pandas()


def csv_chunks(path: Path) -> Iterator[pd.DataFrame]:
    try:
        yield from pd.read_csv(path, chunksize=BATCH_SIZE)
    except pd.errors.EmptyDataError:
        return


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def profile_table(path: Path, datasets_root: Path) -> TableProfile:
    relative = path.relative_to(datasets_root).as_posix()
    dataset_id = relative.split("/")[0] if "/" in relative else "[root]"
    file_type = path.suffix.lower().lstrip(".")
    byte_size = path.stat().st_size

    if file_type == "parquet":
        row_count = pq.ParquetFile(path).metadata.num_rows
        source_types = parquet_column_types(path)
        chunks = parquet_chunks(path)
    elif file_type == "csv":
        row_count = count_csv_rows(path)
        source_types = csv_column_types(path)
        chunks = csv_chunks(path)
    else:
        return TableProfile(dataset_id, relative, file_type, 0, byte_size, {})

    columns = {
        name: AttributeProfile(name=name, source_dtype=dtype)
        for name, dtype in source_types.items()
    }
    for chunk in chunks:
        for column in chunk.columns:
            name = str(column)
            if name not in columns:
                columns[name] = AttributeProfile(name=name, source_dtype=str(chunk[column].dtype))
            columns[name].update(chunk[column])
    return TableProfile(dataset_id, relative, file_type, row_count, byte_size, columns)


def discover_files(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".parquet", ".csv", ".zip"}
    ]


def write_report(profiles: list[TableProfile], output: Path, datasets_root: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    total_tables = sum(1 for profile in profiles if profile.file_type in {"parquet", "csv"})
    total_rows = sum(profile.row_count for profile in profiles if profile.file_type in {"parquet", "csv"})
    lines = [
        "HKG_TMAX_DATASET_ATTRIBUTE_VALUE_PROFILE_FOR_GPT_PRO",
        f"generated_at_utc: {utc_now_iso()}",
        f"source_root: {datasets_root}",
        f"row_tables_profiled: {total_tables}",
        f"row_table_rows_total: {total_rows}",
        "format_note: tagged sections; JSON fields are safe to parse as JSON; CSV example fields are CSV-escaped single-line strings",
        "classification_note: nominal text includes all distinct values only when distinct count is <= 1000; free natural text gets 100 CSV examples",
        "",
    ]
    current_dataset: str | None = None
    for profile in profiles:
        if profile.dataset_id != current_dataset:
            if current_dataset is not None:
                lines.append("END_DATASET")
                lines.append("")
            current_dataset = profile.dataset_id
            lines.append("BEGIN_DATASET")
            lines.append(f"dataset_id: {json.dumps(current_dataset, ensure_ascii=False)}")
        lines.extend(
            [
                "BEGIN_TABLE",
                f"source_file: {json.dumps(profile.source_file, ensure_ascii=False)}",
                f"file_type: {profile.file_type}",
                f"row_count: {profile.row_count}",
                f"byte_size: {profile.byte_size}",
                f"attribute_count: {len(profile.columns)}",
            ],
        )
        if profile.file_type == "zip":
            lines.append("table_note: ZIP payload registered only; row attributes not profiled")
        for column_profile in profile.columns.values():
            lines.extend(column_profile.to_lines())
        lines.append("END_TABLE")
    if current_dataset is not None:
        lines.append("END_DATASET")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_profile_to_json(profile: TableProfile) -> dict[str, Any]:
    table_note = None
    if profile.file_type == "zip":
        table_note = "ZIP payload registered only; row attributes not profiled"
    data_date_ranges = date_ranges_for_table(profile, range_kind="data_temporal_coverage")
    metadata_date_ranges = date_ranges_for_table(profile, range_kind="metadata_timestamp")
    all_date_ranges = data_date_ranges + metadata_date_ranges
    payload: dict[str, Any] = {
        "dataset_id": profile.dataset_id,
        "source_file": profile.source_file,
        "file_type": profile.file_type,
        "row_count": profile.row_count,
        "byte_size": profile.byte_size,
        "attribute_count": len(profile.columns),
        "data_range": summarize_data_range(
            data_date_ranges,
            empty_basis="no non-metadata attributes were classified as datetime_or_date",
        ),
        "metadata_timestamp_range": summarize_data_range(
            metadata_date_ranges,
            empty_basis="no metadata timestamp attributes were classified as datetime_or_date",
        ),
        "all_datetime_ranges": summarize_data_range(
            all_date_ranges,
            empty_basis="no attributes were classified as datetime_or_date",
        ),
        "attributes": [
            column_profile.to_json_object(
                dataset_id=profile.dataset_id,
                source_file=profile.source_file,
                file_type=profile.file_type,
            )
            for column_profile in profile.columns.values()
        ],
    }
    if table_note is not None:
        payload["table_note"] = table_note
    return payload


def write_json_report(profiles: list[TableProfile], output: Path, datasets_root: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_order: list[str] = []
    grouped_tables: dict[str, list[TableProfile]] = {}
    for profile in profiles:
        if profile.dataset_id not in grouped_tables:
            dataset_order.append(profile.dataset_id)
            grouped_tables[profile.dataset_id] = []
        grouped_tables[profile.dataset_id].append(profile)

    total_tables = sum(1 for profile in profiles if profile.file_type in {"parquet", "csv"})
    total_rows = sum(profile.row_count for profile in profiles if profile.file_type in {"parquet", "csv"})
    total_attributes = sum(len(profile.columns) for profile in profiles)
    data_date_ranges = [
        date_range
        for profile in profiles
        for date_range in date_ranges_for_table(profile, range_kind="data_temporal_coverage")
    ]
    metadata_date_ranges = [
        date_range
        for profile in profiles
        for date_range in date_ranges_for_table(profile, range_kind="metadata_timestamp")
    ]
    all_date_ranges = data_date_ranges + metadata_date_ranges
    payload = {
        "profile_name": "HKG_TMAX_DATASET_ATTRIBUTE_VALUE_PROFILE_FOR_GPT_PRO",
        "generated_at_utc": utc_now_iso(),
        "source_root": str(datasets_root),
        "summary": {
            "datasets_profiled": len(dataset_order),
            "files_profiled": len(profiles),
            "row_tables_profiled": total_tables,
            "row_table_rows_total": total_rows,
            "attributes_profiled": total_attributes,
        },
        "data_range": summarize_data_range(
            data_date_ranges,
            empty_basis="no non-metadata attributes were classified as datetime_or_date",
        ),
        "metadata_timestamp_range": summarize_data_range(
            metadata_date_ranges,
            empty_basis="no metadata timestamp attributes were classified as datetime_or_date",
        ),
        "all_datetime_ranges": summarize_data_range(
            all_date_ranges,
            empty_basis="no attributes were classified as datetime_or_date",
        ),
        "format_note": (
            "Hierarchical JSON. Every attribute repeats dataset_id, source_file, and file_type "
            "so attributes remain self-contained when extracted."
        ),
        "classification_note": (
            "nominal_text includes all distinct values only when distinct count is <= 1000; "
            "free_natural_text includes up to 100 examples as an array."
        ),
        "datasets": [
            {
                "dataset_id": dataset_id,
                "data_range": summarize_data_range(
                    [
                        date_range
                        for profile in grouped_tables[dataset_id]
                        for date_range in date_ranges_for_table(profile, range_kind="data_temporal_coverage")
                    ],
                    empty_basis="no non-metadata attributes were classified as datetime_or_date",
                ),
                "metadata_timestamp_range": summarize_data_range(
                    [
                        date_range
                        for profile in grouped_tables[dataset_id]
                        for date_range in date_ranges_for_table(profile, range_kind="metadata_timestamp")
                    ],
                    empty_basis="no metadata timestamp attributes were classified as datetime_or_date",
                ),
                "all_datetime_ranges": summarize_data_range(
                    [
                        date_range
                        for profile in grouped_tables[dataset_id]
                        for date_range in date_ranges_for_table(profile)
                    ],
                    empty_basis="no attributes were classified as datetime_or_date",
                ),
                "tables": [table_profile_to_json(profile) for profile in grouped_tables[dataset_id]],
            }
            for dataset_id in dataset_order
        ],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile all dataset attributes for GPT-Pro review.")
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    datasets_root = args.datasets_root.resolve()
    files = discover_files(datasets_root)
    profiles: list[TableProfile] = []
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(datasets_root).as_posix()
        print(f"[{index}/{len(files)}] profiling {relative}", flush=True)
        profiles.append(profile_table(path, datasets_root))
    write_report(profiles, args.output.resolve(), datasets_root)
    write_json_report(profiles, args.json_output.resolve(), datasets_root)
    print(f"wrote {args.output.resolve()}", flush=True)
    print(f"wrote {args.json_output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
