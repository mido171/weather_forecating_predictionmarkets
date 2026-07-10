from __future__ import annotations

import pytest

from scripts.profile_dataset_attributes_for_gpt_pro import (
    AttributeProfile,
    TableProfile,
    date_ranges_for_table,
    summarize_data_range,
    table_profile_to_json,
)


def test_table_and_dataset_data_ranges_are_built_from_date_attributes() -> None:
    pd = pytest.importorskip("pandas")

    target_date = AttributeProfile(name="target_date", source_dtype="object")
    target_date.update(pd.Series(["2020-01-02", "2020-01-04", None]))
    forecast_max = AttributeProfile(name="forecast_max_c", source_dtype="float64")
    forecast_max.update(pd.Series([21.5, 22.0, None]))
    table = TableProfile(
        dataset_id="sample_dataset",
        source_file="sample_dataset/forecast.parquet",
        file_type="parquet",
        row_count=3,
        byte_size=123,
        columns={"target_date": target_date, "forecast_max_c": forecast_max},
    )

    date_ranges = date_ranges_for_table(table)
    summary = summarize_data_range(date_ranges, empty_basis="no usable dates")
    table_json = table_profile_to_json(table)

    assert date_ranges == [
        {
            "dataset_id": "sample_dataset",
            "source_file": "sample_dataset/forecast.parquet",
            "file_type": "parquet",
            "attribute": "target_date",
            "range_kind": "data_temporal_coverage",
            "min": "2020-01-02T00:00:00+00:00",
            "max": "2020-01-04T00:00:00+00:00",
            "non_null_count": 2,
            "parsed_datetime_count": 2,
        },
    ]
    assert summary["min"] == "2020-01-02T00:00:00+00:00"
    assert summary["max"] == "2020-01-04T00:00:00+00:00"
    assert summary["date_attribute_count"] == 1
    assert table_json["data_range"] == summary


def test_empty_data_range_is_explicit() -> None:
    summary = summarize_data_range([], empty_basis="no usable dates")

    assert summary == {
        "min": None,
        "max": None,
        "basis": "no usable dates",
        "date_attribute_count": 0,
        "date_attributes": [],
    }


def test_blank_date_strings_still_classify_as_datetime() -> None:
    pd = pytest.importorskip("pandas")

    image_time = AttributeProfile(name="image_time_hkt", source_dtype="string")
    image_time.update(pd.Series(["", "2026-06-17T08:15:00+08:00", "2026-06-17T09:00:00+08:00"]))

    assert image_time.semantic_class()[0] == "datetime_or_date"
    assert image_time.datetime_min == "2026-06-17T00:15:00+00:00"
    assert image_time.datetime_max == "2026-06-17T01:00:00+00:00"
