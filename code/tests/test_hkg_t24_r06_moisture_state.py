from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from scripts.run_hkg_t24_r06_moisture_state import (
    HKO_STATION,
    HKT,
    asof_values,
    make_cutoffs,
    valid_columns,
    zip_entry_in_sampling_window,
)


def test_zip_entry_filter_keeps_exact_operational_snapshot_vintages() -> None:
    assert zip_entry_in_sampling_window("20230714-1447-latest_1min_humidity.csv")
    assert zip_entry_in_sampling_window("20230714-1347-latest_1min_humidity.csv")
    assert not zip_entry_in_sampling_window("20230714-1517-latest_1min_humidity.csv")


def test_asof_values_uses_available_at_and_rejects_post_cutoff_records() -> None:
    base = pd.DataFrame({"target_date": [pd.Timestamp("2023-07-15")]})
    cutoffs = make_cutoffs(base)
    observations = pd.DataFrame(
        [
            {
                "station": HKO_STATION,
                "variable": "relative_humidity_pct",
                "observed_at_hkt": datetime(2023, 7, 14, 14, 40, tzinfo=HKT),
                "available_at_hkt": datetime(2023, 7, 14, 14, 40, tzinfo=HKT) + timedelta(minutes=20),
                "value": 72.0,
                "source_file_hash": "old",
            },
            {
                "station": HKO_STATION,
                "variable": "relative_humidity_pct",
                "observed_at_hkt": datetime(2023, 7, 14, 14, 50, tzinfo=HKT),
                "available_at_hkt": datetime(2023, 7, 14, 14, 50, tzinfo=HKT) + timedelta(minutes=20),
                "value": 99.0,
                "source_file_hash": "future",
            },
        ]
    )

    out = asof_values(
        observations,
        cutoffs,
        variable="relative_humidity_pct",
        offset_hours=0,
        station_filter={HKO_STATION},
    )

    assert out.iloc[0]["value"] == 72.0
    assert out.iloc[0]["source_file_hash"] == "old"


def test_valid_columns_removes_duplicates_and_all_null_columns() -> None:
    features = pd.DataFrame(
        {
            "a": [1.0, 2.0],
            "all_null": [np.nan, np.nan],
            "text": ["x", "y"],
        }
    )

    assert valid_columns(features, ["a", "a", "all_null", "text", "missing"]) == ("a",)
