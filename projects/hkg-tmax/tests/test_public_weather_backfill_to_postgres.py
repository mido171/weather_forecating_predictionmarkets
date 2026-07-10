from __future__ import annotations

import argparse
from datetime import date

import pytest

from scripts import backfill_public_weather_to_postgres as backfill


def test_parse_leads_default_range() -> None:
    assert backfill.parse_leads("0:48:3") == list(range(0, 49, 3))
    assert backfill.parse_leads("3,0,3") == [0, 3]


def test_date_span_is_inclusive() -> None:
    assert backfill.date_span(date(2026, 6, 25), date(2026, 6, 27)) == [
        date(2026, 6, 25),
        date(2026, 6, 26),
        date(2026, 6, 27),
    ]


def test_source_normalization_aliases() -> None:
    assert backfill.normalize_sources("gfs,gefs,himawari,radar") == {
        "gfs",
        "gefs_control",
        "himawari_b13_s0510",
        "radar",
    }


def test_expected_inventory_for_selected_window() -> None:
    days = backfill.date_span(date(2026, 6, 25), date(2026, 7, 7))
    counts = backfill.task_counts_for_days(
        days,
        {"gfs", "gefs_control", "himawari_b13_s0510", "radar"},
        [0, 6, 12, 18],
        list(range(0, 49, 3)),
    )
    assert counts["model_objects"] == 1768
    assert counts["himawari_scans"] == 1872
    assert counts["radar_expected_frames_approx"] == 1560


def test_grib_idx_selector_extracts_required_ranges() -> None:
    idx = "\n".join(
        [
            "1:0:d=2026062500:PRMSL:mean sea level:24 hour fcst:",
            "2:100:d=2026062500:CLMR:1 hybrid level:24 hour fcst:",
            "3:200:d=2026062500:TMP:2 m above ground:24 hour fcst:",
            "4:350:d=2026062500:TCDC:entire atmosphere:24 hour fcst:",
            "5:500:d=2026062500:UGRD:10 m above ground:24 hour fcst:",
        ]
    )
    ranges = backfill.parse_grib_idx_ranges(idx, object_length=650)
    assert [(row["variable"], row["offset"], row["end_offset"]) for row in ranges] == [
        ("PRMSL", 0, 99),
        ("TMP", 200, 349),
        ("TCDC", 350, 499),
        ("UGRD", 500, 649),
    ]


def test_gap_zero_range_coalescing_preserves_exact_selected_bytes() -> None:
    ranges = [
        {"offset": 0, "end_offset": 99},
        {"offset": 100, "end_offset": 199},
        {"offset": 300, "end_offset": 349},
    ]
    merged = backfill.merge_selected_ranges(ranges, max_gap_bytes=0)
    assert merged == [
        {"offset": 0, "end_offset": 199, "message_count": 2, "downloaded_gap_bytes": 0},
        {"offset": 300, "end_offset": 349, "message_count": 1},
    ]
    selected_bytes = sum(row["end_offset"] - row["offset"] + 1 for row in ranges)
    downloaded_bytes = sum(row["end_offset"] - row["offset"] + 1 for row in merged)
    assert downloaded_bytes == selected_bytes


def test_nonzero_range_coalescing_records_extra_bytes() -> None:
    ranges = [
        {"offset": 0, "end_offset": 99},
        {"offset": 200, "end_offset": 299},
    ]
    merged = backfill.merge_selected_ranges(ranges, max_gap_bytes=100)
    selected_bytes = sum(row["end_offset"] - row["offset"] + 1 for row in ranges)
    downloaded_bytes = sum(row["end_offset"] - row["offset"] + 1 for row in merged)
    assert merged == [{"offset": 0, "end_offset": 299, "message_count": 2, "downloaded_gap_bytes": 100}]
    assert downloaded_bytes - selected_bytes == 100


def test_optimized_cli_defaults_are_opt_in() -> None:
    parser = backfill.build_arg_parser()
    serial = parser.parse_args(["--start-date", "2026-06-21", "--end-date", "2026-06-21"])
    optimized = parser.parse_args(
        ["--start-date", "2026-06-21", "--end-date", "2026-06-21", "--execution-mode", "optimized"]
    )
    assert serial.execution_mode == "serial"
    assert optimized.execution_mode == "optimized"
    assert optimized.model_fetch_workers == 1
    assert optimized.model_range_workers == 1
    assert optimized.model_normalize_workers == 1
    assert optimized.himawari_workers == 1
    assert optimized.model_range_coalesce_gap_bytes == 0


def test_safe_delete_file_is_bounded_to_staging_root(tmp_path) -> None:
    staging = tmp_path / "staging"
    raw = staging / "deep" / "raw.grib2"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"abc")
    assert backfill.safe_delete_file(raw, staging) == 3
    assert not raw.exists()

    outside = tmp_path / "outside.grib2"
    outside.write_bytes(b"abc")
    with pytest.raises(RuntimeError):
        backfill.safe_delete_file(outside, staging)


def test_file_size_counts_nested_raw_bytes(tmp_path) -> None:
    root = tmp_path / "staging"
    (root / "a").mkdir(parents=True)
    (root / "a" / "one.grib2").write_bytes(b"1234")
    (root / "two.bz2").write_bytes(b"12345")
    assert backfill.file_size(root) == 9


def test_unknown_source_rejected() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        backfill.normalize_sources("gfs,unknown")
