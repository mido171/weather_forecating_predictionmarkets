from __future__ import annotations

import argparse
from datetime import date

import pytest

from scripts import backfill_public_weather_to_postgres as backfill


def backfill_args() -> argparse.Namespace:
    return backfill.build_arg_parser().parse_args(
        ["--start-date", "2026-06-25", "--end-date", "2026-06-25"]
    )


def completed_summary(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "status": "complete",
        "execution_mode": "serial",
        "elapsed_seconds": 12.5,
        "start_date": "2026-06-25",
        "end_date": "2026-06-25",
        "source_issues_touched": 4,
        "fetch_ok": 4,
        "fetch_failed": 0,
        "normalize_ok": 4,
        "normalize_failed": 0,
        "station_features_upserted": 8,
        "area_features_upserted": 2,
        "raw_bytes_deleted": 1024,
        "max_staging_bytes": 2048,
        "final_staging_bytes": 0,
        "max_raw_object_bytes": 512,
        "min_free_disk_bytes": 4096,
        "by_source": {"gfs": {"fetch_ok": 4}},
    }


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
    assert serial.experiment_dir == (
        backfill.REPO_ROOT
        / "experiments"
        / "campaigns"
        / "hkg-tmax"
        / backfill.EXPERIMENT_ID
    )


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


def test_experiment_writes_one_readme_and_keeps_machine_artifacts(tmp_path) -> None:
    args = backfill_args()
    run_id = "run-20260625"
    summary = completed_summary(run_id)

    backfill.initialize_experiment_docs(tmp_path, args, run_id)
    backfill.write_results(tmp_path, summary)
    backfill.write_status(tmp_path, "COMPLETE", summary)

    markdown_files = sorted(
        path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*.md")
    )
    assert markdown_files == ["README.md"]
    assert (tmp_path / "RUN_CONFIG.yaml").is_file()
    assert (tmp_path / "DATA_MANIFEST.yaml").is_file()
    assert (tmp_path / "STATUS.yaml").is_file()
    assert (tmp_path / "results" / "metrics.json").is_file()
    assert (tmp_path / "results" / "runs" / run_id / "metrics.json").is_file()

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "## Acquisition Contract" in readme
    assert "### Hypothesis" in readme
    assert "### Protocol" in readme
    assert "### As-Of Contract" in readme
    assert "### Reproduce" in readme
    assert "## Latest Run" in readme
    assert "### Conclusion" in readme
    assert f"Run id: `{run_id}`" in readme


def test_readme_update_is_idempotent_across_reruns(tmp_path) -> None:
    args = backfill_args()
    first_summary = completed_summary("run-one")
    (tmp_path / "README.md").write_text(
        "# Curated Experiment Summary\n\nManual historical context must remain.\n",
        encoding="utf-8",
    )

    backfill.initialize_experiment_docs(tmp_path, args, "run-one")
    backfill.write_results(tmp_path, first_summary)
    first_readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "Manual historical context must remain." in first_readme

    backfill.initialize_experiment_docs(tmp_path, args, "run-one")
    backfill.write_results(tmp_path, first_summary)
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == first_readme

    second_summary = completed_summary("run-two")
    backfill.initialize_experiment_docs(tmp_path, args, "run-two")
    backfill.write_results(tmp_path, second_summary)
    rerun_readme = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert rerun_readme.count("## Acquisition Contract") == 1
    assert rerun_readme.count("## Latest Run") == 1
    assert "run-one" not in rerun_readme
    assert "run-two" in rerun_readme
    assert "Manual historical context must remain." in rerun_readme
    assert (tmp_path / "results" / "runs" / "run-one" / "metrics.json").is_file()
    assert (tmp_path / "results" / "runs" / "run-two" / "metrics.json").is_file()
