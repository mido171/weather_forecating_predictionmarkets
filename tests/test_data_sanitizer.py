from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.data_sanitizer.data_sanitizer import (
    REQUIRED_COLUMNS,
    dedupe_sanitized_dataframe,
    run_cli,
    sanitize_observations_dataframe,
)


def _base_row() -> dict[str, object]:
    return {
        "request_location_id": "KNYC:9:US",
        "valid_time_utc": "2025-01-01T12:00:00Z",
        "temp": 50.0,
        "dew_pt": 45.0,
        "rh": 70.0,
        "pressure": 29.9,
        "vis": 10.0,
        "wspd": 5.0,
        "wdir": 180.0,
        "gust": 10.0,
        "precip_hrly": 0.0,
        "clds": "SCT",
        "wx_phrase": "Partly Cloudy",
        "uv_index": 3.0,
        "uv_desc": "Moderate",
        "wdir_cardinal": "S",
    }


def _df(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[REQUIRED_COLUMNS].copy()


def test_sentinel_and_bounds_and_cross_field() -> None:
    r = _base_row()
    r["wdir"] = 999
    r["uv_index"] = -10
    r["wspd"] = 1087
    r["temp"] = 208
    r["dew_pt"] = 80
    df = _df([r])
    out, meta = sanitize_observations_dataframe(df, station_universe={"KNYC:9:US"}, collect_triggered_rules=True)
    assert out.shape[0] == 1
    assert pd.isna(out.loc[0, "wdir"])
    assert pd.isna(out.loc[0, "uv_index"])
    assert pd.isna(out.loc[0, "wspd"])
    assert pd.isna(out.loc[0, "temp"])
    assert meta["rule_counts"]["wdir_sentinel_999"] == 1
    assert meta["rule_counts"]["uv_negative_sentinel"] == 1
    assert meta["rule_counts"]["wspd_out_of_bounds"] == 1
    assert meta["rule_counts"]["temp_out_of_bounds"] == 1


def test_dewpoint_clip_and_categorical_normalization() -> None:
    r = _base_row()
    r["dew_pt"] = 80
    r["temp"] = 70
    r["wdir_cardinal"] = "vrb"
    r["clds"] = "overcast"
    out, meta = sanitize_observations_dataframe(_df([r]), station_universe={"KNYC:9:US"})
    assert float(out.loc[0, "dew_pt"]) == float(out.loc[0, "temp"])
    assert out.loc[0, "wdir_cardinal"] == "VAR"
    assert out.loc[0, "clds"] == "OVC"
    assert meta["rule_counts"]["dewpt_gt_temp_clipped"] == 1


def test_dedupe_best_non_null() -> None:
    a = _base_row()
    b = _base_row()
    b["temp"] = np.nan
    b["wspd"] = np.nan
    df = _df([b, a])  # second row has more non-null values
    sane, _ = sanitize_observations_dataframe(df, station_universe={"KNYC:9:US"})
    deduped, stats = dedupe_sanitized_dataframe(sane, "best_non_null")
    assert deduped.shape[0] == 1
    assert stats["rows_dropped_by_dedupe"] == 1
    assert float(deduped.iloc[0]["temp"]) == 50.0


def test_cli_report_contains_rule_counts(tmp_path: Path) -> None:
    in_csv = tmp_path / "obs.csv"
    out_csv = tmp_path / "obs.sanitized.csv.gz"
    report_json = tmp_path / "report.json"
    samples_csv = tmp_path / "samples.csv"
    manifest = tmp_path / "manifest.jsonl"
    universe_csv = tmp_path / "station_universe.csv"

    rows = []
    r1 = _base_row()
    r1["wdir"] = 999
    r1["uv_index"] = -3
    rows.append(r1)
    r2 = _base_row()
    r2["valid_time_utc"] = "bad-time"
    rows.append(r2)
    _df(rows).to_csv(in_csv, index=False)
    pd.DataFrame([{"request_location_id": "KNYC:9:US", "role": "target"}]).to_csv(universe_csv, index=False)

    args = type("Args", (), {})()
    args.input = str(in_csv)
    args.output = str(out_csv)
    args.report_out = str(report_json)
    args.samples_out = str(samples_csv)
    args.station_universe = str(universe_csv)
    args.schema_profile = ""
    args.config = ""
    args.chunksize = 1000
    args.compression = "gzip"
    args.emit_flags = "false"
    args.drop_invalid_timestamps = "true"
    args.dedupe_policy = "best_non_null"
    args.strict_columns = "true"
    args.allow_extra_columns = "false"
    args.enforce_30m_grid = "false"
    args.fill_wdir_from_cardinal = "false"
    args.manifest_path = str(manifest)
    args.max_samples = 1000

    report = run_cli(args)
    assert report["row_counts"]["input_rows"] == 2
    assert report["row_counts"]["rows_output"] == 1
    parsed = json.loads(report_json.read_text(encoding="utf-8"))
    assert "rules" in parsed
    assert parsed["rules"]["hit_counts"]["wdir_sentinel_999"] == 1
