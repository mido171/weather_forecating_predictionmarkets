from __future__ import annotations

from datetime import date

from klga_tmax.providers.gribstream.catalog import spec_summary_rows
from klga_tmax.providers.gribstream.models import ResolvedSelector
from klga_tmax.providers.gribstream.plan import (
    DEFAULT_CUTOFF_ID,
    MODEL_SPECS,
    T1245_CUTOFF_ID,
    as_of_utc,
    build_chunk,
    build_runs_chunk,
    cutoff_utc,
    effective_target_start,
    model_spec_by_id,
    runs_model_time_for_target,
    valid_times_for_target,
)


def _temperature_selector() -> tuple[ResolvedSelector, ...]:
    return (
        ResolvedSelector(
            alias="temperature_2m",
            request_variables=(
                {
                    "name": "TMP",
                    "level": "2 m above ground",
                    "info": "",
                    "alias": "temperature_2m",
                },
            ),
            variable_name="TMP",
            variable_level="2 m above ground",
            variable_info="",
            shared_parameter="temperature_2m",
            unit_hint="K",
        ),
    )


def test_prior_day_action_plan_lists_all_models_from_dates_and_credit_total() -> None:
    rows = spec_summary_rows(end_date=date(2026, 6, 28), cutoff_id=DEFAULT_CUTOFF_ID)
    assert len(rows) == 15
    assert {row["model_id"] for row in rows} == {spec.model_id for spec in MODEL_SPECS}
    assert sum(int(row["expected_total_credits"]) for row in rows) == 1_150_158
    gfs = next(row for row in rows if row["model_id"] == "gfs")
    assert gfs["catalog_archive_start"] == "2021-03-22"
    assert gfs["effective_target_from"] == "2021-03-23"
    assert gfs["buffer_minutes"] == 240


def test_t1245_action_plan_uses_catalog_archive_start_and_updated_credit_total() -> None:
    rows = spec_summary_rows(end_date=date(2026, 6, 28), cutoff_id=T1245_CUTOFF_ID)
    assert len(rows) == 15
    assert sum(int(row["expected_total_credits"]) for row in rows) == 1_150_873
    hrrr = next(row for row in rows if row["model_id"] == "hrrr")
    gfs = next(row for row in rows if row["model_id"] == "gfs")
    assert hrrr["effective_target_from"] == "2014-07-30"
    assert hrrr["target_days"] == 4352
    assert gfs["catalog_archive_start"] == "2021-03-22"
    assert gfs["effective_target_from"] == "2021-03-22"
    assert gfs["target_days"] == 1925
    assert effective_target_start(model_spec_by_id("gfs"), cutoff_id=T1245_CUTOFF_ID).isoformat() == "2021-03-22"


def test_canonical_cutoff_and_model_valid_times() -> None:
    target_date = date(2026, 6, 28)
    assert cutoff_utc(target_date, cutoff_id=DEFAULT_CUTOFF_ID).isoformat() == "2026-06-27T20:45:00+00:00"
    gfs_times = valid_times_for_target(model_spec_by_id("gfs"), target_date)
    assert len(gfs_times) == 10
    assert gfs_times[0].isoformat() == "2026-06-28T16:00:00+00:00"
    assert gfs_times[-1].isoformat() == "2026-06-29T01:00:00+00:00"
    synoptic = valid_times_for_target(model_spec_by_id("ifsoper"), target_date)
    assert [item.isoformat() for item in synoptic] == [
        "2026-06-28T18:00:00+00:00",
        "2026-06-29T00:00:00+00:00",
    ]


def test_t1245_cutoff_asof_and_model_valid_times() -> None:
    target_date = date(2026, 6, 28)
    assert cutoff_utc(target_date, cutoff_id=T1245_CUTOFF_ID).isoformat() == "2026-06-28T12:45:00+00:00"
    assert as_of_utc(target_date, model_spec_by_id("gfs"), cutoff_id=T1245_CUTOFF_ID).isoformat() == "2026-06-28T08:45:00+00:00"
    assert as_of_utc(target_date, model_spec_by_id("hrrr"), cutoff_id=T1245_CUTOFF_ID).isoformat() == "2026-06-28T10:30:00+00:00"
    assert as_of_utc(target_date, model_spec_by_id("rtma"), cutoff_id=T1245_CUTOFF_ID).isoformat() == "2026-06-28T11:45:00+00:00"
    rtma = valid_times_for_target(model_spec_by_id("rtma"), target_date, cutoff_id=T1245_CUTOFF_ID)
    nbmqmd = valid_times_for_target(model_spec_by_id("nbmqmd"), target_date, cutoff_id=T1245_CUTOFF_ID)
    assert [item.isoformat() for item in rtma] == ["2026-06-28T11:00:00+00:00"]
    assert [item.isoformat() for item in nbmqmd] == ["2026-06-29T06:00:00+00:00"]


def test_timeseries_payload_uses_dataset_scoped_body_and_single_asof() -> None:
    chunk = build_chunk(
        spec=model_spec_by_id("gfs"),
        target_start_date=date(2026, 6, 28),
        target_end_date=date(2026, 6, 28),
        coordinate_tier_name="B",
        selectors=_temperature_selector(),
    )
    payload = chunk.request_payload
    assert "model" not in payload
    assert "latitude" not in payload
    assert "longitude" not in payload
    assert payload["asOf"] == "2026-06-27T16:45:00Z"
    assert len(payload["coordinates"]) == 10
    assert payload["coordinates"][0]["name"] == "GP_KLGA_EXACT"
    assert len(payload["timesList"]) == 10
    assert payload["variables"] == [
        {
            "name": "TMP",
            "level": "2 m above ground",
            "info": "",
            "alias": "temperature_2m",
        }
    ]
    assert "members" not in payload


def test_t1245_payload_uses_target_day_asof_and_nbmqmd_06z_max18_valid_time() -> None:
    gfs = build_chunk(
        spec=model_spec_by_id("gfs"),
        target_start_date=date(2026, 6, 28),
        target_end_date=date(2026, 6, 28),
        coordinate_tier_name="B",
        selectors=_temperature_selector(),
        cutoff_id=T1245_CUTOFF_ID,
    )
    nbmqmd = build_chunk(
        spec=model_spec_by_id("nbmqmd"),
        target_start_date=date(2026, 6, 28),
        target_end_date=date(2026, 6, 28),
        coordinate_tier_name="B",
        selectors=_temperature_selector(),
        cutoff_id=T1245_CUTOFF_ID,
    )
    assert gfs.cutoff_id == T1245_CUTOFF_ID
    assert gfs.request_payload["asOf"] == "2026-06-28T08:45:00Z"
    assert nbmqmd.request_payload["asOf"] == "2026-06-28T11:00:00Z"
    assert nbmqmd.request_payload["timesList"] == ["2026-06-29T06:00:00Z"]


def test_ensemble_chunk_includes_all_members_and_expected_credits() -> None:
    chunk = build_chunk(
        spec=model_spec_by_id("ifsenfo"),
        target_start_date=date(2026, 6, 28),
        target_end_date=date(2026, 6, 28),
        coordinate_tier_name="B",
        selectors=_temperature_selector(),
    )
    assert chunk.request_payload["asOf"] == "2026-06-27T17:45:00Z"
    assert chunk.request_payload["members"] == list(range(51))
    assert chunk.estimated_credits == 102


def test_request_hash_includes_dataset_path_not_only_body_and_cutoff_profile() -> None:
    selectors = _temperature_selector()
    gefs = build_chunk(
        spec=model_spec_by_id("gefsatmos"),
        target_start_date=date(2026, 6, 28),
        target_end_date=date(2026, 6, 28),
        coordinate_tier_name="B",
        selectors=selectors,
    )
    aigefs = build_chunk(
        spec=model_spec_by_id("aigefssfc"),
        target_start_date=date(2026, 6, 28),
        target_end_date=date(2026, 6, 28),
        coordinate_tier_name="B",
        selectors=selectors,
    )
    gefs_t1245 = build_chunk(
        spec=model_spec_by_id("gefsatmos"),
        target_start_date=date(2026, 6, 28),
        target_end_date=date(2026, 6, 28),
        coordinate_tier_name="B",
        selectors=selectors,
        cutoff_id=T1245_CUTOFF_ID,
    )
    assert gefs.request_payload == aigefs.request_payload
    assert gefs.request_sha256 != aigefs.request_sha256
    assert gefs.request_sha256 != gefs_t1245.request_sha256


def test_t1245_runs_chunk_uses_model_run_times_and_filters_synoptic_leads() -> None:
    gfs_run = runs_model_time_for_target(
        model_spec_by_id("gfs"),
        date(2026, 6, 28),
        cutoff_id=T1245_CUTOFF_ID,
    )
    rap_run = runs_model_time_for_target(
        model_spec_by_id("rap"),
        date(2026, 6, 28),
        cutoff_id=T1245_CUTOFF_ID,
    )
    assert gfs_run.isoformat() == "2026-06-28T06:00:00+00:00"
    assert rap_run.isoformat() == "2026-06-28T08:00:00+00:00"

    chunk = build_runs_chunk(
        spec=model_spec_by_id("gefsatmosmean"),
        target_start_date=date(2026, 6, 28),
        target_end_date=date(2026, 6, 30),
        coordinate_tier_name="B",
        selectors=_temperature_selector(),
        cutoff_id=T1245_CUTOFF_ID,
        group_index=0,
    )
    assert chunk.endpoint_type == "runs"
    assert chunk.request_payload["timesList"] == [
        "2026-06-28T06:00:00Z",
        "2026-06-29T06:00:00Z",
        "2026-06-30T06:00:00Z",
    ]
    assert chunk.request_payload["minLeadTime"] == "12h"
    assert chunk.request_payload["maxLeadTime"] == "12h"
    assert [item.isoformat() for item in chunk.valid_times_utc] == [
        "2026-06-28T18:00:00+00:00",
        "2026-06-29T18:00:00+00:00",
        "2026-06-30T18:00:00+00:00",
    ]
    assert len(chunk.expected_run_valid_pairs_utc) == 3
