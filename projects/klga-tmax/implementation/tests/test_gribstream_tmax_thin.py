from __future__ import annotations

from datetime import date, datetime, timezone

from klga_tmax.providers.gribstream.features import build_tmax_thin_gold_features
from klga_tmax.providers.gribstream.models import GribStreamParsedValue, ResolvedSelector
from klga_tmax.providers.gribstream.plan import (
    T1245_CUTOFF_ID,
    TMAX_THIN_FEATURE_PROFILE,
    TMAX_THIN_MODEL_SPECS,
    TMAX_THIN_PERSISTENCE_MODE,
    build_runs_chunk,
    build_tmax_thin_runs_job_plan,
    tmax_thin_model_spec_by_id,
    tmax_thin_spec_summary_rows,
    valid_times_for_target,
)


def _selectors(count: int, prefix: str = "temperature_2m") -> tuple[ResolvedSelector, ...]:
    return tuple(
        ResolvedSelector(
            alias=prefix if count == 1 else f"{prefix}_{idx}",
            request_variables=(
                {
                    "name": "TMP",
                    "level": "2 m above ground",
                    "info": "",
                    "alias": prefix if count == 1 else f"{prefix}_{idx}",
                },
            ),
            variable_name="TMP",
            variable_level="2 m above ground",
            variable_info="",
            unit_hint="K",
        )
        for idx in range(count)
    )


def _thin_selector_map() -> dict[str, tuple[ResolvedSelector, ...]]:
    selectors: dict[str, tuple[ResolvedSelector, ...]] = {}
    for spec in TMAX_THIN_MODEL_SPECS:
        if spec.model_id == "rtma":
            selectors[spec.model_id] = _selectors(3, "rtma_var")
        elif spec.model_id == "nbmqmd":
            selectors[spec.model_id] = _selectors(21, "tmp_max18_p")
        else:
            selectors[spec.model_id] = _selectors(1)
    return selectors


def _value(
    *,
    model_id: str = "gfs",
    target_date: date = date(2026, 6, 28),
    grid_point_id: str = "GP_KLGA_EXACT",
    valid_hour: int = 18,
    member: str = "deterministic",
    alias: str = "temperature_2m",
    kelvin: float = 300.15,
) -> GribStreamParsedValue:
    run_time = datetime(2026, 6, 28, 6, tzinfo=timezone.utc)
    valid_time = datetime(2026, 6, 28, valid_hour, tzinfo=timezone.utc)
    cutoff = datetime(2026, 6, 28, 12, 45, tzinfo=timezone.utc)
    return GribStreamParsedValue(
        model_id=model_id,
        endpoint_type="runs",
        target_date=target_date,
        cutoff_id=T1245_CUTOFF_ID,
        cutoff_utc=cutoff,
        as_of_utc=datetime(2026, 6, 28, 8, 45, tzinfo=timezone.utc),
        coordinate_tier="KLGA",
        grid_point_id=grid_point_id,
        lat=40.77945,
        lon=-73.88027,
        forecasted_at_utc=run_time,
        forecasted_time_utc=valid_time,
        forecast_hour=(valid_time - run_time).total_seconds() / 3600.0,
        member=member,
        variable_alias=alias,
        variable_name="TMP",
        variable_level="2 m above ground",
        variable_info="",
        unit_original="K",
        value_original=kelvin,
        unit_canonical="K",
        value_canonical=kelvin,
        index_updated_at_utc=None,
        provider_available_at_utc=datetime(2026, 6, 28, 8, 45, tzinfo=timezone.utc),
        effective_available_at_utc=datetime(2026, 6, 28, 8, 45, tzinfo=timezone.utc),
        availability_method="conservative_lag_rule",
        raw_row_hash=f"{model_id}-{grid_point_id}-{valid_hour}-{member}-{alias}-{kelvin}",
        raw_row_json={},
    )


def test_tmax_thin_summary_excludes_urma_and_matches_credit_budget() -> None:
    rows = tmax_thin_spec_summary_rows(end_date=date(2026, 6, 28))
    assert len(rows) == 14
    assert "urma" not in {row["model_id"] for row in rows}
    assert sum(int(row["expected_total_credits"]) for row in rows) == 400_750
    nbm = next(row for row in rows if row["model_id"] == "nbm")
    hrrr = next(row for row in rows if row["model_id"] == "hrrr")
    assert nbm["fetch_shape"] == "hourly_peak"
    assert nbm["coordinate_tier"] == "B"
    assert hrrr["coordinate_tier"] == "B"


def test_tmax_thin_plan_uses_mixed_tiers_and_profile_hashing() -> None:
    plan = build_tmax_thin_runs_job_plan(
        job_id="thin_test",
        end_date=date(2026, 6, 28),
        selectors_by_model=_thin_selector_map(),
    )
    assert plan.coordinate_tier == "MIXED_TMAX_THIN"
    assert sum(chunk.estimated_credits for chunk in plan.chunks) == 404_160
    assert {chunk.feature_profile for chunk in plan.chunks} == {TMAX_THIN_FEATURE_PROFILE}
    assert {chunk.persistence_mode for chunk in plan.chunks} == {TMAX_THIN_PERSISTENCE_MODE}
    assert any(chunk.model_id == "hrrr" and chunk.coordinate_tier == "B" for chunk in plan.chunks)
    assert any(chunk.model_id == "gefsatmos" and chunk.coordinate_tier == "KLGA" for chunk in plan.chunks)


def test_nbm_fallback_uses_peak_temperature_curve() -> None:
    spec = tmax_thin_model_spec_by_id("nbm")
    valid_times = valid_times_for_target(spec, date(2026, 6, 28), cutoff_id=T1245_CUTOFF_ID)
    assert len(valid_times) == 10
    assert valid_times[0].isoformat() == "2026-06-28T16:00:00+00:00"
    assert valid_times[-1].isoformat() == "2026-06-29T01:00:00+00:00"


def test_thin_request_hash_changes_with_feature_profile_even_when_payload_matches() -> None:
    spec = tmax_thin_model_spec_by_id("gfs")
    selectors = _selectors(1)
    broad_like = build_runs_chunk(
        spec=spec,
        target_start_date=date(2026, 6, 28),
        target_end_date=date(2026, 6, 28),
        coordinate_tier_name="B",
        selectors=selectors,
        cutoff_id=T1245_CUTOFF_ID,
    )
    thin = build_runs_chunk(
        spec=spec,
        target_start_date=date(2026, 6, 28),
        target_end_date=date(2026, 6, 28),
        coordinate_tier_name="B",
        selectors=selectors,
        cutoff_id=T1245_CUTOFF_ID,
        feature_profile=TMAX_THIN_FEATURE_PROFILE,
        persistence_mode=TMAX_THIN_PERSISTENCE_MODE,
    )
    assert broad_like.request_payload == thin.request_payload
    assert broad_like.request_sha256 != thin.request_sha256


def test_tmax_thin_feature_builder_creates_deterministic_and_ensemble_features() -> None:
    deterministic = (
        _value(model_id="gfs", valid_hour=16, kelvin=298.15),
        _value(model_id="gfs", valid_hour=17, kelvin=301.15),
        _value(model_id="gfs", valid_hour=18, kelvin=300.15),
    )
    deterministic_features = build_tmax_thin_gold_features(deterministic)
    names = {feature.feature_name for feature in deterministic_features}
    assert "grib_gfs_klga_core_peak_window_max_tmp_2m_f" in names
    peak = next(feature for feature in deterministic_features if feature.feature_name == "grib_gfs_klga_core_peak_window_max_tmp_2m_f")
    assert round(float(peak.feature_value), 2) == 82.40

    ensemble = tuple(
        _value(model_id="gefsatmos", member=str(member), valid_hour=18, kelvin=298.15 + member)
        for member in range(3)
    )
    ensemble_features = build_tmax_thin_gold_features(ensemble)
    ensemble_names = {feature.feature_name for feature in ensemble_features}
    assert "grib_gefsatmos_klga_core_valid_18z_member_count" in ensemble_names
    assert "grib_gefsatmos_klga_core_valid_18z_prob_tmp_2m_ge_80f" in ensemble_names
    assert not any(name == "grib_gefsatmos_klga_core_member_count" for name in ensemble_names)

    mean_row_with_member_zero = (
        _value(model_id="gefsatmosmean", member="0", valid_hour=18, kelvin=300.15),
    )
    mean_features = build_tmax_thin_gold_features(mean_row_with_member_zero)
    mean_names = {feature.feature_name for feature in mean_features}
    assert "grib_gefsatmosmean_klga_core_valid_18z_tmp_2m_f" in mean_names
    assert not any("member_count" in name for name in mean_names)
