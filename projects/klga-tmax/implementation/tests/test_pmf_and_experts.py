from __future__ import annotations

import pytest

from klga_tmax.constants import TEMP_GRID_F
from klga_tmax.models.experts import EXPERTS, build_expert_forecasts, combine_experts
from klga_tmax.models.pmf import gaussian_pmf, summarize_pmf, validate_pmf


def test_gaussian_pmf_uses_full_whole_degree_grid() -> None:
    pmf = gaussian_pmf(75.0, 3.0)
    assert list(map(int, pmf.keys())) == list(TEMP_GRID_F)
    assert sum(pmf.values()) == pytest.approx(1.0)
    validate_pmf(pmf)
    summary = summarize_pmf(pmf)
    assert 71 <= summary.prediction_interval_low_f <= summary.median_tmax_f
    assert summary.median_tmax_f <= summary.prediction_interval_high_f <= 79


def test_all_required_experts_emit_valid_pmfs() -> None:
    features = {
        "climatology_wu_tmax_mean_31d_f": 78.0,
        "wu_history_tmax_mean_14d_f": 79.0,
        "wu_history_tmax_lag_2d_f": 82.0,
        "station_actuals_nearby_tmax_mean_lag2_f": 80.0,
        "mos_guidance_tmax_mean_f": 81.0,
        "mos_guidance_tmax_std_f": 2.0,
        "gribstream_tmax_mean_f": 83.0,
        "gribstream_tmax_std_f": 3.0,
        "grib_nbm_tmax_f": 82.0,
        "grib_hrrr_peak_window_max_tmp_f": 84.0,
        "grib_gefs_tmax_f": 80.5,
        "grib_aifsoper_tmax_f": 81.5,
        "obs_klga_latest_temp_f": 76.0,
        "risk_sea_breeze_final_score": 0.5,
        "risk_backdoor_front_final_score": 0.0,
        "risk_marine_layer_final_score": 0.5,
    }
    forecasts = build_expert_forecasts(features)
    assert len(forecasts) == len(EXPERTS) == 9
    assert {forecast.expert_name for forecast in forecasts} == {
        expert.name for expert in EXPERTS
    }
    for forecast in forecasts:
        validate_pmf(forecast.summary.pmf)
        assert forecast.summary.expected_tmax_f == pytest.approx(
            sum(int(temp) * probability for temp, probability in forecast.summary.pmf.items())
        )
        assert forecast.feature_hash
        assert forecast.diagnostics["fallback_used"] is False or forecast.status != "ok"


def test_combiner_outputs_normalized_final_pmf_and_weights() -> None:
    forecasts = build_expert_forecasts(
        {
            "mos_guidance_tmax_mean_f": 74.0,
            "gribstream_tmax_mean_f": 76.0,
            "climatology_wu_tmax_mean_31d_f": 75.0,
            "wu_history_tmax_mean_14d_f": 73.5,
            "obs_klga_latest_temp_f": 71.0,
        }
    )
    combined, weights, diagnostics = combine_experts(forecasts)
    validate_pmf(combined.pmf)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert set(weights) == {forecast.expert_name for forecast in forecasts}
    assert diagnostics["expert_count"] == 9
