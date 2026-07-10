from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_hkg_t24_nonlinear_local_residual_fusion_lab import (
    DELTA_GRID,
    LocalFusionSpec,
    apply_fixed_delta_spec,
    apply_prior_delta_spec,
    forecast_range_bucket,
    leakage_audit,
    local_group_key,
    rh_bucket,
    select_prior_delta,
    weather_bucket,
    wind_bucket,
)


def make_spec(**overrides: object) -> LocalFusionSpec:
    values = {
        "candidate_id": "test",
        "mode": "prior_best_delta",
        "candidate_class": "causal_prior_delta_selector",
        "group_mode": "global",
        "min_history": 1,
        "fallback_delta": 0.0,
        "temperature_c": 0.0,
        "fixed_delta": 0.0,
        "cap_low": 0.0,
        "cap_high": 0.50,
    }
    values.update(overrides)
    return LocalFusionSpec(**values)  # type: ignore[arg-type]


def base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "target_tmax_c": [24.0, 30.0],
            "official_family_prediction_c": [20.0, 20.0],
            "station_family_prediction_c": [24.0, 30.0],
            "base_0069_prediction_c": [20.0, 20.0],
            "base_0069_station_weight": [0.0, 0.0],
            "forecast_source_family": ["rss_archive", "rss_archive"],
            "family_disagreement_c": [4.0, 10.0],
            "abs_family_disagreement_c": [4.0, 10.0],
            "active_member_count": [3, 3],
            "fold_id": ["fold", "fold"],
            "signeddiff_bucket": ["station_warmer_ge_1c", "station_warmer_ge_1c"],
            "absdiff_bucket": ["absdiff_gt_2p50", "absdiff_gt_2p50"],
            "active_count_bucket": ["station_stack_three_plus_members", "station_stack_three_plus_members"],
            "forecast_range_bucket": ["range_le_3c", "range_le_3c"],
            "forecast_max_bucket": ["level_20_24c", "level_28_32c"],
            "forecast_midpoint_bucket": ["level_20_24c", "level_24_28c"],
            "rh_bucket": ["rh_medium", "rh_medium"],
            "weather_bucket": ["weather_rain", "weather_rain"],
            "wind_bucket": ["wind_east_normal", "wind_east_normal"],
            "gate_weight_bucket": ["gate_medium", "gate_medium"],
            "station_correction_bucket": ["stackcorr_neutral", "stackcorr_neutral"],
            "selected_family": ["router", "router"],
        }
    )


def test_forecast_and_text_buckets_are_stable() -> None:
    assert forecast_range_bucket(3.0) == "range_le_3c"
    assert forecast_range_bucket(4.5) == "range_4_5c"
    assert rh_bucket(60, 95) == "rh_medium"
    assert weather_bucket("Cloudy with showers") == "weather_rain"
    assert wind_bucket("East force 4 to 5") == "wind_east_normal"
    assert wind_bucket("Northeast force 6 offshore") == "wind_north_east_strong"


def test_local_group_key_combines_pre_target_trigger_features() -> None:
    key = local_group_key(base_frame().iloc[0], "source_signeddiff_weather_active")

    assert key == "rss_archive|station_warmer_ge_1c|weather_rain|station_stack_three_plus_members"


def test_prior_delta_selector_excludes_current_row() -> None:
    spec = make_spec(group_mode="global", min_history=1)

    out = apply_prior_delta_spec(base_frame(), spec)

    assert out["station_delta"].iloc[0] == 0.0
    assert out["station_delta"].iloc[1] == max(DELTA_GRID)
    assert out["prior_count"].iloc[1] == 1


def test_prior_soft_delta_stays_inside_grid() -> None:
    spec = make_spec(mode="prior_soft_delta", temperature_c=0.05)
    abs_sums = np.linspace(10.0, 1.0, len(DELTA_GRID))

    delta = select_prior_delta(abs_sums=abs_sums, count=10, spec=spec)

    assert min(DELTA_GRID) <= delta <= max(DELTA_GRID)


def test_fixed_delta_spec_clips_station_weight() -> None:
    frame = base_frame()
    frame["base_0069_station_weight"] = [0.48, 0.02]
    spec = make_spec(mode="fixed_delta", fixed_delta=0.08)

    out = apply_fixed_delta_spec(frame, spec)

    assert out["station_weight"].iloc[0] == 0.50
    assert out["station_weight"].iloc[1] == 0.10


def test_leakage_audit_keeps_fixed_deltas_non_deployable() -> None:
    frame = base_frame()
    frame["forecast_range_c"] = [3.0, 3.0]
    scoreboard = pd.DataFrame(
        {
            "candidate_class": ["diagnostic_fixed_local_delta", "causal_prior_delta_selector"],
            "deployable_gate_passed": [False, True],
            "delta_mae_vs_0069": [-0.01, -0.01],
            "fold_delta_max_vs_0069": [0.0, 0.0],
            "late_delta_mae_vs_0069": [0.0, 0.0],
        }
    )

    audit = leakage_audit(frame, scoreboard)

    assert audit["passed"].astype(bool).all()
