from __future__ import annotations

import math

import pandas as pd

from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import (
    add_past_doy_anomaly,
    apply_tertile_bins,
    classify_feature_family,
    quantile_edges_from_train,
    safe_corr,
    update_markdown_section,
)


def test_classify_feature_family_core_prefixes() -> None:
    assert classify_feature_family("isd_pressure_plane_lat_slope_hpa_per_deg") == "isd_station_network"
    assert classify_feature_family("igra_thickness_1000_500_m_change_48h") == "upper_air"
    assert classify_feature_family("ua_theta_e_850hpa_k") == "upper_air"
    assert classify_feature_family("daily_kings_park_evaporation_lag7") == "hko_daily_climate"
    assert classify_feature_family("target_roll365_mean_lag7_c") == "target_memory"
    assert classify_feature_family("day_of_year") == "calendar_climatology"


def test_add_past_doy_anomaly_uses_only_prior_same_doy() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(
                ["2000-01-01", "2001-01-01", "2002-01-01", "2003-01-01", "2000-01-02"]
            ),
            "target_tmax_c": [10.0, 12.0, 15.0, 11.0, 20.0],
            "day_of_year": [1, 1, 1, 1, 2],
        }
    )
    out = add_past_doy_anomaly(frame, min_past_years=2)
    jan1 = out[out["day_of_year"].eq(1)].sort_values("target_date").reset_index(drop=True)
    assert math.isnan(float(jan1.loc[0, "target_anomaly_vs_past_doy_c"]))
    assert math.isnan(float(jan1.loc[1, "target_anomaly_vs_past_doy_c"]))
    assert jan1.loc[2, "past_doy_mean_tmax_c"] == 11.0
    assert jan1.loc[2, "target_anomaly_vs_past_doy_c"] == 4.0
    assert jan1.loc[3, "past_doy_mean_tmax_c"] == (10.0 + 12.0 + 15.0) / 3.0


def test_quantile_bins_from_train_and_apply() -> None:
    series = pd.Series(range(3000), dtype=float)
    edges = quantile_edges_from_train(series)
    assert edges is not None
    binned = apply_tertile_bins(pd.Series([edges[0] - 1, (edges[0] + edges[1]) / 2, edges[1] + 1, None]), edges)
    assert binned.tolist()[:3] == ["low", "mid", "high"]
    assert pd.isna(binned.iloc[3])


def test_safe_corr_respects_min_rows() -> None:
    n_rows, corr = safe_corr(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), min_rows=3)
    assert n_rows == 2
    assert math.isnan(corr)
    n_rows, corr = safe_corr(pd.Series([1.0, 2.0, 3.0]), pd.Series([2.0, 4.0, 6.0]), min_rows=3)
    assert n_rows == 3
    assert corr == 1.0


def test_update_markdown_section_preserves_literal_backslashes(tmp_path) -> None:
    path = tmp_path / "MILESTONES.md"
    path.write_text("# Title\n\n## Existing\n\nold\n", encoding="utf-8")
    update_markdown_section(
        path,
        heading="Existing",
        section="Run `.\\.venv\\Scripts\\python.exe script.py`",
    )
    text = path.read_text(encoding="utf-8")
    assert r".\.venv\Scripts\python.exe" in text
