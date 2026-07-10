from __future__ import annotations

from datetime import date, timedelta

from hkg_t24.features.snapshot_builder import build_target_memory_features


def _labels(count: int = 120) -> list[tuple[date, float]]:
    start = date(2020, 1, 1)
    return [(start + timedelta(days=offset), float(20 + (offset % 15))) for offset in range(count)]


def test_synthetic_120_label_target_memory_counts_use_lag2_not_lag1() -> None:
    features = build_target_memory_features(_labels())

    def count_with_value(feature_name: str) -> int:
        return sum(1 for values in features.values() if values.get(feature_name) is not None)

    assert count_with_value("target__lag2_tmax_c") == 118
    assert count_with_value("target__lag3_tmax_c") == 117
    assert count_with_value("target__lag7_tmax_c") == 113
    assert count_with_value("target__lag14_tmax_c") == 106
    assert count_with_value("target__lag30_tmax_c") == 90
    assert count_with_value("target__lag60_tmax_c") == 60
    assert count_with_value("target__roll7_mean_lag2_c") == 112
    assert count_with_value("target__roll14_mean_lag2_c") == 105
    assert count_with_value("target__roll30_mean_lag2_c") == 89
    assert count_with_value("target__slope7_lag2_c_per_day") == 112
    assert count_with_value("target__slope30_lag2_c_per_day") == 89
    assert count_with_value("target__hot_spell_length_lag2_days") == 118
    assert all("target__lag1_tmax_c" not in values for values in features.values())
    assert all(
        "target__lag2_tmax_c__is_missing" in values
        and "target__year_index__is_missing" not in values
        for values in features.values()
    )
