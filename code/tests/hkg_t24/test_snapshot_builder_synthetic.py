from __future__ import annotations

from datetime import date, timedelta

from hkg_t24.features.snapshot_builder import build_target_memory_features


def _labels(count: int = 120) -> list[tuple[date, float]]:
    start = date(2020, 1, 1)
    return [(start + timedelta(days=offset), float(20 + (offset % 15))) for offset in range(count)]


def test_synthetic_120_label_target_memory_counts_use_lag2_not_lag1() -> None:
    features = build_target_memory_features(_labels())

    def count_with(feature_name: str) -> int:
        return sum(1 for values in features.values() if feature_name in values)

    assert count_with("target__lag2_tmax_c") == 118
    assert count_with("target__lag3_tmax_c") == 117
    assert count_with("target__lag7_tmax_c") == 113
    assert count_with("target__roll7_lag2_mean_tmax_c") == 112
    assert count_with("target__roll14_lag2_mean_tmax_c") == 105
    assert count_with("target__roll30_lag2_mean_tmax_c") == 89
    assert count_with("target__slope7_lag2_tmax_c") == 112
    assert count_with("target__slope30_lag2_tmax_c") == 89
    assert count_with("target__hot_spell_lag2") == 118
    assert all("target__lag1_tmax_c" not in values for values in features.values())
