from __future__ import annotations

from datetime import date

import pytest

from klga_tmax.features.leakage import LabelLeakageViolation, assert_daily_high_label_history_safe


def test_daily_high_error_features_may_use_labels_only_through_t_minus_2() -> None:
    assert_daily_high_label_history_safe(
        target_date=date(2026, 6, 28),
        label_dates_used=[date(2026, 6, 25), date(2026, 6, 26)],
        feature_name="actual_klga_high_error_mean_3d_f",
    )


def test_daily_high_error_features_reject_t_minus_1_label() -> None:
    with pytest.raises(LabelLeakageViolation):
        assert_daily_high_label_history_safe(
            target_date=date(2026, 6, 28),
            label_dates_used=[date(2026, 6, 27)],
            feature_name="actual_klga_high_error_mean_3d_f",
        )


def test_daily_high_error_features_reject_target_day_label() -> None:
    with pytest.raises(LabelLeakageViolation):
        assert_daily_high_label_history_safe(
            target_date=date(2026, 6, 28),
            label_dates_used=[date(2026, 6, 28)],
            feature_name="actual_klga_high_error_mean_3d_f",
        )
