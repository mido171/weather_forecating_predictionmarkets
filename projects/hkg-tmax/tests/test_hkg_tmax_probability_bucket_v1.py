from __future__ import annotations

import numpy as np
import pandas as pd
from hkg_tmax_probability.bucket_rules import (
    BUCKET_KEYS,
    bucket_index,
    bucket_key,
    normalize_probability_matrix,
)
from hkg_tmax_probability.cdf_calibration import monotone_cdf_projection
from hkg_tmax_probability.forecast_selection import CutoffProfile, select_latest_eligible_forecasts
from hkg_tmax_probability.label_publication_audit import apply_first_publication_labels
from hkg_tmax_probability.leakage_audit import audit_live_output, audit_modeling_table
from hkg_tmax_probability.models import optimize_stack_weights, residual_pmf_to_bucket_probs
from hkg_tmax_probability.scoring import ranked_probability_score
from hkg_tmax_probability.validation import SplitWindow, train_validation_frames


def test_decimal_bucket_boundaries() -> None:
    assert bucket_key("24.9") == "24_or_below"
    assert bucket_key("25.0") == "25"
    assert bucket_key("25.9") == "25"
    assert bucket_key("31.9") == "31"
    assert bucket_key("32.0") == "32"
    assert bucket_key("33.9") == "33"
    assert bucket_key("34.0") == "34_or_higher"
    assert bucket_index("31.9") == BUCKET_KEYS.index("31")


def test_no_post_cutoff_rows_and_deterministic_tiebreak() -> None:
    forecasts = pd.DataFrame(
        [
            {
                "target_date": "2026-07-06",
                "issue_at_utc": "2026-07-05T10:00:00Z",
                "snapshot_at_utc": "2026-07-05T10:05:00Z",
                "ingested_at_utc": "2026-07-05T10:07:00Z",
                "source_archive_mtime_utc": "2026-07-05T10:08:00Z",
                "raw_sha256": "a",
                "bulletin_id": "a",
                "forecast_min_c": 27,
                "forecast_max_c": 32,
                "forecast_range_c": 5,
            },
            {
                "target_date": "2026-07-06",
                "issue_at_utc": "2026-07-05T10:00:00Z",
                "snapshot_at_utc": "2026-07-05T10:06:00Z",
                "ingested_at_utc": "2026-07-05T10:07:00Z",
                "source_archive_mtime_utc": "2026-07-05T10:08:00Z",
                "raw_sha256": "b",
                "bulletin_id": "b",
                "forecast_min_c": 28,
                "forecast_max_c": 33,
                "forecast_range_c": 5,
            },
            {
                "target_date": "2026-07-06",
                "issue_at_utc": "2026-07-05T16:30:00Z",
                "snapshot_at_utc": "2026-07-05T16:30:00Z",
                "ingested_at_utc": "2026-07-05T16:30:00Z",
                "source_archive_mtime_utc": "2026-07-05T16:30:00Z",
                "raw_sha256": "late",
                "bulletin_id": "late",
                "forecast_min_c": 29,
                "forecast_max_c": 34,
                "forecast_range_c": 5,
            },
        ]
    )
    selected, eligible = select_latest_eligible_forecasts(
        forecasts,
        pd.Series(pd.to_datetime(["2026-07-06"])),
        [CutoffProfile("primary", "23:59", True)],
    )
    assert (pd.to_datetime(eligible["issue_at_utc"], utc=True) <= pd.to_datetime(eligible["cutoff_at_utc"], utc=True)).all()
    assert selected.iloc[0]["bulletin_id"] == "b"


def test_leakage_audit_forbidden_predictors_and_live_trading_fields() -> None:
    modeling = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2022-01-01"]),
            "cutoff_profile": ["primary"],
            "issue_at_utc": pd.to_datetime(["2021-12-31T12:00:00Z"]),
            "cutoff_at_utc": pd.to_datetime(["2021-12-31T15:59:00Z"]),
            "target_table": ["canonical_core"],
        }
    )
    audit = audit_modeling_table(modeling, predictor_columns=["forecast_max_c", "target_tmax_c"])
    assert audit["status"] == "fail"
    assert "target_tmax_c" in audit["violations"]["forbidden_predictor_columns"]
    assert audit_live_output({"bucket_probabilities": {}, "ev": 0.2})["status"] == "fail"
    assert audit_live_output({"bucket_probabilities": {}, "scope": "weather_probability_only"})["status"] == "pass"


def test_sealed_rows_not_used_for_sealed_training() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2023-12-31", "2024-01-01", "2024-01-02"]),
            "target_table": ["canonical_core", "sealed_confirmation", "sealed_confirmation"],
            "is_primary_cutoff": [True, True, True],
        }
    )
    train, validation = train_validation_frames(
        frame,
        SplitWindow("sealed", "2024-12-31", "2024-01-01", "2024-12-31", sealed=True),
        primary_only=True,
    )
    assert (train["target_table"] != "sealed_confirmation").all()
    assert len(validation) == 2


def test_pmf_rows_sum_to_one() -> None:
    residual_pmf = np.zeros(241)
    residual_pmf[120] = 0.5
    residual_pmf[130] = 0.5
    probs = residual_pmf_to_bucket_probs(np.array([310, 320]), residual_pmf)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0).all()


def test_rps_orders_better_distribution_lower() -> None:
    truth = [7]
    good = normalize_probability_matrix(np.array([[0, 0, 0, 0, 0, 0, 0.05, 0.9, 0.05, 0, 0]], dtype=float))
    bad = normalize_probability_matrix(np.array([[0.9, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float))
    assert ranked_probability_score(good, truth)[0] < ranked_probability_score(bad, truth)[0]


def test_monotone_cdf_calibration_projection() -> None:
    cdf = np.array([[0.2, 0.4, 0.3, 0.8], [0.9, 0.1, 0.2, 0.7]])
    projected = monotone_cdf_projection(cdf)
    assert np.all(np.diff(projected, axis=1) >= -1e-12)
    assert np.all((projected >= 0) & (projected <= 1))


def test_stack_weight_constraints() -> None:
    y = np.array([0, 1, 2, 3])
    blocks = [
        normalize_probability_matrix(np.eye(11)[[0, 1, 2, 3]] + 1e-6),
        normalize_probability_matrix(np.ones((4, 11))),
        normalize_probability_matrix(np.eye(11)[[1, 2, 3, 4]] + 1e-6),
        normalize_probability_matrix(np.eye(11)[[0, 1, 2, 3]] + 1e-6),
        normalize_probability_matrix(np.eye(11)[[0, 1, 2, 3]] + 1e-6),
    ]
    weights = optimize_stack_weights(blocks, y)
    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()
    assert weights[4] >= 0.15


def test_label_publication_bucket_change_audit_application() -> None:
    modeling = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2022-07-01"]),
            "bucket_key": ["31"],
            "bucket_index": [7],
        }
    )
    audit = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2022-07-01"]),
            "first_publication_tmax_c": [32.0],
            "first_publication_bucket_key": ["32"],
        }
    )
    out = apply_first_publication_labels(modeling, audit)
    assert out.iloc[0]["first_publication_bucket_key"] == "32"
    assert out.iloc[0]["first_publication_bucket_index"] == 8


def test_no_trading_fields_in_live_output_contract() -> None:
    payload = {
        "target_date": "2026-07-06",
        "method": "B4_hierarchical_residual_pmf",
        "bucket_probabilities": {"31": 0.5, "32": 0.5},
        "scope": "weather_probability_only",
    }
    assert audit_live_output(payload)["status"] == "pass"
