from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_0102_timestamp_proof_unlock_queue import (
    audit_feature_unlock_queue,
    family_source_status,
    matching_time_proof_columns,
    source_unlock_decision,
    truthy_count,
)


def test_matching_time_proof_columns_excludes_valid_and_retrieved_times() -> None:
    columns = [
        "valid_at_utc",
        "raw_retrieved_at_utc",
        "issued_at_utc",
        "first_available_at_utc",
        "local_date",
    ]

    assert matching_time_proof_columns(columns) == ["first_available_at_utc", "issued_at_utc"]


def test_truthy_count_handles_string_and_boolean_flags() -> None:
    assert truthy_count(pd.Series([True, False, None])) == 1
    assert truthy_count(pd.Series(["true", "False", "1", "no", "yes"])) == 3


def test_upper_air_stays_blocked_without_available_at_or_release_latency() -> None:
    decision = source_unlock_decision(
        source_family="upper_air",
        rows=10,
        provider_time_proof_columns=[],
        operational_input_allowed_true_rows=0,
        release_latency_proven_true_rows=0,
    )

    assert decision["unlock_decision"] is False
    assert decision["proof_status"] == "blocked_missing_upper_air_available_at_or_release_latency"
    assert "release-latency" in str(decision["required_next_evidence"])


def test_upper_air_can_unlock_with_all_row_operational_and_available_at_proof() -> None:
    decision = source_unlock_decision(
        source_family="upper_air",
        rows=10,
        provider_time_proof_columns=["available_at_utc"],
        operational_input_allowed_true_rows=10,
        release_latency_proven_true_rows=0,
    )

    assert decision["unlock_decision"] is True
    assert decision["proof_status"] == "provider_available_at_or_release_latency_proven"


def test_hko_daily_stays_blocked_without_publication_timestamp() -> None:
    decision = source_unlock_decision(
        source_family="hko_daily_climate",
        rows=10,
        provider_time_proof_columns=[],
        operational_input_allowed_true_rows=10,
        release_latency_proven_true_rows=0,
    )

    assert decision["unlock_decision"] is False
    assert decision["proof_status"] == "blocked_missing_daily_publication_timestamp"
    assert "first-publication" in str(decision["required_next_evidence"])


def test_audit_feature_unlock_queue_keeps_blocked_families_diagnostic_only() -> None:
    atlas = pd.DataFrame(
        [
            {
                "feature": "igra_hgt_1000hpa_m",
                "family": "upper_air",
                "diagnostic_score": 3.0,
                "timestamp_audit_status": "timestamp_audit_required",
                "allowed_for_future_walkforward": False,
                "cutoff_rule": "must prove",
                "required_proof_before_model": "available_at",
                "blocker": "missing",
            },
            {
                "feature": "daily_north_point_sea_temperature_am_lag7",
                "family": "marine_proxy",
                "diagnostic_score": 2.0,
                "timestamp_audit_status": "publication_lag_audit_required",
                "allowed_for_future_walkforward": False,
                "cutoff_rule": "must prove",
                "required_proof_before_model": "published_at",
                "blocker": "missing",
            },
        ]
    )
    source_evidence = pd.DataFrame(
        [
            {
                "source_family": "upper_air",
                "source_id": "igra",
                "status": "present",
                "rows": 10,
                "provider_time_proof_columns": "",
                "operational_input_allowed_true_rows": 0,
                "release_latency_proven_true_rows": 0,
                "proof_status": "blocked_missing_upper_air_available_at_or_release_latency",
                "unlock_decision": False,
                "required_next_evidence": "missing proof",
                "source_time_policy_values": "valid time only",
            },
            {
                "source_family": "marine_proxy",
                "source_id": "hko_daily",
                "status": "present",
                "rows": 10,
                "provider_time_proof_columns": "",
                "operational_input_allowed_true_rows": 0,
                "release_latency_proven_true_rows": 0,
                "proof_status": "blocked_missing_daily_publication_timestamp",
                "unlock_decision": False,
                "required_next_evidence": "missing publication",
                "source_time_policy_values": "finalized table",
            },
        ]
    )

    audited = audit_feature_unlock_queue(atlas, family_source_status(source_evidence))

    assert audited["post_0102_allowed_for_future_walkforward"].tolist() == [False, False]
    assert audited["post_0102_status"].tolist() == ["still_diagnostic_only", "still_diagnostic_only"]
