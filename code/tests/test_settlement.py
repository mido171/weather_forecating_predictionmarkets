from decimal import Decimal

import pytest

from hkg_tmax.settlement import BucketSet, SettlementError, load_bucket_set, rules_hash


def test_integer_range_fixture_maps_decimal_without_rounding(repo_root) -> None:
    buckets = load_bucket_set(repo_root / "config" / "example_market_buckets.yaml")
    assert buckets.winner(Decimal("30.0")).label == "30°C"
    assert buckets.winner(Decimal("30.5")).label == "30°C"
    assert buckets.winner(Decimal("30.9")).label == "30°C"
    assert buckets.winner(Decimal("31.0")).label == "31°C"
    assert buckets.winner(Decimal("25.9")).label == "25°C or below"
    assert buckets.winner(Decimal("35.0")).label == "35°C or higher"


def test_overlap_is_rejected() -> None:
    with pytest.raises(SettlementError, match="Overlap"):
        BucketSet.from_mappings(
            [
                {"label": "low", "lower_inclusive": None, "upper_exclusive": 31},
                {"label": "high", "lower_inclusive": 30, "upper_exclusive": None},
            ]
        )


def test_gap_is_rejected() -> None:
    with pytest.raises(SettlementError, match="Gap"):
        BucketSet.from_mappings(
            [
                {"label": "low", "lower_inclusive": None, "upper_exclusive": 30},
                {"label": "high", "lower_inclusive": 31, "upper_exclusive": None},
            ]
        )


def test_rules_hash_normalizes_whitespace() -> None:
    assert rules_hash("a  b\nc") == rules_hash("a b c")
    assert rules_hash("a  b", normalized=False) != rules_hash("a b", normalized=False)
