from decimal import Decimal

import pytest

from hkg_tmax.distribution import DiscreteTemperatureDistribution, DistributionError
from hkg_tmax.settlement import load_bucket_set


def test_decimal_distribution_maps_to_buckets(repo_root) -> None:
    distribution = DiscreteTemperatureDistribution.from_mapping(
        {"30.9": 0.4, "31.0": 0.6}
    )
    buckets = load_bucket_set(repo_root / "config" / "project" / "example_market_buckets.yaml")
    probabilities = distribution.bucket_probabilities(buckets)
    assert probabilities["30°C"] == 0.4
    assert probabilities["31°C"] == 0.6
    assert distribution.mean == Decimal("30.96")


def test_mass_must_sum_to_one() -> None:
    with pytest.raises(DistributionError):
        DiscreteTemperatureDistribution.from_mapping({"30.0": 0.4, "31.0": 0.4})
