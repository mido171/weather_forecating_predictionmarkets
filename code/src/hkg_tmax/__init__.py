"""HKG Tmax leakage-safe research infrastructure."""

from .settlement import Bucket, BucketSet
from .timeutils import HONG_KONG_TZ, asof_eligible, cutoff_for_local_date

__all__ = [
    "Bucket",
    "BucketSet",
    "HONG_KONG_TZ",
    "asof_eligible",
    "cutoff_for_local_date",
]

__version__ = "0.1.0"
