from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_hex(payload: bytes | str) -> str:
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hashlib.sha256(data).hexdigest()


def payload_hash(payload: bytes | str | dict[str, Any] | list[Any]) -> str:
    if isinstance(payload, bytes | str):
        return sha256_hex(payload)
    return sha256_hex(canonical_json(payload))


def retrieved_bucket_utc(retrieved_at_utc: datetime, *, bucket_seconds: int = 60) -> datetime:
    if retrieved_at_utc.tzinfo is None or retrieved_at_utc.utcoffset() is None:
        raise ValueError("retrieved_at_utc must be timezone-aware")
    retrieved = retrieved_at_utc.astimezone(timezone.utc)
    bucket = int(retrieved.timestamp()) // bucket_seconds * bucket_seconds
    return datetime.fromtimestamp(bucket, tz=timezone.utc)


def source_request_id(
    *,
    source_name: str,
    source_endpoint: str,
    request_params: dict[str, Any],
    retrieved_at_utc: datetime,
    bucket_seconds: int = 60,
) -> str:
    bucket = retrieved_bucket_utc(retrieved_at_utc, bucket_seconds=bucket_seconds)
    raw = canonical_json(
        {
            "source_name": source_name,
            "source_endpoint": source_endpoint,
            "request_params": request_params,
            "retrieved_bucket_utc": bucket.isoformat(),
        }
    )
    return sha256_hex(raw)
