from __future__ import annotations

import re
from pathlib import Path

from .fetch import FetchPolicy, fetch_and_archive
from .storage import RawSnapshot


class MarketError(ValueError):
    """Raised for unsafe or invalid market identifiers."""


_SAFE_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def snapshot_polymarket_event(
    root: Path,
    slug: str,
    policy: FetchPolicy | None = None,
) -> RawSnapshot:
    if not _SAFE_SLUG.fullmatch(slug):
        raise MarketError(f"Unsafe/invalid Polymarket event slug: {slug!r}")
    url = f"https://gamma-api.polymarket.com/events/slug/{slug}"
    return fetch_and_archive(
        url=url,
        source_id=f"polymarket_event_{slug}",
        raw_root=root / "data" / "raw",
        policy=policy,
    )
