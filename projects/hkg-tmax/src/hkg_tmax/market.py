from __future__ import annotations

import re
from pathlib import Path

from .fetch import FetchPolicy, fetch_and_archive
from .hashing import sha256_text
from .paths import ProjectPaths
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
    source_id = f"polymarket_event_{sha256_text(slug)[:16]}"
    return fetch_and_archive(
        url=url,
        source_id=source_id,
        raw_root=ProjectPaths.from_project_root(root).data_root / "raw",
        policy=policy,
    )
