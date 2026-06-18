from __future__ import annotations

import os
import time
from collections.abc import Iterable
from pathlib import Path

from .config import Source, SourceCatalog
from .fetch import FetchError, FetchPolicy, fetch_and_archive
from .storage import RawSnapshot


class SourceFetchError(RuntimeError):
    """Raised when one or more requested sources fail."""


def unresolved_template(url: str) -> bool:
    return "{" in url or "}" in url


def is_http_url(url: str) -> bool:
    return url.startswith("https://") or url.startswith("http://")


def fetch_sources(
    *,
    root: Path,
    catalog: SourceCatalog,
    source_ids: Iterable[str] | None = None,
    tag: str | None = None,
    continue_on_error: bool = False,
) -> tuple[list[RawSnapshot], list[str]]:
    selected = catalog.select(source_ids=source_ids, tag=tag)
    timeout = float(os.getenv("HKG_TMAX_HTTP_TIMEOUT_SECONDS", "60"))
    interval = float(os.getenv("HKG_TMAX_MIN_REQUEST_INTERVAL_SECONDS", "1.0"))
    user_agent = os.getenv(
        "HKG_TMAX_USER_AGENT",
        "HKG-Tmax-Research/0.1 (+set-contact-in-.env)",
    )
    policy = FetchPolicy(timeout_seconds=timeout, user_agent=user_agent)
    snapshots: list[RawSnapshot] = []
    errors: list[str] = []

    for index, source in enumerate(selected):
        try:
            snapshots.append(_fetch_one(root, source, policy))
        except (FetchError, ValueError) as exc:
            message = f"{source.id}: {exc}"
            if not continue_on_error:
                raise SourceFetchError(message) from exc
            errors.append(message)
        if index < len(selected) - 1 and interval > 0:
            time.sleep(interval)
    return snapshots, errors


def _fetch_one(root: Path, source: Source, policy: FetchPolicy) -> RawSnapshot:
    url = source.url
    if unresolved_template(url):
        raise ValueError(f"URL requires template parameters: {url}")
    if not is_http_url(url):
        raise ValueError(f"Unsupported non-HTTP source for snapshot fetch: {url}")
    return fetch_and_archive(
        url=url,
        source_id=source.id,
        raw_root=root / "data" / "raw",
        policy=policy,
    )



def write_source_inventory(root: Path, catalog: SourceCatalog) -> Path:
    """Render a human-readable source inventory from the authoritative YAML catalog."""
    lines = [
        "# Source Inventory",
        "",
        "Generated from `config/data_sources.yaml`. Endpoint implementation and source-contract "
        "status must still be verified individually.",
        "",
        "| ID | Provider | Priority | Point-in-time role | Research role | Cadence | Access |",
        "|---|---|---|---|---|---|---|",
    ]
    for source in catalog.sources:
        raw = source.raw
        access_method = source.access.get("method", "—")
        values = [
            source.id,
            source.provider,
            raw.get("priority", "—"),
            source.point_in_time_status,
            source.role,
            raw.get("cadence", "—"),
            access_method,
        ]
        escaped = [str(value).replace("|", r"\|").replace("\n", " ") for value in values]
        lines.append("| " + " | ".join(escaped) + " |")

    counts: dict[str, int] = {}
    for source in catalog.sources:
        counts[source.point_in_time_status] = counts.get(source.point_in_time_status, 0) + 1
    lines.extend(["", "## Counts by point-in-time role", ""])
    for role, count in sorted(counts.items()):
        lines.append(f"- **{role}:** {count}")
    lines.extend(
        [
            "",
            "## Required next action",
            "",
            "For each implemented source, create a source contract under `docs/source_contracts/` "
            "and verify its official endpoint, timestamps, cadence, revision policy, terms, and tests.",
            "",
        ]
    )
    path = root / "reports" / "source_inventory.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
