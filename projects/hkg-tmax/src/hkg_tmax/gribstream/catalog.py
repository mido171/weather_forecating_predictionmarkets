"""Live GribStream catalog selector resolution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable

import httpx


@dataclass(frozen=True)
class ResolvedSelector:
    dataset: str
    semantic_variable: str
    semantic_family: str
    native_name: str
    native_level: str
    native_info: str
    alias: str
    native_unit: str
    source_sha256: str
    retrieved_at_utc: str
    source_json: dict[str, Any]

    def as_request_variable(self) -> dict[str, str]:
        return {
            "name": self.native_name,
            "level": self.native_level,
            "info": self.native_info,
            "alias": self.alias,
        }


class SelectorResolutionError(RuntimeError):
    """Raised when the live catalog does not expose the required exact selector."""


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def stable_json_sha(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _walk_json(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_json(item)


def candidate_selectors(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Extract selector-like objects from the catalog response."""

    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in _walk_json(payload):
        name = item.get("name")
        level = item.get("level")
        if not isinstance(name, str) or not isinstance(level, str):
            continue
        info = item.get("info", "")
        if info is None:
            info = ""
        if not isinstance(info, str):
            continue
        key = (name, level, info)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"name": name, "level": level, "info": info})
    return rows


def select_exact_temperature_2m(
    payload: dict[str, Any],
    *,
    dataset: str = "gfs",
    alias: str = "temperature_2m",
    retrieved_at_utc: str | None = None,
) -> ResolvedSelector:
    """Resolve the HKG T06 GFS 2 m temperature selector, failing closed on drift."""

    for row in candidate_selectors(payload):
        if row["name"].upper() == "TMP" and row["level"] == "2 m above ground" and row["info"] == "":
            selector_source = {
                "dataset": dataset,
                "semantic_variable": "temperature_2m",
                "selector": row,
            }
            return ResolvedSelector(
                dataset=dataset,
                semantic_variable="temperature_2m",
                semantic_family="surface",
                native_name=row["name"],
                native_level=row["level"],
                native_info=row["info"],
                alias=alias,
                native_unit="K",
                source_sha256=stable_json_sha(selector_source),
                retrieved_at_utc=retrieved_at_utc or utc_now_iso(),
                source_json=selector_source,
            )
    available = sorted(
        f"{row['name']}|{row['level']}|{row['info']}" for row in candidate_selectors(payload)
    )
    raise SelectorResolutionError(
        "Live GribStream catalog did not expose exact GFS selector "
        "TMP / 2 m above ground / empty info. Available selector candidates: "
        + "; ".join(available[:25]),
    )


def fetch_shared_parameter(
    *,
    dataset: str,
    parameter: str,
    alias: str,
    base_url: str = "https://gribstream.com",
    http_client: httpx.Client | None = None,
) -> tuple[dict[str, Any], str]:
    """Fetch a public shared-parameter catalog payload."""

    owns_client = http_client is None
    client = http_client or httpx.Client(timeout=httpx.Timeout(20.0, read=60.0))
    try:
        url = f"{base_url.rstrip('/')}/api/v2/catalog/shared-parameters/{parameter}"
        response = client.get(url, params={"dataset": dataset, "alias": alias})
        response.raise_for_status()
        return response.json(), utc_now_iso()
    finally:
        if owns_client:
            client.close()


def resolve_temperature_2m_selector(
    *,
    dataset: str = "gfs",
    base_url: str = "https://gribstream.com",
    http_client: httpx.Client | None = None,
) -> ResolvedSelector:
    payload, retrieved_at_utc = fetch_shared_parameter(
        dataset=dataset,
        parameter="temperature_2m",
        alias="temperature_2m",
        base_url=base_url,
        http_client=http_client,
    )
    return select_exact_temperature_2m(
        payload,
        dataset=dataset,
        alias="temperature_2m",
        retrieved_at_utc=retrieved_at_utc,
    )
