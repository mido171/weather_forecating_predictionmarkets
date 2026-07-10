from pathlib import Path
from types import SimpleNamespace

import pytest

from hkg_tmax import market
from hkg_tmax.hashing import sha256_text
from hkg_tmax.market import MarketError, snapshot_polymarket_event


def test_snapshot_polymarket_event_uses_short_hashed_source_id(monkeypatch, tmp_path) -> None:
    calls = {}

    def fake_fetch_and_archive(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(content_path=Path("raw.json"))

    monkeypatch.setattr(market, "fetch_and_archive", fake_fetch_and_archive)

    snapshot_polymarket_event(tmp_path, "highest-temperature-in-hong-kong-on-june-18-2026")

    assert calls["url"].endswith("/highest-temperature-in-hong-kong-on-june-18-2026")
    assert calls["source_id"] == (
        "polymarket_event_"
        + sha256_text("highest-temperature-in-hong-kong-on-june-18-2026")[:16]
    )


def test_snapshot_polymarket_event_rejects_unsafe_slug(tmp_path) -> None:
    with pytest.raises(MarketError, match="Unsafe/invalid"):
        snapshot_polymarket_event(tmp_path, "../bad")
