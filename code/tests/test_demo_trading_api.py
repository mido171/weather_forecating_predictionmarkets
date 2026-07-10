from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from hkg_tmax_demo_trading.api import create_app
from hkg_tmax_demo_trading.service import ResourceUnavailable


class FakeDemoService:
    def __init__(self) -> None:
        self.trade_payloads: list[dict[str, Any]] = []
        self.reset_called = False

    def list_markets(self, start: date, end: date) -> dict[str, Any]:
        return {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "markets": [
                {
                    "targetDate": "2026-07-06",
                    "slug": "highest-temperature-in-hong-kong-on-july-6-2026",
                    "title": "Highest temperature in Hong Kong on July 6?",
                    "openTradeCount": 0,
                    "settlement": None,
                    "snapshot": {"status": "ok"},
                }
            ],
        }

    def get_market(self, target_date: date, as_of_profile: str | None = None) -> dict[str, Any]:
        return self.refresh_market(target_date, as_of_profile)

    def refresh_market(self, target_date: date, as_of_profile: str | None = None) -> dict[str, Any]:
        return {
            "snapshotId": 1,
            "targetDate": target_date.isoformat(),
            "as_of_profile": as_of_profile or "t_minus_1_2359_hkt",
            "status": "ok",
            "event": {"slug": "highest-temperature-in-hong-kong-on-july-6-2026"},
            "forecast": {"forecastMaxC": 32.0, "source": "fake"},
            "model": {
                "method": "B4_hierarchical_residual_pmf",
                "trainRows": 100,
                "cutoff_profile": as_of_profile or "t_minus_1_2359_hkt",
            },
            "profile": {
                "id": as_of_profile or "t_minus_1_2359_hkt",
                "label": "17:59 Stockholm",
                "tradeable": True,
            },
            "marketRows": [
                {
                    "bucket": "32",
                    "label": "32",
                    "modelProbability": 0.55,
                    "marketBuyYesC": 40,
                    "marketBuyNoC": 60,
                    "yesEdgePp": 15,
                    "noEdgePp": -15,
                }
            ],
        }

    def create_trade(
        self,
        *,
        target_date: date,
        bucket_key: str,
        side: str,
        stake_usd: Any,
        manual_price_cents: Any | None = None,
        as_of_profile: str | None = None,
    ) -> dict[str, Any]:
        self.trade_payloads.append(
            {
                "target_date": target_date,
                "bucket_key": bucket_key,
                "side": side,
                "stake_usd": stake_usd,
                "manual_price_cents": manual_price_cents,
                "as_of_profile": as_of_profile,
            }
        )
        return {
            "trade": {
                "id": 10,
                "bucket_key": bucket_key,
                "side": side,
                "metadata_json": {"no_real_order": True},
            },
            "account": self.get_account(None),
        }

    def get_account(self, since: date | None) -> dict[str, Any]:
        return {
            "sessionId": 1,
            "cashUsd": 975.0,
            "equityUsd": 1000.0,
            "totalPnlUsd": 0.0,
            "changeSinceUsd": 0.0,
            "since": None if since is None else since.isoformat(),
            "openTrades": [],
            "trades": [],
            "balanceCurve": [],
        }

    def settle(self) -> dict[str, Any]:
        return {"settled": [], "account": self.get_account(None)}

    def settle_trade_as_win(self, trade_id: int) -> dict[str, Any]:
        return {
            "trade": {"id": trade_id, "status": "settled", "result": "win"},
            "account": self.get_account(None),
        }

    def settle_trade_as_loss(self, trade_id: int) -> dict[str, Any]:
        return {
            "trade": {"id": trade_id, "status": "settled", "result": "loss"},
            "account": self.get_account(None),
        }

    def reset_account(self) -> dict[str, Any]:
        self.reset_called = True
        return {"account": self.get_account(None), "session": {"id": 2}}


def test_api_exposes_market_trade_account_and_reset(tmp_path: Path) -> None:
    service = FakeDemoService()
    app = create_app(service=service, static_dir=tmp_path / "missing", apply_schema_on_startup=False)
    client = TestClient(app)

    assert client.get("/api/health").json()["ok"] is True

    profiles = client.get("/api/profiles").json()
    assert profiles["defaultProfile"] == "t_minus_1_2359_hkt"
    assert profiles["strategyGate"]["minEdgePp"] == 15.0
    assert profiles["profiles"][0]["id"] == "t_minus_1_1800_hkt"

    markets = client.get("/api/markets?start=2026-07-01&end=2026-07-10").json()
    assert markets["markets"][0]["targetDate"] == "2026-07-06"

    market = client.get("/api/markets/2026-07-06?asOfProfile=t_minus_1_2100_hkt").json()
    assert market["marketRows"][0]["bucket"] == "32"
    assert market["as_of_profile"] == "t_minus_1_2100_hkt"

    trade = client.post(
        "/api/trades",
        json={
            "targetDate": "2026-07-06",
            "bucketKey": "32",
            "side": "yes",
            "stakeUsd": 25,
            "manualPriceCents": 40,
            "asOfProfile": "t_minus_1_2100_hkt",
        },
    ).json()
    assert trade["trade"]["metadata_json"]["no_real_order"] is True
    assert service.trade_payloads[0]["manual_price_cents"] == 40
    assert service.trade_payloads[0]["as_of_profile"] == "t_minus_1_2100_hkt"

    account = client.get("/api/account?since=2026-07-01").json()
    assert account["since"] == "2026-07-01"

    settled = client.post("/api/trades/10/settle-win", json={}).json()
    assert settled["trade"]["result"] == "win"
    assert settled["trade"]["status"] == "settled"

    settled_loss = client.post("/api/trades/10/settle-loss", json={}).json()
    assert settled_loss["trade"]["result"] == "loss"
    assert settled_loss["trade"]["status"] == "settled"

    reset = client.post("/api/account/reset", json={}).json()
    assert reset["session"]["id"] == 2
    assert service.reset_called is True


def test_api_maps_expected_resource_errors_to_409(tmp_path: Path) -> None:
    class FailingService(FakeDemoService):
        def create_trade(self, **_: Any) -> dict[str, Any]:
            raise ResourceUnavailable("Model probability is unavailable")

    app = create_app(service=FailingService(), static_dir=tmp_path / "missing", apply_schema_on_startup=False)
    client = TestClient(app)

    response = client.post(
        "/api/trades",
        json={"targetDate": "2026-07-06", "bucketKey": "32", "side": "yes", "stakeUsd": 25},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Model probability is unavailable"
