"""FastAPI app for the local HKG Polymarket demo backtester."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hkg_tmax.paths import find_project_root

from .domain import (
    DEFAULT_WINDOW_END,
    DEFAULT_WINDOW_START,
    DemoTradingError,
)
from .probability import profile_catalog
from .service import DemoTradingService, ResourceUnavailable
from .store import PgDemoTradingStore

DEFAULT_DATABASE_URL = "postgresql://postgres:root@127.0.0.1:5432/hkg_tmax_research"


class RefreshMarketRequest(BaseModel):
    asOfProfile: str | None = Field(
        default=None,
        description="Validated cutoff profile such as t_minus_1_2100_hkt, or live_now for exploratory snapshots",
    )


class TradeCreateRequest(BaseModel):
    targetDate: date
    bucketKey: str
    side: str
    stakeUsd: float = Field(gt=0)
    manualPriceCents: float | None = Field(default=None, gt=0, le=100)
    asOfProfile: str | None = Field(default=None)


def default_database_url() -> str:
    return (
        os.environ.get("HKG_TMAX_DEMO_DATABASE_URL")
        or os.environ.get("HKG_TMAX_DATABASE_URL")
        or os.environ.get("HKG_TMAX_DB_DSN")
        or os.environ.get("DATABASE_URL")
        or DEFAULT_DATABASE_URL
    )


def default_repo_root() -> Path:
    return find_project_root(Path(__file__))


def default_static_dir(repo_root: Path) -> Path:
    return repo_root / "apps" / "polymarket-backtester" / "dist"


def create_service(*, repo_root: Path | None = None, database_url: str | None = None) -> DemoTradingService:
    resolved_root = repo_root or default_repo_root()
    resolved_database_url = database_url or default_database_url()
    return DemoTradingService(
        store=PgDemoTradingStore(resolved_database_url),
        repo_root=resolved_root,
        database_url=resolved_database_url,
    )


def _error_response(exc: Exception, status_code: int) -> HTTPException:
    return HTTPException(status_code=status_code, detail=str(exc))


def create_app(
    *,
    service: DemoTradingService | Any | None = None,
    repo_root: Path | None = None,
    database_url: str | None = None,
    static_dir: Path | None = None,
    apply_schema_on_startup: bool = True,
) -> FastAPI:
    resolved_root = repo_root or default_repo_root()
    app_service = service or create_service(repo_root=resolved_root, database_url=database_url)
    app = FastAPI(
        title="HKG Polymarket Demo Backtester",
        version="0.1.0",
        description="Local-only fictitious HKG Tmax Polymarket backtester. No real orders.",
    )
    app.state.service = app_service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:6000",
            "http://localhost:6000",
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if apply_schema_on_startup:

        @app.on_event("startup")
        def _apply_schema() -> None:
            app.state.service.apply_schema()

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "service": "hkg_tmax_demo_trading"}

    @app.get("/api/profiles")
    def profiles() -> dict[str, Any]:
        return {
            "profiles": profile_catalog(),
            "defaultProfile": "t_minus_1_2359_hkt",
            "strategyGate": {"minEdgePp": 15.0, "minWinProbability": 0.70},
        }

    @app.get("/api/markets")
    def markets(
        start: date = DEFAULT_WINDOW_START,
        end: date = DEFAULT_WINDOW_END,
    ) -> dict[str, Any]:
        try:
            return app.state.service.list_markets(start, end)
        except DemoTradingError as exc:
            raise _error_response(exc, 400) from exc

    @app.get("/api/markets/{target_date}")
    def market(target_date: date, asOfProfile: str | None = None) -> dict[str, Any]:
        try:
            return app.state.service.get_market(target_date, asOfProfile)
        except ResourceUnavailable as exc:
            raise _error_response(exc, 409) from exc
        except DemoTradingError as exc:
            raise _error_response(exc, 400) from exc

    @app.post("/api/markets/{target_date}/refresh")
    def refresh_market(target_date: date, request: RefreshMarketRequest | None = None) -> dict[str, Any]:
        try:
            return app.state.service.refresh_market(
                target_date,
                None if request is None else request.asOfProfile,
            )
        except ResourceUnavailable as exc:
            raise _error_response(exc, 409) from exc
        except DemoTradingError as exc:
            raise _error_response(exc, 400) from exc

    @app.post("/api/trades")
    def create_trade(request: TradeCreateRequest) -> dict[str, Any]:
        try:
            return app.state.service.create_trade(
                target_date=request.targetDate,
                bucket_key=request.bucketKey,
                side=request.side,
                stake_usd=request.stakeUsd,
                manual_price_cents=request.manualPriceCents,
                as_of_profile=request.asOfProfile,
            )
        except ResourceUnavailable as exc:
            raise _error_response(exc, 409) from exc
        except DemoTradingError as exc:
            raise _error_response(exc, 400) from exc

    @app.get("/api/account")
    def account(since: date | None = None) -> dict[str, Any]:
        try:
            return app.state.service.get_account(since)
        except DemoTradingError as exc:
            raise _error_response(exc, 400) from exc

    @app.post("/api/settle")
    def settle() -> dict[str, Any]:
        try:
            return app.state.service.settle()
        except DemoTradingError as exc:
            raise _error_response(exc, 400) from exc

    @app.post("/api/trades/{trade_id}/settle-win")
    def settle_trade_as_win(trade_id: int) -> dict[str, Any]:
        try:
            return app.state.service.settle_trade_as_win(trade_id)
        except ResourceUnavailable as exc:
            raise _error_response(exc, 409) from exc
        except DemoTradingError as exc:
            raise _error_response(exc, 400) from exc

    @app.post("/api/trades/{trade_id}/settle-loss")
    def settle_trade_as_loss(trade_id: int) -> dict[str, Any]:
        try:
            return app.state.service.settle_trade_as_loss(trade_id)
        except ResourceUnavailable as exc:
            raise _error_response(exc, 409) from exc
        except DemoTradingError as exc:
            raise _error_response(exc, 400) from exc

    @app.post("/api/account/reset")
    def reset_account() -> dict[str, Any]:
        try:
            return app.state.service.reset_account()
        except DemoTradingError as exc:
            raise _error_response(exc, 400) from exc

    resolved_static = static_dir or default_static_dir(resolved_root)
    if resolved_static.exists():
        app.mount("/", StaticFiles(directory=resolved_static, html=True), name="static")
    else:

        @app.get("/", response_class=HTMLResponse)
        def missing_static() -> str:
            return (
                "<!doctype html><title>HKG Demo Backtester</title>"
                "<main style='font-family: system-ui; padding: 32px'>"
                "<h1>Frontend build not found</h1>"
                "<p>Run <code>npm --prefix apps/polymarket-backtester run build</code>, "
                "then restart the server.</p></main>"
            )

    return app
