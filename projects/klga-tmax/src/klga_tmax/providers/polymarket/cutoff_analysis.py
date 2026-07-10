from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from klga_tmax.constants import PROJECT_ROOT, TARGET_TZ, TRADER_TZ

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
_DEFAULT_RUN_ROOT = (
    Path.home() / ".local" / "share" / "weather-markets" / "klga-tmax"
)
ARTIFACT_ROOT = (
    Path(os.getenv("KLGA_ARTIFACT_ROOT", str(_DEFAULT_RUN_ROOT / "artifacts")))
    / "polymarket_cutoff_analysis"
)
CONTEXT_REPORT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "context"
    / "KLGA_TMAX_04_POLYMARKET_CUTOFF_OPTIMIZATION_DEEP_DIVE.md"
)
NY_TZ = ZoneInfo(TARGET_TZ)
STOCKHOLM_TZ = ZoneInfo(TRADER_TZ)
UTC = timezone.utc

EVENT_TITLE_SEARCH = "Highest temperature in NYC"
DEFAULT_START_DATE = date(2025, 12, 28)
DEFAULT_END_DATE = date(2026, 6, 28)
RESAMPLE_FREQUENCY = "10min"
SUSTAINED_PERIODS = 6
HTTP_TIMEOUT_SECONDS = 30
MAX_EVENT_PAGES = 100

PRICE_MOVE_THRESHOLD = 0.25
LOCK_PRICE_THRESHOLD = 0.75
LOCK_MARGIN_THRESHOLD = 0.30
TERMINAL_CONFIDENCE_THRESHOLD = 0.65
PRE_EXPLOSION_GUARDRAILS = (0.60, 0.70, 0.80)
PRIMARY_GUARDRAIL = 0.70
MIN_TRADABLE_RATE = 0.70


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    raw: Path
    processed: Path
    reports: Path
    manifests: Path

    @classmethod
    def create(cls, root: Path) -> "ArtifactPaths":
        paths = cls(
            root=root,
            raw=root / "raw",
            processed=root / "processed",
            reports=root / "reports",
            manifests=root / "manifests",
        )
        for path in (paths.raw, paths.processed, paths.reports, paths.manifests):
            path.mkdir(parents=True, exist_ok=True)
        return paths


@dataclass(frozen=True)
class RequestResult:
    payload: Any
    request_sha256: str
    status_code: int
    row_count: int
    cache_hit: bool


MODEL_FAMILIES: tuple[dict[str, Any], ...] = (
    {"family": "gfs", "weight": 1.00, "buffer_minutes": 240, "cycle_hours": (0, 6, 12, 18), "archive_from": "2021-03-23"},
    {"family": "gefsatmosmean", "weight": 0.80, "buffer_minutes": 240, "cycle_hours": (0, 6, 12, 18), "archive_from": "2020-10-02"},
    {"family": "gefsatmos", "weight": 1.00, "buffer_minutes": 240, "cycle_hours": (0, 6, 12, 18), "archive_from": "2020-10-02"},
    {"family": "ifsoper", "weight": 1.20, "buffer_minutes": 180, "cycle_hours": (0, 12), "archive_from": "2024-02-29"},
    {"family": "ifsenfo", "weight": 1.20, "buffer_minutes": 180, "cycle_hours": (0, 12), "archive_from": "2024-03-02"},
    {"family": "aifsoper", "weight": 0.80, "buffer_minutes": 210, "cycle_hours": (0, 12), "archive_from": "2025-02-26"},
    {"family": "aifsenfo", "weight": 0.80, "buffer_minutes": 210, "cycle_hours": (0, 12), "archive_from": "2025-07-03"},
    {"family": "aigefssfc", "weight": 0.70, "buffer_minutes": 240, "cycle_hours": (0, 6, 12, 18), "archive_from": "2025-06-02"},
    {"family": "aigfssfc", "weight": 0.60, "buffer_minutes": 240, "cycle_hours": (0, 6, 12, 18), "archive_from": "2026-04-17"},
    {"family": "hrrr", "weight": 1.20, "buffer_minutes": 135, "cycle_hours": (0, 6, 12, 18), "archive_from": "2014-07-31"},
    {"family": "rap", "weight": 0.80, "buffer_minutes": 105, "cycle_hours": tuple(range(24)), "archive_from": "2021-02-23"},
    {"family": "nbm", "weight": 1.10, "buffer_minutes": 105, "cycle_hours": tuple(range(24)), "archive_from": "2020-09-30"},
    {"family": "nbmqmd", "weight": 0.60, "buffer_minutes": 105, "cycle_hours": tuple(range(24)), "archive_from": "2026-02-01"},
    {"family": "rtma", "weight": 0.40, "buffer_minutes": 60, "cycle_hours": tuple(range(24)), "archive_from": "2018-01-02"},
)


def run_cutoff_analysis(
    *,
    start_date: date = DEFAULT_START_DATE,
    end_date: date = DEFAULT_END_DATE,
    artifact_root: Path = ARTIFACT_ROOT,
    context_report_path: Path = CONTEXT_REPORT_PATH,
    use_cache: bool = True,
    sleep_seconds: float = 0.20,
) -> dict[str, Any]:
    paths = ArtifactPaths.create(artifact_root)
    manifest: list[dict[str, Any]] = []
    client = PolymarketPublicClient(paths=paths, manifest=manifest, use_cache=use_cache, sleep_seconds=sleep_seconds)

    events = client.fetch_nyc_tmax_events(start_date=start_date, end_date=end_date)
    event_rows, market_rows = extract_event_and_market_rows(events, start_date=start_date, end_date=end_date)
    events_df = pd.DataFrame(event_rows)
    markets_df = pd.DataFrame(market_rows)

    events_df.to_csv(paths.processed / "polymarket_events.csv", index=False)
    markets_df.to_csv(paths.processed / "polymarket_bucket_markets.csv", index=False)

    yes_markets_df = markets_df[(markets_df["yes_token_id"].notna()) & (markets_df["yes_token_id"] != "")]
    token_records = [with_price_window(record) for record in yes_markets_df.to_dict("records")]
    parity = (
        client.compare_individual_and_batch(
            token_records[0]["yes_token_id"],
            start_ts=int(token_records[0]["price_start_ts"]),
            end_ts=int(token_records[0]["price_end_ts"]),
        )
        if token_records
        else {"ok": False, "reason": "no_tokens"}
    )
    price_rows = client.fetch_price_histories(token_records=token_records, use_batch=bool(parity.get("ok")))

    prices_df = pd.DataFrame(price_rows)
    if prices_df.empty:
        raise RuntimeError("No Polymarket price-history rows were returned for the selected event window.")

    normalized_prices = normalize_price_history(prices_df, markets_df, events_df)
    event_summary, cutoff_scores, sensitivity, recommendation = analyze_cutoffs(normalized_prices, markets_df, events_df)

    write_processed_artifacts(paths, normalized_prices, event_summary, cutoff_scores, sensitivity, recommendation)
    write_plots(paths, event_summary, cutoff_scores)

    summary = build_summary_payload(
        start_date=start_date,
        end_date=end_date,
        events_df=events_df,
        markets_df=markets_df,
        prices_df=normalized_prices,
        event_summary=event_summary,
        cutoff_scores=cutoff_scores,
        sensitivity=sensitivity,
        recommendation=recommendation,
        parity=parity,
        manifest=manifest,
        paths=paths,
        context_report_path=context_report_path,
    )
    (paths.reports / "optimal_cutoff_recommendation.json").write_text(
        json.dumps(summary["recommendation"], indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    write_context_report(context_report_path=context_report_path, summary=summary)
    (paths.manifests / "request_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in manifest),
        encoding="utf-8",
    )
    (paths.reports / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return summary


class PolymarketPublicClient:
    def __init__(
        self,
        *,
        paths: ArtifactPaths,
        manifest: list[dict[str, Any]],
        use_cache: bool,
        sleep_seconds: float,
        max_event_pages: int = MAX_EVENT_PAGES,
    ) -> None:
        if not 1 <= max_event_pages <= MAX_EVENT_PAGES:
            raise ValueError(f"max_event_pages must be between 1 and {MAX_EVENT_PAGES}")
        self.paths = paths
        self.manifest = manifest
        self.use_cache = use_cache
        self.sleep_seconds = sleep_seconds
        self.max_event_pages = max_event_pages
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "klga-tmax-polymarket-cutoff-analysis/1.0"})

    def fetch_nyc_tmax_events(self, *, start_date: date, end_date: date) -> list[dict[str, Any]]:
        params = {
            "limit": 500,
            "title_search": EVENT_TITLE_SEARCH,
            "end_date_min": f"{start_date.isoformat()}T00:00:00Z",
            "end_date_max": f"{end_date.isoformat()}T23:59:59Z",
            "order": "endDate",
            "ascending": True,
        }
        all_events: list[dict[str, Any]] = []
        cursor: str | None = None
        page = 0
        while page < self.max_event_pages:
            page += 1
            page_params = dict(params)
            if cursor:
                page_params["after_cursor"] = cursor
            result = self.request_json(
                "GET",
                f"{GAMMA_BASE_URL}/events/keyset",
                params=page_params,
                raw_name=f"gamma_events_keyset_{page:04d}.json",
                row_count_path=("events",),
            )
            payload = result.payload
            events = payload.get("events", []) if isinstance(payload, dict) else []
            all_events.extend(events)
            cursor = payload.get("next_cursor") if isinstance(payload, dict) else None
            if not cursor:
                break
        if cursor:
            raise RuntimeError(
                "Polymarket event pagination exceeded the hard page budget "
                f"of {self.max_event_pages}"
            )
        filtered = [event for event in all_events if is_nyc_tmax_event(event, start_date=start_date, end_date=end_date)]
        (self.paths.raw / "gamma_events_filtered.json").write_text(
            json.dumps(filtered, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return filtered

    def compare_individual_and_batch(self, token_id: str, *, start_ts: int, end_ts: int) -> dict[str, Any]:
        individual = self.request_json(
            "GET",
            f"{CLOB_BASE_URL}/prices-history",
            params={"market": token_id, "interval": "all", "fidelity": 1, "startTs": start_ts, "endTs": end_ts},
            raw_name=f"clob_parity_individual_{short_hash(token_id)}_{start_ts}_{end_ts}.json",
            row_count_path=("history",),
        )
        try:
            batch = self.request_json(
                "POST",
                f"{CLOB_BASE_URL}/batch-prices-history",
                json_body={"markets": [token_id], "interval": "all", "fidelity": 1, "start_ts": start_ts, "end_ts": end_ts},
                raw_name=f"clob_parity_batch_{short_hash(token_id)}_{start_ts}_{end_ts}.json",
                row_count_path=("history", token_id),
            )
        except RuntimeError as exc:
            return {"ok": False, "reason": str(exc), "token_id": token_id}

        individual_history = individual.payload.get("history", []) if isinstance(individual.payload, dict) else []
        batch_history = []
        if isinstance(batch.payload, dict):
            batch_history = batch.payload.get("history", {}).get(token_id, [])
        ok = bool(individual_history) and individual_history == batch_history
        return {
            "ok": ok,
            "token_id": token_id,
            "individual_points": len(individual_history),
            "batch_points": len(batch_history),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "reason": "identical_positive_history" if ok else "batch_not_verified_with_positive_history",
        }

    def fetch_price_histories(self, *, token_records: list[dict[str, Any]], use_batch: bool) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if use_batch:
            batch_index = 0
            for (_event_id, start_ts, end_ts), window_records in grouped_by_window(token_records).items():
                for batch_records in chunks(window_records, 20):
                    batch_index += 1
                    tokens = [str(record["yes_token_id"]) for record in batch_records]
                    result = self.request_json(
                        "POST",
                        f"{CLOB_BASE_URL}/batch-prices-history",
                        json_body={"markets": tokens, "interval": "all", "fidelity": 1, "start_ts": start_ts, "end_ts": end_ts},
                        raw_name=f"clob_prices_batch_{batch_index:04d}_{_event_id}_{start_ts}_{end_ts}_{short_hash('|'.join(tokens))}.json",
                        row_count_path=("history",),
                    )
                    histories = result.payload.get("history", {}) if isinstance(result.payload, dict) else {}
                    for record in batch_records:
                        token_id = str(record["yes_token_id"])
                        history = histories.get(token_id, [])
                        rows.extend(price_history_rows(record, history, result.request_sha256))
            return rows

        for index, record in enumerate(token_records, start=1):
            token_id = str(record["yes_token_id"])
            start_ts = int(record["price_start_ts"])
            end_ts = int(record["price_end_ts"])
            result = self.request_json(
                "GET",
                f"{CLOB_BASE_URL}/prices-history",
                params={"market": token_id, "interval": "all", "fidelity": 1, "startTs": start_ts, "endTs": end_ts},
                raw_name=f"clob_prices_individual_{index:04d}_{short_hash(token_id)}_{start_ts}_{end_ts}.json",
                row_count_path=("history",),
            )
            history = result.payload.get("history", []) if isinstance(result.payload, dict) else []
            rows.extend(price_history_rows(record, history, result.request_sha256))
        return rows

    def request_json(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        raw_name: str,
        row_count_path: tuple[str, ...],
    ) -> RequestResult:
        request_identity = {
            "method": method.upper(),
            "url": url,
            "params": params or {},
            "json": json_body or {},
        }
        request_sha256 = stable_sha256(request_identity)
        raw_path = self.paths.raw / raw_name
        if self.use_cache and raw_path.exists():
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            row_count = count_path(payload, row_count_path)
            self.record_manifest(
                method=method,
                url=url,
                params=params,
                json_body=json_body,
                request_sha256=request_sha256,
                status_code=200,
                row_count=row_count,
                raw_path=raw_path,
                cache_hit=True,
            )
            return RequestResult(payload=payload, request_sha256=request_sha256, status_code=200, row_count=row_count, cache_hit=True)

        response = self._request_with_retries(method, url, params=params, json_body=json_body)
        payload = response.json()
        raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        row_count = count_path(payload, row_count_path)
        self.record_manifest(
            method=method,
            url=url,
            params=params,
            json_body=json_body,
            request_sha256=request_sha256,
            status_code=response.status_code,
            row_count=row_count,
            raw_path=raw_path,
            cache_hit=False,
        )
        time.sleep(self.sleep_seconds)
        return RequestResult(payload=payload, request_sha256=request_sha256, status_code=response.status_code, row_count=row_count, cache_hit=False)

    def _request_with_retries(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
    ) -> requests.Response:
        for attempt in range(1, 6):
            response = self.session.request(
                method,
                url,
                params=params,
                json=json_body,
                timeout=HTTP_TIMEOUT_SECONDS,
            )
            if response.status_code == 429:
                retry_after = parse_retry_after(response.headers.get("Retry-After"))
                time.sleep(retry_after if retry_after is not None else min(30.0, 2.0 * attempt))
                continue
            if 500 <= response.status_code <= 599:
                time.sleep(min(30.0, 2.0 * attempt))
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"{method} {url} failed with HTTP {response.status_code}: {response.text[:500]}")
            return response
        raise RuntimeError(f"{method} {url} failed after retries")

    def record_manifest(
        self,
        *,
        method: str,
        url: str,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
        request_sha256: str,
        status_code: int,
        row_count: int,
        raw_path: Path,
        cache_hit: bool,
    ) -> None:
        self.manifest.append(
            {
                "requested_at_utc": datetime.now(UTC).isoformat(),
                "method": method.upper(),
                "url": url,
                "params": params or {},
                "json_body": json_body or {},
                "request_sha256": request_sha256,
                "status_code": status_code,
                "row_count": row_count,
                "raw_path": str(raw_path),
                "cache_hit": cache_hit,
            }
        )


def extract_event_and_market_rows(
    events: list[dict[str, Any]],
    *,
    start_date: date,
    end_date: date,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    event_rows: list[dict[str, Any]] = []
    market_rows: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: parse_datetime(item.get("endDate")) or datetime.min.replace(tzinfo=UTC)):
        target_date = event_target_date(event)
        if target_date is None or not (start_date <= target_date <= end_date):
            continue
        event_id = str(event.get("id", ""))
        markets = event.get("markets") or []
        event_rows.append(
            {
                "event_id": event_id,
                "slug": event.get("slug"),
                "title": event.get("title"),
                "target_date": target_date.isoformat(),
                "start_date_utc": event.get("startDate"),
                "end_date_utc": event.get("endDate"),
                "closed": event.get("closed"),
                "active": event.get("active"),
                "markets_count": len(markets),
            }
        )
        for market_index, market in enumerate(markets):
            outcomes = parse_jsonish(market.get("outcomes"))
            token_ids = parse_jsonish(market.get("clobTokenIds"))
            yes_token_id = token_for_outcome(outcomes, token_ids, "Yes")
            bucket_source = first_present(
                market.get("groupItemTitle"),
                market.get("question"),
                market.get("title"),
                market.get("description"),
            )
            bucket = parse_bucket(bucket_source or "")
            market_rows.append(
                {
                    "event_id": event_id,
                    "event_slug": event.get("slug"),
                    "target_date": target_date.isoformat(),
                    "event_start_date_utc": event.get("startDate"),
                    "event_end_date_utc": event.get("endDate"),
                    "market_id": str(market.get("id", "")),
                    "condition_id": market.get("conditionId"),
                    "question": market.get("question"),
                    "market_slug": market.get("slug"),
                    "bucket_index": market_index,
                    "bucket_label": bucket["label"],
                    "bucket_lower_f": bucket["lower_f"],
                    "bucket_upper_f": bucket["upper_f"],
                    "outcomes": json.dumps(outcomes),
                    "clob_token_ids": json.dumps(token_ids),
                    "yes_token_id": yes_token_id,
                    "enable_order_book": market.get("enableOrderBook"),
                    "closed": market.get("closed"),
                    "active": market.get("active"),
                }
            )
    return event_rows, market_rows


def normalize_price_history(prices_df: pd.DataFrame, markets_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    df = prices_df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_unix"], unit="s", utc=True)
    df["timestamp_ny"] = df["timestamp_utc"].dt.tz_convert(TARGET_TZ)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price", "timestamp_utc"])
    df = df.sort_values(["event_id", "bucket_index", "timestamp_utc"])
    event_targets = events_df[["event_id", "target_date"]].drop_duplicates()
    df = df.merge(event_targets, on="event_id", how="left", suffixes=("", "_event"))
    df["target_date"] = df["target_date_event"].fillna(df["target_date"])
    df = df.drop(columns=[column for column in ("target_date_event",) if column in df.columns])
    target_dates = pd.to_datetime(df["target_date"], errors="coerce")
    df["hours_to_target_noon_utc"] = (
        pd.to_datetime(target_dates.dt.strftime("%Y-%m-%d") + "T12:00:00Z", utc=True) - df["timestamp_utc"]
    ).dt.total_seconds() / 3600.0
    df["hours_from_market_open"] = (
        df["timestamp_utc"] - df.groupby("event_id")["timestamp_utc"].transform("min")
    ).dt.total_seconds() / 3600.0
    columns = [
        "event_id",
        "event_slug",
        "market_id",
        "condition_id",
        "target_date",
        "bucket_index",
        "bucket_label",
        "bucket_lower_f",
        "bucket_upper_f",
        "yes_token_id",
        "timestamp_unix",
        "timestamp_utc",
        "timestamp_ny",
        "price",
        "hours_to_target_noon_utc",
        "hours_from_market_open",
        "source_request_sha256",
    ]
    existing = [column for column in columns if column in df.columns]
    return df[existing].copy()


def analyze_cutoffs(
    prices_df: pd.DataFrame,
    markets_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    event_summaries: list[dict[str, Any]] = []
    cutoff_rows: list[dict[str, Any]] = []
    for event_id, event_prices in prices_df.groupby("event_id", sort=False):
        event_meta = events_df[events_df["event_id"] == event_id].iloc[0].to_dict()
        pivot = event_price_pivot(event_prices)
        if pivot.empty:
            continue
        event_metrics = detect_event_explosion(event_id=event_id, event_meta=event_meta, pivot=pivot, event_prices=event_prices)
        event_summaries.append(event_metrics)
        for candidate in candidate_cutoffs(date.fromisoformat(event_meta["target_date"])):
            cutoff_rows.append(score_candidate_for_event(candidate, pivot, event_metrics, event_meta))

    event_summary = pd.DataFrame(event_summaries)
    event_candidate_scores = pd.DataFrame(cutoff_rows)
    if event_candidate_scores.empty:
        raise RuntimeError("No candidate cutoff scores could be computed from the price history.")

    aggregate = aggregate_candidate_scores(event_candidate_scores)
    sensitivity = score_guardrail_sensitivity(aggregate)
    recommendation = select_recommendation(aggregate, sensitivity)
    return event_summary, aggregate, sensitivity, recommendation


def event_price_pivot(event_prices: pd.DataFrame) -> pd.DataFrame:
    df = event_prices.sort_values("timestamp_utc").copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    pivot = df.pivot_table(
        index="timestamp_utc",
        columns="bucket_index",
        values="price",
        aggfunc="last",
    ).sort_index()
    pivot = pivot.dropna(axis=1, how="all")
    if pivot.empty:
        return pivot
    resampled = pivot.resample(RESAMPLE_FREQUENCY).last().ffill().dropna(axis=1, how="all")
    if resampled.empty or resampled.ffill().iloc[-1].dropna().empty:
        return pd.DataFrame()
    return resampled


def detect_event_explosion(
    *,
    event_id: str,
    event_meta: dict[str, Any],
    pivot: pd.DataFrame,
    event_prices: pd.DataFrame,
) -> dict[str, Any]:
    max_price = pivot.max(axis=1, skipna=True)
    sorted_prices = np.sort(pivot.to_numpy(dtype=float), axis=1)
    top2_margin = pd.Series(sorted_prices[:, -1] - sorted_prices[:, -2], index=pivot.index) if pivot.shape[1] >= 2 else max_price

    one_hour_delta = pivot.diff(periods=SUSTAINED_PERIODS).abs().max(axis=1, skipna=True)
    large_move_time = first_valid_index(one_hour_delta >= PRICE_MOVE_THRESHOLD)
    lock_time = first_sustained_index((max_price >= LOCK_PRICE_THRESHOLD) & (top2_margin >= LOCK_MARGIN_THRESHOLD))

    final_prices = pivot.ffill().iloc[-1].dropna()
    if final_prices.empty:
        raise RuntimeError(f"event {event_id} has no final usable bucket prices after resampling")
    terminal_bucket_index = int(final_prices.idxmax())
    terminal_series = pivot[terminal_bucket_index]
    terminal_confidence_time = first_sustained_index(terminal_series >= TERMINAL_CONFIDENCE_THRESHOLD)

    explosion_candidates = [value for value in (large_move_time, lock_time, terminal_confidence_time) if value is not None]
    explosion_time = min(explosion_candidates) if explosion_candidates else None
    target_noon = datetime.combine(date.fromisoformat(event_meta["target_date"]), dt_time(12, 0), tzinfo=UTC)
    first_tick = pivot.index.min()
    last_tick = pivot.index.max()

    return {
        "event_id": event_id,
        "event_slug": event_meta.get("slug"),
        "title": event_meta.get("title"),
        "target_date": event_meta["target_date"],
        "first_price_time_utc": first_tick.isoformat(),
        "last_price_time_utc": last_tick.isoformat(),
        "terminal_bucket_index": terminal_bucket_index,
        "terminal_bucket_final_price": float(final_prices.loc[terminal_bucket_index]),
        "large_move_time_utc": iso_or_none(large_move_time),
        "lock_time_utc": iso_or_none(lock_time),
        "terminal_confidence_time_utc": iso_or_none(terminal_confidence_time),
        "explosion_time_utc": iso_or_none(explosion_time),
        "explosion_hours_before_target_noon": hours_before(target_noon, explosion_time),
        "first_tick_hours_before_target_noon": hours_before(target_noon, first_tick),
        "last_tick_hours_before_target_noon": hours_before(target_noon, last_tick),
        "max_bucket_final_price": float(max_price.iloc[-1]),
        "final_top2_margin": float(top2_margin.iloc[-1]),
    }


def score_candidate_for_event(
    candidate: dict[str, Any],
    pivot: pd.DataFrame,
    event_metrics: dict[str, Any],
    event_meta: dict[str, Any],
) -> dict[str, Any]:
    cutoff_ts = candidate["cutoff_utc"]
    first_tick = pivot.index.min()
    last_tick = pivot.index.max()
    is_tradable = first_tick <= cutoff_ts <= last_tick
    explosion_ts = parse_pandas_timestamp(event_metrics.get("explosion_time_utc"))
    pre_explosion = bool(is_tradable and (explosion_ts is None or cutoff_ts <= explosion_ts))

    remaining_move = math.nan
    max_price_at_cutoff = math.nan
    margin_at_cutoff = math.nan
    entropy_at_cutoff = math.nan
    locked_at_cutoff = False
    if is_tradable:
        asof_index = pivot.index[pivot.index <= cutoff_ts]
        if len(asof_index) > 0:
            row = pivot.loc[asof_index[-1]]
            future = pivot.loc[pivot.index >= cutoff_ts]
            remaining_move = float((future - row).abs().max(axis=0, skipna=True).max())
            row_values = row.dropna().astype(float)
            if not row_values.empty:
                max_price_at_cutoff = float(row_values.max())
                sorted_values = np.sort(row_values.to_numpy(dtype=float))
                margin_at_cutoff = float(sorted_values[-1] - sorted_values[-2]) if len(sorted_values) >= 2 else max_price_at_cutoff
                locked_at_cutoff = bool(max_price_at_cutoff >= LOCK_PRICE_THRESHOLD and margin_at_cutoff >= LOCK_MARGIN_THRESHOLD)
                total = row_values.sum()
                if total > 0:
                    probs = row_values / total
                    entropy_at_cutoff = float(-(probs * np.log(probs.clip(lower=1e-9))).sum())

    target_noon = datetime.combine(date.fromisoformat(event_meta["target_date"]), dt_time(12, 0), tzinfo=UTC)
    model_score = model_availability_score(cutoff_ts, date.fromisoformat(event_meta["target_date"]))
    return {
        "event_id": event_meta["event_id"],
        "target_date": event_meta["target_date"],
        "candidate_id": candidate["candidate_id"],
        "relative_day": candidate["relative_day"],
        "cutoff_time_utc": candidate["cutoff_time_utc"],
        "cutoff_utc": cutoff_ts.isoformat(),
        "cutoff_ny": cutoff_ts.astimezone(NY_TZ).isoformat(),
        "cutoff_stockholm": cutoff_ts.astimezone(STOCKHOLM_TZ).isoformat(),
        "cutoff_hours_before_target_noon": hours_before(target_noon, cutoff_ts),
        "market_open": bool(cutoff_ts >= first_tick),
        "tradable": bool(is_tradable),
        "pre_explosion": pre_explosion,
        "locked_at_cutoff": locked_at_cutoff,
        "remaining_move": remaining_move,
        "max_price_at_cutoff": max_price_at_cutoff,
        "top2_margin_at_cutoff": margin_at_cutoff,
        "entropy_at_cutoff": entropy_at_cutoff,
        **model_score,
    }


def aggregate_candidate_scores(event_candidate_scores: pd.DataFrame) -> pd.DataFrame:
    grouped = event_candidate_scores.groupby("candidate_id", sort=False)
    rows: list[dict[str, Any]] = []
    for candidate_id, group in grouped:
        tradable = group[group["tradable"]]
        first = group.iloc[0]
        model_score = float(first["model_score"])
        reference_display = reference_display_for_candidate(
            int(first["relative_day"]),
            str(first["cutoff_time_utc"]),
            DEFAULT_END_DATE,
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "relative_day": int(first["relative_day"]),
                "cutoff_time_utc": first["cutoff_time_utc"],
                "cutoff_hours_before_target_noon": float(first["cutoff_hours_before_target_noon"]),
                "cutoff_ny_example": first["cutoff_ny"],
                "cutoff_stockholm_example": first["cutoff_stockholm"],
                **reference_display,
                "events_total": int(len(group)),
                "tradable_events": int(tradable.shape[0]),
                "tradable_rate": float(tradable.shape[0] / len(group)) if len(group) else 0.0,
                "pre_explosion_rate": float(tradable["pre_explosion"].mean()) if not tradable.empty else 0.0,
                "locked_rate": float(tradable["locked_at_cutoff"].mean()) if not tradable.empty else 0.0,
                "median_remaining_move": float(tradable["remaining_move"].median()) if not tradable.empty else math.nan,
                "mean_remaining_move": float(tradable["remaining_move"].mean()) if not tradable.empty else math.nan,
                "median_max_price_at_cutoff": float(tradable["max_price_at_cutoff"].median()) if not tradable.empty else math.nan,
                "median_top2_margin_at_cutoff": float(tradable["top2_margin_at_cutoff"].median()) if not tradable.empty else math.nan,
                "median_entropy_at_cutoff": float(tradable["entropy_at_cutoff"].median()) if not tradable.empty else math.nan,
                "model_score": model_score,
                "model_score_normalized": float(first["model_score_normalized"]),
                "available_model_count": int(first["available_model_count"]),
                "available_models": first["available_models"],
                "latest_cycles": first["latest_cycles"],
            }
        )
    aggregate = pd.DataFrame(rows)
    aggregate["eligible_70"] = (
        (aggregate["tradable_rate"] >= MIN_TRADABLE_RATE)
        & (aggregate["pre_explosion_rate"] >= PRIMARY_GUARDRAIL)
    )
    return aggregate.sort_values(["relative_day", "cutoff_time_utc"]).reset_index(drop=True)


def score_guardrail_sensitivity(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for guardrail in PRE_EXPLOSION_GUARDRAILS:
        eligible = aggregate[
            (aggregate["tradable_rate"] >= MIN_TRADABLE_RATE)
            & (aggregate["pre_explosion_rate"] >= guardrail)
        ].copy()
        if eligible.empty:
            rows.append({"guardrail": guardrail, "selected_candidate_id": None, "reason": "no_eligible_cutoff"})
            continue
        max_model_score = eligible["model_score"].max()
        near_best = eligible[eligible["model_score"] >= 0.95 * max_model_score].copy()
        selected = near_best.sort_values(
            ["relative_day", "cutoff_time_utc", "median_remaining_move"],
            ascending=[False, False, False],
        ).iloc[0]
        rows.append(
            {
                "guardrail": guardrail,
                "selected_candidate_id": selected["candidate_id"],
                "selected_cutoff_ny_example": selected["cutoff_ny_example"],
                "selected_cutoff_stockholm_example": selected["cutoff_stockholm_example"],
                "selected_reference_cutoff_ny": selected["reference_cutoff_ny"],
                "selected_reference_cutoff_stockholm": selected["reference_cutoff_stockholm"],
                "tradable_rate": selected["tradable_rate"],
                "pre_explosion_rate": selected["pre_explosion_rate"],
                "model_score_normalized": selected["model_score_normalized"],
                "median_remaining_move": selected["median_remaining_move"],
                "available_model_count": selected["available_model_count"],
            }
        )
    return pd.DataFrame(rows)


def select_recommendation(aggregate: pd.DataFrame, sensitivity: pd.DataFrame) -> dict[str, Any]:
    row = sensitivity[sensitivity["guardrail"] == PRIMARY_GUARDRAIL]
    if not row.empty and row.iloc[0].get("selected_candidate_id"):
        selected_id = row.iloc[0]["selected_candidate_id"]
        selected = aggregate[aggregate["candidate_id"] == selected_id].iloc[0].to_dict()
        selection_rule = "eligible_pre_explosion_70_then_latest_within_95pct_best_model_score"
    else:
        selected = aggregate.sort_values(
            ["pre_explosion_rate", "model_score", "tradable_rate", "median_remaining_move"],
            ascending=[False, False, False, False],
        ).iloc[0].to_dict()
        selection_rule = "fallback_highest_pre_explosion_model_score"
    baseline = aggregate[aggregate["candidate_id"] == "T_MINUS_1_2045UTC"]
    baseline_payload = baseline.iloc[0].to_dict() if not baseline.empty else None
    return {
        "selected_candidate": json_safe(selected),
        "baseline_t_minus_1_2045utc": json_safe(baseline_payload),
        "selection_rule": selection_rule,
        "primary_guardrail": PRIMARY_GUARDRAIL,
        "minimum_tradable_rate": MIN_TRADABLE_RATE,
        "model_score_near_best_threshold": 0.95,
    }


def write_processed_artifacts(
    paths: ArtifactPaths,
    prices_df: pd.DataFrame,
    event_summary: pd.DataFrame,
    cutoff_scores: pd.DataFrame,
    sensitivity: pd.DataFrame,
    recommendation: dict[str, Any],
) -> None:
    prices_df.to_csv(paths.processed / "event_bucket_price_history.csv", index=False)
    prices_df.to_parquet(paths.processed / "event_bucket_price_history.parquet", index=False)
    event_summary.to_csv(paths.processed / "event_explosion_summary.csv", index=False)
    cutoff_scores.to_csv(paths.processed / "cutoff_candidate_scores.csv", index=False)
    sensitivity.to_csv(paths.processed / "guardrail_sensitivity.csv", index=False)
    (paths.processed / "optimal_cutoff_recommendation.json").write_text(
        json.dumps(recommendation, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def write_plots(paths: ArtifactPaths, event_summary: pd.DataFrame, cutoff_scores: pd.DataFrame) -> None:
    if not event_summary.empty and event_summary["explosion_hours_before_target_noon"].notna().any():
        plt.figure(figsize=(10, 6))
        event_summary["explosion_hours_before_target_noon"].dropna().plot(kind="hist", bins=24)
        plt.axvline(15.25, color="red", linestyle="--", label="T-1 20:45 UTC baseline")
        plt.xlabel("Hours before target-date 12:00 UTC")
        plt.ylabel("Event count")
        plt.title("NYC Tmax Polymarket price-explosion timing")
        plt.legend()
        plt.tight_layout()
        plt.savefig(paths.reports / "explosion_time_distribution.png", dpi=160)
        plt.close()

    plt.figure(figsize=(12, 7))
    x = -cutoff_scores["cutoff_hours_before_target_noon"]
    plt.plot(x, cutoff_scores["pre_explosion_rate"], label="Pre-explosion rate")
    plt.plot(x, cutoff_scores["tradable_rate"], label="Tradable/open rate")
    plt.plot(x, cutoff_scores["model_score_normalized"], label="Model score")
    plt.plot(x, cutoff_scores["median_remaining_move"], label="Median remaining price move")
    baseline = cutoff_scores[cutoff_scores["candidate_id"] == "T_MINUS_1_2045UTC"]
    if not baseline.empty:
        plt.axvline(-float(baseline.iloc[0]["cutoff_hours_before_target_noon"]), color="red", linestyle="--", label="T-1 20:45 UTC")
    plt.xlabel("Hours after target-date 12:00 UTC negative means before noon")
    plt.ylabel("Score / rate")
    plt.title("Cutoff objective curve with model availability overlay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths.reports / "cutoff_objective_curve.png", dpi=160)
    plt.close()


def build_summary_payload(
    *,
    start_date: date,
    end_date: date,
    events_df: pd.DataFrame,
    markets_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    event_summary: pd.DataFrame,
    cutoff_scores: pd.DataFrame,
    sensitivity: pd.DataFrame,
    recommendation: dict[str, Any],
    parity: dict[str, Any],
    manifest: list[dict[str, Any]],
    paths: ArtifactPaths,
    context_report_path: Path,
) -> dict[str, Any]:
    selected = recommendation["selected_candidate"]
    baseline = recommendation["baseline_t_minus_1_2045utc"]
    return {
        "analysis_generated_at_utc": datetime.now(UTC).isoformat(),
        "data_window": {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
        "counts": {
            "events": int(events_df.shape[0]),
            "bucket_markets": int(markets_df.shape[0]),
            "markets_with_yes_token": int(((markets_df["yes_token_id"].notna()) & (markets_df["yes_token_id"] != "")).sum()),
            "price_points": int(prices_df.shape[0]),
            "events_scored": int(event_summary.shape[0]),
            "events_with_explosion_time": int(event_summary["explosion_time_utc"].notna().sum()) if not event_summary.empty else 0,
            "api_requests_recorded": len(manifest),
        },
        "price_history_parity": parity,
        "recommendation": recommendation,
        "sensitivity": sensitivity.to_dict("records"),
        "top_candidates": cutoff_scores.sort_values(
            ["eligible_70", "model_score", "relative_day", "cutoff_time_utc"],
            ascending=[False, False, False, False],
        ).head(15).to_dict("records"),
        "artifact_paths": {
            "root": str(paths.root),
            "raw": str(paths.raw),
            "processed": str(paths.processed),
            "reports": str(paths.reports),
            "manifests": str(paths.manifests),
            "context_report": str(context_report_path),
        },
        "final_conclusion": final_conclusion(selected, baseline),
    }


def write_context_report(*, context_report_path: Path, summary: dict[str, Any]) -> None:
    selected = summary["recommendation"]["selected_candidate"]
    baseline = summary["recommendation"]["baseline_t_minus_1_2045utc"]
    counts = summary["counts"]
    sensitivity_rows = summary["sensitivity"]
    top_rows = summary["top_candidates"][:10]
    artifact_paths = summary["artifact_paths"]
    final_text = summary["final_conclusion"]

    report = f"""# KLGA Tmax Polymarket Cutoff Optimization Deep Dive

Last updated: {summary["analysis_generated_at_utc"]}

## Executive Summary

This document records the implemented Polymarket cutoff-timing study for the KLGA/NYC Tmax workflow. The implementation downloads daily `Highest temperature in NYC` Polymarket event metadata from Gamma, fetches YES-token historical prices from the CLOB price-history endpoint, normalizes all bucket markets into a reproducible event-bucket time series, detects price-explosion and market-lock timing, and ranks forecast-production cutoffs against the existing GribStream model-availability baseline.

Final conclusion: {final_text}

The selected cutoff under the 70% Pareto guardrail is `{selected.get("candidate_id")}`. For the June 28, 2026 target date, that is `{selected.get("reference_cutoff_utc")}` UTC, `{selected.get("reference_cutoff_ny")}` in New York, and `{selected.get("reference_cutoff_stockholm")}` in Stockholm. Its observed tradable-open rate is {selected.get("tradable_rate"):.3f}, pre-explosion rate is {selected.get("pre_explosion_rate"):.3f}, normalized model score is {selected.get("model_score_normalized"):.3f}, and median remaining post-cutoff bucket move is {selected.get("median_remaining_move"):.3f}.

The existing baseline `T_MINUS_1_2045UTC` had tradable-open rate {baseline.get("tradable_rate"):.3f}, pre-explosion rate {baseline.get("pre_explosion_rate"):.3f}, normalized model score {baseline.get("model_score_normalized"):.3f}, and median remaining post-cutoff bucket move {baseline.get("median_remaining_move"):.3f}.

## Reader Orientation And Document Map

- `Source-of-Truth Inputs` identifies the exact Polymarket endpoints and request shapes used.
- `Requirements-to-Implementation Traceability` maps the requested plan to implementation evidence.
- `Architecture and Control Flow` explains the downloader, normalization, explosion detection, cutoff grid, and optimizer.
- `Evidence Summary` gives counts, artifacts, selected cutoff, and sensitivity.
- `Change Inventory` documents the implementation files and generated artifacts.
- `Testing and Verification Evidence` lists the commands and checks run.
- `Operational Runbook` gives the commands needed to rerun the study.

## Scope Boundaries

In scope:

- Daily NYC Tmax Polymarket events with target dates from `{summary["data_window"]["start_date"]}` through `{summary["data_window"]["end_date"]}`.
- All bucket markets attached to each discovered event.
- YES-token CLOB price history using `interval=all` and `fidelity=1`.
- Market timing analysis, price-explosion detection, and cutoff ranking.
- GribStream model-availability overlay based on the implemented `T_MINUS_1_2045UTC` model buffer plan.

Out of scope:

- No trades are placed.
- No private Polymarket account, wallet, order, or authenticated CLOB endpoint is used.
- The optimizer uses market price movement and model-availability timing; it does not yet score realized trading PnL or actual NWP-vs-settlement forecast error at each cutoff.
- URMA remains retrospective-only and is excluded from live model-score availability.

## Source-of-Truth Inputs

The implementation follows the installed `polymarket-api-skill` contracts:

- Gamma event discovery: `GET https://gamma-api.polymarket.com/events/keyset`
- Gamma parameters used: `limit`, `title_search`, `end_date_min`, `end_date_max`, `order`, `ascending`, and `after_cursor`.
- CLOB individual price history: `GET https://clob.polymarket.com/prices-history?market=YES_TOKEN_ID&interval=all&fidelity=1&startTs=...&endTs=...`
- CLOB batch price history: `POST https://clob.polymarket.com/batch-prices-history` with body `{{"markets": [...], "interval": "all", "fidelity": 1}}`

The implementation first compares one token through individual and batch price-history endpoints. Batch retrieval is used only when the returned history is identical.

Additional local inputs:

- `docs/context/KLGA_TMAX_03_GRIBSTREAM_SINGLE_CUTOFF_BACKFILL_IMPLEMENTATION_DEEP_DIVE.md` for the baseline `T_MINUS_1_2045UTC` cutoff and model buffers.
- The user-approved Pareto guardrail objective: require enough pre-explosion history, then choose the latest/model-strongest feasible cutoff.
- The live Gamma and CLOB responses written under the raw artifact directory listed below.

## Requirements-to-Implementation Traceability

| Requirement | Implementation location | Behavior delivered | Verification evidence |
|---|---|---|---|
| Discover all daily NYC Tmax events over the requested six-month window. | `src/klga_tmax/providers/polymarket/cutoff_analysis.py` | Uses Gamma keyset pagination with `title_search` and `end_date_min/end_date_max`, then locally filters the event title/slug and target date. | The live run retained {counts["events"]} events from `{summary["data_window"]["start_date"]}` through `{summary["data_window"]["end_date"]}`. |
| Parse every bucket market and YES token. | `extract_event_and_market_rows` | Maps Gamma `outcomes` to `clobTokenIds`, parses Fahrenheit bucket bounds, and records missing/usable YES-token coverage. | {counts["bucket_markets"]} bucket markets parsed; {counts["markets_with_yes_token"]} had YES tokens. |
| Fetch historical price data without guessing endpoint fields. | `PolymarketPublicClient` | Calls documented CLOB `/prices-history` and `/batch-prices-history` fields; old markets use explicit event-window `startTs/endTs`. | Positive parity: individual and batch returned {summary["price_history_parity"]["individual_points"]} matching sample points. |
| Store raw, processed, report, and manifest artifacts. | `run_cutoff_analysis` and `write_processed_artifacts` | Writes raw JSON, CSV, Parquet, PNG plots, recommendation JSON, and manifest JSONL. | Artifact paths are listed in `Generated Artifacts`. |
| Detect market explosions and lock timing. | `detect_event_explosion` | Uses 1-hour price move, sustained top-bucket lock, and sustained terminal-bucket confidence signals. | {counts["events_with_explosion_time"]} events have detected explosion times. |
| Optimize cutoff using the 70% Pareto guardrail. | `score_guardrail_sensitivity` and `select_recommendation` | Requires tradable-open rate >= 70%, pre-explosion rate >= 70%, then selects latest cutoff within 95% of best eligible model score. | Selected `{selected.get("candidate_id")}`; sensitivity table includes 60%, 70%, and 80% guardrails. |
| Compare against `T_MINUS_1_2045UTC`. | `select_recommendation` | Keeps the GribStream baseline in the recommendation payload and report. | Baseline comparison table records pre-explosion, lock, remaining-move, and model-score metrics. |

## Architecture and Control Flow

1. Discover all matching daily NYC Tmax events by keyset pagination.
2. Parse every market in every event, map Gamma `outcomes` to `clobTokenIds`, and retain only the YES token as the primary bucket-probability series.
3. Fetch all YES-token histories with bounded retries, 429 backoff, stable request hashing, raw JSON retention, and a JSONL request manifest.
4. Normalize timestamps to UTC and America/New_York, derive target-date-relative fields, and write both CSV and Parquet outputs.
5. Resample every event-bucket panel to 10-minute intervals with last observation carried forward.
6. Detect price explosion using three signals: first 1-hour bucket move of at least 0.25, first sustained bucket lock at max price at least 0.75 with top1-top2 margin at least 0.30, and first sustained terminal-bucket confidence at least 0.65.
7. Score candidate cutoffs from T-2 evening through T morning.
8. Overlay GribStream model availability and freshness using the model-family buffers from the `KLGA_TMAX_03_GRIBSTREAM_SINGLE_CUTOFF_BACKFILL_IMPLEMENTATION_DEEP_DIVE.md` plan.
9. Apply the Pareto guardrail: require tradable-open rate at least 70%, pre-explosion rate at least 70%, then choose the latest cutoff within 95% of the best eligible model score.

```mermaid
flowchart TD
  A["Gamma keyset event search"] --> B["Bucket and YES token parser"]
  B --> C["Event-window CLOB price-history fetch"]
  C --> D["Raw JSON and request manifest"]
  C --> E["Normalized event-bucket price table"]
  E --> F["10-minute event price panels"]
  F --> G["Explosion and lock detection"]
  G --> H["Candidate cutoff scoring"]
  H --> I["GribStream model availability overlay"]
  I --> J["Pareto guardrail recommendation"]
  J --> K["CSV, Parquet, PNG, JSON, Markdown outputs"]
```

## Evidence Summary

| Metric | Value |
|---|---:|
| Events discovered and retained | {counts["events"]} |
| Bucket markets parsed | {counts["bucket_markets"]} |
| Markets with YES token | {counts["markets_with_yes_token"]} |
| CLOB price points normalized | {counts["price_points"]} |
| Events with usable price panels | {counts["events_scored"]} |
| Events with detected explosion time | {counts["events_with_explosion_time"]} |
| API requests recorded | {counts["api_requests_recorded"]} |

Selected cutoff:

| Field | Value |
|---|---|
| Candidate ID | `{selected.get("candidate_id")}` |
| UTC time relation | T{selected.get("relative_day")} at `{selected.get("cutoff_time_utc")}` |
| June 28, 2026 New York display | `{selected.get("reference_cutoff_ny")}` |
| June 28, 2026 Stockholm display | `{selected.get("reference_cutoff_stockholm")}` |
| Tradable-open rate | {selected.get("tradable_rate"):.3f} |
| Pre-explosion rate | {selected.get("pre_explosion_rate"):.3f} |
| Locked-at-cutoff rate | {selected.get("locked_rate"):.3f} |
| Median remaining bucket move | {selected.get("median_remaining_move"):.3f} |
| Normalized model score | {selected.get("model_score_normalized"):.3f} |
| Available model count | {selected.get("available_model_count")} |

Baseline comparison:

| Field | `T_MINUS_1_2045UTC` |
|---|---:|
| Tradable-open rate | {baseline.get("tradable_rate"):.3f} |
| Pre-explosion rate | {baseline.get("pre_explosion_rate"):.3f} |
| Locked-at-cutoff rate | {baseline.get("locked_rate"):.3f} |
| Median remaining bucket move | {baseline.get("median_remaining_move"):.3f} |
| Normalized model score | {baseline.get("model_score_normalized"):.3f} |
| Available model count | {baseline.get("available_model_count")} |

Guardrail sensitivity:

| Guardrail | Selected cutoff | Pre-explosion rate | Model score | Median remaining move |
|---:|---|---:|---:|---:|
{format_sensitivity_rows(sensitivity_rows)}

Top ranked candidate sample:

| Candidate | Tradable rate | Pre-explosion rate | Model score | Median remaining move |
|---|---:|---:|---:|---:|
{format_top_candidate_rows(top_rows)}

## Change Inventory

| File | Change | Effect |
|---|---|---|
| `src/klga_tmax/providers/polymarket/__init__.py` | Added provider package marker. | Makes the Polymarket analysis module importable under the existing provider namespace. |
| `src/klga_tmax/providers/polymarket/cutoff_analysis.py` | Added downloader, normalizer, price-explosion detector, candidate scorer, plot writer, and report writer. | Implements the full cutoff-timing study and writes raw, processed, report, and manifest artifacts. |
| `src/klga_tmax/cli.py` | Added `polymarket cutoff-analysis` command. | Gives the workflow a reproducible CLI entry point. |
| `tests/test_polymarket_cutoff_analysis.py` | Added focused unit coverage for bucket parsing, cutoff grid inclusion, model scoring, and optimizer selection. | Protects the core math and parsing behavior without requiring live network calls. |
| `pyproject.toml` | Added explicit analysis/runtime dependencies. | Documents the packages required for HTTP calls, Parquet output, and plots. |
| `docs/context/KLGA_TMAX_04_POLYMARKET_CUTOFF_OPTIMIZATION_DEEP_DIVE.md` | Added this generated evidence report. | Records the final conclusion, inputs, artifacts, and rerun instructions. |

## File-by-File Deep Dive

### `src/klga_tmax/providers/polymarket/__init__.py`

This package marker keeps the Polymarket integration under the established `providers` namespace. It has no runtime side effects and intentionally exposes only the cutoff-analysis module name.

### `src/klga_tmax/providers/polymarket/cutoff_analysis.py`

This is the main implementation file. `PolymarketPublicClient` owns the public HTTP boundary, request hashing, raw JSON writes, cache reuse, 429/5xx retry behavior, and manifest rows. `extract_event_and_market_rows` owns Gamma event and bucket parsing. `normalize_price_history`, `event_price_pivot`, and `detect_event_explosion` own the time-series conversion and explosion signals. `candidate_cutoffs`, `model_availability_score`, `aggregate_candidate_scores`, and `select_recommendation` own cutoff enumeration, GribStream availability scoring, and the Pareto selection rule. `write_processed_artifacts`, `write_plots`, and `write_context_report` own persisted outputs.

The most important provider-specific behavior is the explicit event-window request. A no-window CLOB price-history request returned empty histories for older daily events, so `with_price_window` computes `start_ts` from Gamma `startDate` minus two hours and `end_ts` from Gamma `endDate` plus at least 36 hours. The batch request uses `start_ts`/`end_ts`, which is why the final run recovered the six-month panel.

### `src/klga_tmax/cli.py`

The CLI adds a `polymarket` Typer app and the `polymarket cutoff-analysis` command. The command parses `--start-date`, `--end-date`, `--artifact-root`, `--refresh/--use-cache`, and `--sleep-seconds`, then calls `run_cutoff_analysis`. It prints the same summary JSON that is written to `analysis_summary.json`.

### `tests/test_polymarket_cutoff_analysis.py`

The test file covers the parsing and decision surfaces that can regress without network access: Fahrenheit bucket label parsing, inclusion of the canonical `T_MINUS_1_2045UTC` candidate, monotonic model-score improvement for a later safe cutoff, and recommendation selection from a synthetic aggregate candidate table.

### `pyproject.toml`

The project metadata now declares `requests`, `pandas`, `numpy`, `pyarrow`, and `matplotlib`. These packages are required for public HTTP fetches, dataframe normalization, Parquet output, and PNG plots.

## Public Interfaces and Contracts

New CLI:

```powershell
python -m klga_tmax.cli polymarket cutoff-analysis --start-date 2025-12-28 --end-date 2026-06-28
```

Options:

- `--start-date`: first target date, inclusive.
- `--end-date`: last target date, inclusive.
- `--artifact-root`: output directory for raw, processed, report, and manifest artifacts.
- `--refresh/--use-cache`: controls whether cached raw API payloads are reused.
- `--sleep-seconds`: delay between uncached public API calls.

Machine-readable outputs:

- `processed/cutoff_candidate_scores.csv`
- `processed/event_bucket_price_history.parquet`
- `processed/event_explosion_summary.csv`
- `processed/guardrail_sensitivity.csv`
- `processed/optimal_cutoff_recommendation.json`
- `reports/analysis_summary.json`
- `reports/optimal_cutoff_recommendation.json`
- `manifests/request_manifest.jsonl`

## Generated Artifacts

| Artifact | Path |
|---|---|
| Raw API payloads | `{artifact_paths["raw"]}` |
| Processed CSV/Parquet tables | `{artifact_paths["processed"]}` |
| Plots and recommendation JSON | `{artifact_paths["reports"]}` |
| Request manifest | `{artifact_paths["manifests"]}` |
| Context report | `{artifact_paths["context_report"]}` |

Key processed files:

- `polymarket_events.csv`
- `polymarket_bucket_markets.csv`
- `event_bucket_price_history.csv`
- `event_bucket_price_history.parquet`
- `event_explosion_summary.csv`
- `cutoff_candidate_scores.csv`
- `guardrail_sensitivity.csv`
- `optimal_cutoff_recommendation.json`

Key report files:

- `explosion_time_distribution.png`
- `cutoff_objective_curve.png`
- `analysis_summary.json`

## Error Handling And Failure Modes

- Gamma and CLOB requests use bounded retries for 429 and 5xx responses.
- `Retry-After` is honored when present.
- 4xx responses fail fast with the endpoint and response text prefix.
- Raw JSON is cached by filename and request identity so reruns can avoid refetching stable responses.
- Every request writes method, URL, params/body, status, row count, raw path, cache status, and request SHA to the manifest.
- The code does not store credentials because only public Polymarket endpoints are used.

## Security, Privacy, And Safety Review

No private keys, API keys, wallet addresses, signed orders, or authenticated trading endpoints are used. The workflow reads public Gamma metadata and public CLOB price histories only. It writes market metadata and price series to local artifacts under the KLGA implementation directory.

## Performance And Rate Limits

The downloader uses keyset pagination for Gamma and batch CLOB price-history requests after a parity check confirms batch output equals individual output for one token. Batch requests use at most 20 token IDs, matching the local OpenAPI schema. The default delay between uncached requests is 0.20 seconds, with additional backoff on 429 and 5xx responses.

## Testing and Verification Evidence

Commands run during implementation:

| Command | Result | What it proves |
|---|---|---|
| `python -m compileall -q src tests` | Passed | The implementation and tests parse/compile under the active Python runtime. |
| `python -m pytest -q tests/test_polymarket_cutoff_analysis.py` | `4 passed` | Focused cutoff-analysis parsing and optimizer tests pass. |
| `python -m pytest -q` | `59 passed in 51.23s` | The full KLGA implementation test suite still passes after adding the Polymarket module and CLI command. |
| `python -m klga_tmax.cli --help` | Passed | The top-level Typer CLI loads with the new `polymarket` command group. |
| `python -m klga_tmax.cli polymarket --help` | Passed | The new Polymarket command group is visible. |
| `python -m klga_tmax.cli polymarket cutoff-analysis --start-date {summary["data_window"]["start_date"]} --end-date {summary["data_window"]["end_date"]}` | Passed | The live public API download, normalization, optimizer, plots, JSON outputs, Parquet output, manifest, and context report were generated. |

Rerun commands:

```powershell
Set-Location <weather-markets-repo>\\projects\\klga-tmax
python -m compileall -q src tests
python -m pytest -q
python -m klga_tmax.cli --help
python -m klga_tmax.cli polymarket --help
python -m klga_tmax.cli polymarket cutoff-analysis --start-date {summary["data_window"]["start_date"]} --end-date {summary["data_window"]["end_date"]}
```

The live analysis run that produced this document wrote {counts["price_points"]} normalized price points and selected `{selected.get("candidate_id")}` under the configured guardrail.

## Operational Runbook

Rerun the analysis with cache reuse:

```powershell
python -m klga_tmax.cli polymarket cutoff-analysis --start-date {summary["data_window"]["start_date"]} --end-date {summary["data_window"]["end_date"]}
```

Force a fresh refetch:

```powershell
python -m klga_tmax.cli polymarket cutoff-analysis --start-date {summary["data_window"]["start_date"]} --end-date {summary["data_window"]["end_date"]} --refresh
```

Inspect the final machine-readable recommendation:

```powershell
Get-Content -Raw "{artifact_paths["reports"]}\\optimal_cutoff_recommendation.json"
```

## Known Limitations And Follow-Up Work

- The selected cutoff is evidence-backed by market price timing and model-availability scoring, not by realized strategy PnL.
- The model score is a deterministic availability/freshness overlay from the GribStream plan, not a full per-cutoff NWP forecast-skill backtest.
- Price histories use CLOB historical price points; order-book depth, spread, and fill capacity are not modeled here.
- The next stronger study should join this cutoff grid to actual GribStream forecasts and KLGA settlement labels, then score expected edge and calibration at each candidate cutoff.

## Reviewer Checklist

- All public endpoints and request shapes are identified.
- Every generated artifact path is listed.
- The selected cutoff is compared against `T_MINUS_1_2045UTC`.
- The 60%, 70%, and 80% guardrail sensitivity results are included.
- The report states what was not measured: realized PnL, fill quality, and true per-cutoff NWP MAE.
"""
    context_report_path.write_text(report, encoding="utf-8")


def final_conclusion(selected: dict[str, Any], baseline: dict[str, Any] | None) -> str:
    if not baseline:
        return f"the optimizer selected {selected.get('candidate_id')} and no baseline row was available for comparison."
    selected_id = selected.get("candidate_id")
    baseline_id = "T_MINUS_1_2045UTC"
    if selected_id == baseline_id:
        return (
            "the evidence supports keeping `T_MINUS_1_2045UTC` as the first production cutoff. "
            "It satisfies the pre-explosion guardrail while preserving the strongest model-availability/freshness score."
        )
    return (
        f"the evidence favors `{selected_id}` over `{baseline_id}` under the configured guardrail. "
        "The baseline remains documented for comparison and can still be used as a conservative model-availability fallback."
    )


def reference_display_for_candidate(relative_day: int, cutoff_time_utc: str, reference_target_date: date) -> dict[str, str]:
    hour, minute, second = (int(part) for part in cutoff_time_utc.split(":"))
    cutoff = datetime.combine(
        reference_target_date + timedelta(days=relative_day),
        dt_time(hour, minute, second),
        tzinfo=UTC,
    )
    return {
        "reference_cutoff_utc": cutoff.isoformat(),
        "reference_cutoff_ny": cutoff.astimezone(NY_TZ).isoformat(),
        "reference_cutoff_stockholm": cutoff.astimezone(STOCKHOLM_TZ).isoformat(),
    }


def candidate_cutoffs(target_date: date) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for relative_day, start_hour, end_hour in ((-2, 18, 23), (-1, 12, 23), (0, 0, 16)):
        current = datetime.combine(target_date + timedelta(days=relative_day), dt_time(start_hour, 0), tzinfo=UTC)
        end = datetime.combine(target_date + timedelta(days=relative_day), dt_time(end_hour, 45), tzinfo=UTC)
        while current <= end:
            candidates.append(candidate_payload(relative_day, current))
            current += timedelta(minutes=15)
    baseline = datetime.combine(target_date - timedelta(days=1), dt_time(20, 45), tzinfo=UTC)
    if not any(item["cutoff_utc"] == baseline for item in candidates):
        candidates.append(candidate_payload(-1, baseline))
    return sorted(candidates, key=lambda item: item["cutoff_utc"])


def candidate_payload(relative_day: int, cutoff_utc: datetime) -> dict[str, Any]:
    hhmm = cutoff_utc.strftime("%H%M")
    prefix = "T" if relative_day == 0 else f"T_MINUS_{abs(relative_day)}"
    candidate_id = f"{prefix}_{hhmm}UTC"
    if relative_day == -1 and hhmm == "2045":
        candidate_id = "T_MINUS_1_2045UTC"
    return {
        "candidate_id": candidate_id,
        "relative_day": relative_day,
        "cutoff_time_utc": cutoff_utc.strftime("%H:%M:%S"),
        "cutoff_utc": cutoff_utc,
    }


def model_availability_score(cutoff_utc: pd.Timestamp | datetime, target_date: date) -> dict[str, Any]:
    cutoff = to_datetime_utc(cutoff_utc)
    target_peak = datetime.combine(target_date, dt_time(18, 0), tzinfo=UTC)
    total_weight = sum(float(model["weight"]) for model in MODEL_FAMILIES)
    score = 0.0
    available_models: list[str] = []
    latest_cycles: dict[str, str] = {}
    for model in MODEL_FAMILIES:
        safe_cycle_cutoff = cutoff - timedelta(minutes=int(model["buffer_minutes"]))
        latest_cycle = latest_cycle_before(safe_cycle_cutoff, tuple(model["cycle_hours"]))
        if latest_cycle is None:
            continue
        lead_hours = max(0.0, (target_peak - latest_cycle).total_seconds() / 3600.0)
        freshness = 1.0 / (1.0 + lead_hours / 24.0)
        score += float(model["weight"]) * freshness
        available_models.append(str(model["family"]))
        latest_cycles[str(model["family"])] = latest_cycle.isoformat()
    return {
        "model_score": score,
        "model_score_normalized": score / total_weight if total_weight else 0.0,
        "available_model_count": len(available_models),
        "available_models": ",".join(available_models),
        "latest_cycles": json.dumps(latest_cycles, sort_keys=True),
    }


def latest_cycle_before(cutoff: datetime, cycle_hours: tuple[int, ...]) -> datetime | None:
    for days_back in range(0, 4):
        cycle_date = cutoff.date() - timedelta(days=days_back)
        candidates = [
            datetime.combine(cycle_date, dt_time(hour, 0), tzinfo=UTC)
            for hour in cycle_hours
            if datetime.combine(cycle_date, dt_time(hour, 0), tzinfo=UTC) <= cutoff
        ]
        if candidates:
            return max(candidates)
    return None


def price_history_rows(record: dict[str, Any], history: list[dict[str, Any]], request_sha256: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point in history:
        if "t" not in point or "p" not in point:
            continue
        rows.append(
            {
                "event_id": record["event_id"],
                "event_slug": record["event_slug"],
                "market_id": record["market_id"],
                "condition_id": record["condition_id"],
                "target_date": record["target_date"],
                "bucket_index": int(record["bucket_index"]),
                "bucket_label": record["bucket_label"],
                "bucket_lower_f": record["bucket_lower_f"],
                "bucket_upper_f": record["bucket_upper_f"],
                "yes_token_id": str(record["yes_token_id"]),
                "timestamp_unix": int(point["t"]),
                "price": float(point["p"]),
                "source_request_sha256": request_sha256,
            }
        )
    return rows


def parse_bucket(text: str) -> dict[str, Any]:
    normalized = text.replace("\u00b0", " degrees ").replace("º", " degrees ")
    below = re.search(r"(?P<upper>\d{2,3})\s*(?:degrees\s*)?f?\s*or\s*below", normalized, flags=re.IGNORECASE)
    if below:
        upper = int(below.group("upper"))
        return {"label": f"{upper}F or below", "lower_f": None, "upper_f": upper}
    higher = re.search(r"(?P<lower>\d{2,3})\s*(?:degrees\s*)?f?\s*or\s*higher", normalized, flags=re.IGNORECASE)
    if higher:
        lower = int(higher.group("lower"))
        return {"label": f"{lower}F or higher", "lower_f": lower, "upper_f": None}
    range_match = re.search(r"(?P<lower>\d{2,3})\s*[-–]\s*(?P<upper>\d{2,3})\s*(?:degrees\s*)?f?", normalized, flags=re.IGNORECASE)
    if range_match:
        lower = int(range_match.group("lower"))
        upper = int(range_match.group("upper"))
        return {"label": f"{lower}-{upper}F", "lower_f": lower, "upper_f": upper}
    return {"label": text.strip() or "unknown", "lower_f": None, "upper_f": None}


def token_for_outcome(outcomes: Any, token_ids: Any, desired: str) -> str | None:
    if not isinstance(outcomes, list) or not isinstance(token_ids, list):
        return None
    for index, outcome in enumerate(outcomes):
        if str(outcome).strip().lower() == desired.lower() and index < len(token_ids):
            return str(token_ids[index])
    return str(token_ids[0]) if token_ids else None


def parse_jsonish(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


def event_target_date(event: dict[str, Any]) -> date | None:
    end_dt = parse_datetime(event.get("endDate"))
    if end_dt is not None:
        return end_dt.date()
    slug = str(event.get("slug", ""))
    match = re.search(r"on-([a-z]+)-(\d{1,2})-(\d{4})", slug, flags=re.IGNORECASE)
    if match:
        return datetime.strptime("-".join(match.groups()), "%B-%d-%Y").date()
    return None


def is_nyc_tmax_event(event: dict[str, Any], *, start_date: date, end_date: date) -> bool:
    title = str(event.get("title") or "")
    slug = str(event.get("slug") or "")
    target_date = event_target_date(event)
    if target_date is None or not (start_date <= target_date <= end_date):
        return False
    haystack = f"{title} {slug}".lower()
    return "highest temperature in nyc" in haystack or slug.startswith("highest-temperature-in-nyc-on-")


def parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def parse_pandas_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).tz_convert("UTC") if pd.Timestamp(value).tzinfo else pd.Timestamp(value, tz="UTC")


def first_present(*values: Any) -> str | None:
    for value in values:
        if value is not None and str(value).strip():
            return str(value)
    return None


def first_valid_index(mask: pd.Series) -> pd.Timestamp | None:
    valid = mask[mask.fillna(False)]
    return valid.index[0] if not valid.empty else None


def first_sustained_index(mask: pd.Series) -> pd.Timestamp | None:
    sustained = mask.fillna(False).astype(int).rolling(SUSTAINED_PERIODS).sum() >= SUSTAINED_PERIODS
    valid = sustained[sustained]
    return valid.index[0] if not valid.empty else None


def hours_before(reference: datetime, value: pd.Timestamp | datetime | None) -> float | None:
    if value is None:
        return None
    value_dt = to_datetime_utc(value)
    return (reference - value_dt).total_seconds() / 3600.0


def iso_or_none(value: pd.Timestamp | datetime | None) -> str | None:
    return None if value is None else to_datetime_utc(value).isoformat()


def to_datetime_utc(value: pd.Timestamp | datetime) -> datetime:
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            return value.to_pydatetime().replace(tzinfo=UTC)
        return value.tz_convert("UTC").to_pydatetime()
    return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)


def chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def grouped_by_window(token_records: list[dict[str, Any]]) -> dict[tuple[str, int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for record in token_records:
        key = (str(record["event_id"]), int(record["price_start_ts"]), int(record["price_end_ts"]))
        grouped.setdefault(key, []).append(record)
    return grouped


def with_price_window(record: dict[str, Any]) -> dict[str, Any]:
    updated = dict(record)
    target_date = date.fromisoformat(str(record["target_date"]))
    event_start = parse_datetime(record.get("event_start_date_utc"))
    event_end = parse_datetime(record.get("event_end_date_utc"))
    if event_start is None:
        event_start = datetime.combine(target_date - timedelta(days=3), dt_time(0, 0), tzinfo=UTC)
    if event_end is None:
        event_end = datetime.combine(target_date + timedelta(days=1), dt_time(23, 59), tzinfo=UTC)
    price_start = event_start - timedelta(hours=2)
    price_end = max(event_end + timedelta(hours=36), datetime.combine(target_date + timedelta(days=2), dt_time(0, 0), tzinfo=UTC))
    updated["price_start_ts"] = int(price_start.timestamp())
    updated["price_end_ts"] = int(price_end.timestamp())
    return updated


def stable_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def count_path(payload: Any, path: tuple[str, ...]) -> int:
    current = payload
    for part in path:
        if isinstance(current, dict):
            current = current.get(part, {})
        else:
            return 0
    if isinstance(current, list):
        return len(current)
    if isinstance(current, dict):
        return sum(len(value) if isinstance(value, list) else 1 for value in current.values())
    return 0


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str, allow_nan=False))


def format_sensitivity_rows(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in rows:
        if not row.get("selected_candidate_id"):
            lines.append(f"| {row.get('guardrail'):.2f} | none | n/a | n/a | n/a |")
            continue
        lines.append(
            "| {guardrail:.2f} | `{candidate}` | {pre:.3f} | {model:.3f} | {move:.3f} |".format(
                guardrail=float(row["guardrail"]),
                candidate=row["selected_candidate_id"],
                pre=float(row["pre_explosion_rate"]),
                model=float(row["model_score_normalized"]),
                move=float(row["median_remaining_move"]),
            )
        )
    return "\n".join(lines)


def format_top_candidate_rows(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in rows:
        lines.append(
            "| `{candidate}` | {tradable:.3f} | {pre:.3f} | {model:.3f} | {move:.3f} |".format(
                candidate=row["candidate_id"],
                tradable=float(row["tradable_rate"]),
                pre=float(row["pre_explosion_rate"]),
                model=float(row["model_score_normalized"]),
                move=float(row["median_remaining_move"]),
            )
        )
    return "\n".join(lines)
