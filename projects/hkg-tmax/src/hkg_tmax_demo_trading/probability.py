"""B4 probability snapshot adapter for HKG Tmax demo trading."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import yaml
from hkg_tmax_probability.bucket_rules import BUCKET_KEYS
from hkg_tmax_probability.data_build import build_modeling_table
from hkg_tmax_probability.models import hierarchical_month_forecast_pmf_predict, select_b4_alphas

from .domain import HKT, UTC

HKO_FLW_URL = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=flw&lang=en"
HKO_FND_URL = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=fnd&lang=en"
DEFAULT_DATABASE_URL = "postgresql://postgres:root@127.0.0.1:5432/hkg_tmax_research"
LOCAL_FORECAST_SOURCE = "HKO flw local forecast"
FND_FORECAST_SOURCE = "HKO fnd 9-day forecast"

MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

DEFAULT_VALIDATED_PROFILE = "t_minus_1_2359_hkt"

VALIDATED_PROFILE_ORDER = [
    "t_minus_1_1800_hkt",
    "t_minus_1_1900_hkt",
    "t_minus_1_2000_hkt",
    "t_minus_1_2100_hkt",
    "t_minus_1_2200_hkt",
    "t_minus_1_2300_hkt",
    "t_minus_1_2359_hkt",
]

PROFILE_ALIASES = {
    "H24N": DEFAULT_VALIDATED_PROFILE,
    "h24n": DEFAULT_VALIDATED_PROFILE,
    "tminus1_2359": DEFAULT_VALIDATED_PROFILE,
    "t_minus_1_2359": DEFAULT_VALIDATED_PROFILE,
}

PROFILE_METRICS: dict[str, dict[str, Any]] = {
    "t_minus_1_1800_hkt": {
        "label": "12:00 Stockholm",
        "stockholm_entry": "12:00",
        "hkt_cutoff": "18:00",
        "quality_score": 62,
        "operational": {"n": 5050, "raw_mae": 0.9648, "b4_rps": 0.04479, "b4_nll": 1.11268},
        "strict_common": {"n": 5050, "raw_mae": 0.9648, "b4_rps": 0.04479},
        "risk_label": "strict threshold only",
    },
    "t_minus_1_1900_hkt": {
        "label": "13:00 Stockholm",
        "stockholm_entry": "13:00",
        "hkt_cutoff": "19:00",
        "quality_score": 63,
        "operational": {"n": 5056, "raw_mae": 0.9645, "b4_rps": 0.04479, "b4_nll": 1.11274},
        "strict_common": {"n": 5050, "raw_mae": 0.9640, "b4_rps": 0.04474},
        "risk_label": "strict threshold only",
    },
    "t_minus_1_2000_hkt": {
        "label": "14:00 Stockholm",
        "stockholm_entry": "14:00",
        "hkt_cutoff": "20:00",
        "quality_score": 65,
        "operational": {"n": 5086, "raw_mae": 0.9599, "b4_rps": 0.04442, "b4_nll": 1.10603},
        "strict_common": {"n": 5050, "raw_mae": 0.9605, "b4_rps": 0.04447},
        "risk_label": "strict threshold only",
    },
    "t_minus_1_2100_hkt": {
        "label": "15:00 Stockholm",
        "stockholm_entry": "15:00",
        "hkt_cutoff": "21:00",
        "quality_score": 67,
        "operational": {"n": 5088, "raw_mae": 0.9595, "b4_rps": 0.04442, "b4_nll": 1.10580},
        "strict_common": {"n": 5050, "raw_mae": 0.9600, "b4_rps": 0.04446},
        "risk_label": "acceptable with strict edge",
    },
    "t_minus_1_2200_hkt": {
        "label": "16:00 Stockholm",
        "stockholm_entry": "16:00",
        "hkt_cutoff": "22:00",
        "quality_score": 68,
        "operational": {"n": 5092, "raw_mae": 0.9584, "b4_rps": 0.04433, "b4_nll": 1.10470},
        "strict_common": {"n": 5050, "raw_mae": 0.9591, "b4_rps": 0.04440},
        "risk_label": "acceptable with strict edge",
    },
    "t_minus_1_2300_hkt": {
        "label": "17:00 Stockholm",
        "stockholm_entry": "17:00",
        "hkt_cutoff": "23:00",
        "quality_score": 70,
        "operational": {"n": 5099, "raw_mae": 0.9566, "b4_rps": 0.04424, "b4_nll": 1.10328},
        "strict_common": {"n": 5050, "raw_mae": 0.9578, "b4_rps": 0.04435},
        "risk_label": "preferred late-window",
    },
    "t_minus_1_2359_hkt": {
        "label": "17:59 Stockholm",
        "stockholm_entry": "17:59",
        "hkt_cutoff": "23:59",
        "quality_score": 82,
        "operational": {"n": 5629, "raw_mae": 0.9309, "b4_rps": 0.04193, "b4_nll": 1.05096},
        "strict_common": {"n": 5050, "raw_mae": 0.9444, "b4_rps": 0.04401},
        "risk_label": "strongest validated profile",
    },
}


class ForecastUnavailable(RuntimeError):
    """Raised when no leakage-safe forecast anchor is available."""


@dataclass(frozen=True)
class ForecastSnapshot:
    source: str
    update_time_hkt: datetime
    target_date: date
    forecast_min_c: float | None
    forecast_max_c: float
    as_of_profile: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class ProbabilitySnapshot:
    forecast: ForecastSnapshot
    model: dict[str, Any]


def normalize_as_of_profile(as_of_profile: str | None) -> str:
    profile = (as_of_profile or "").strip()
    if not profile:
        return ""
    profile = PROFILE_ALIASES.get(profile, profile)
    if profile == "live_now" or profile in PROFILE_METRICS:
        return profile
    raise ForecastUnavailable(f"Unsupported as_of_profile: {as_of_profile}")


def profile_metadata(as_of_profile: str | None) -> dict[str, Any]:
    try:
        profile = normalize_as_of_profile(as_of_profile) or DEFAULT_VALIDATED_PROFILE
    except ForecastUnavailable:
        profile = str(as_of_profile or "unknown")
        return {
            "id": profile,
            "label": profile,
            "stockholmEntry": "unknown",
            "hktCutoff": "unknown",
            "validationStatus": "legacy_unknown",
            "tradeable": False,
            "leakageSafe": False,
            "sourceContract": "Unknown or legacy snapshot profile",
            "asOfRule": "This snapshot cannot be used for apples-to-apples trading until refreshed with a validated profile.",
            "warning": "Unsupported profile id; display only.",
            "qualityScore": None,
            "operational": None,
            "strictCommon": None,
        }
    if profile == "live_now":
        return {
            "id": "live_now",
            "label": "Live exploratory",
            "stockholmEntry": "now",
            "hktCutoff": "live",
            "validationStatus": "not_apples_to_apples",
            "tradeable": False,
            "leakageSafe": False,
            "sourceContract": "HKO live flw/fnd API; not the validated Info.gov lead-1 cutoff row set",
            "asOfRule": "Exploratory live forecast only; do not use for 12-18 Stockholm apples-to-apples trades.",
            "warning": "Live 9-day/local API snapshots are not the same row family used in the cutoff backtests.",
            "qualityScore": None,
            "operational": None,
            "strictCommon": None,
        }
    metrics = PROFILE_METRICS[profile]
    return {
        "id": profile,
        "label": metrics["label"],
        "stockholmEntry": metrics["stockholm_entry"],
        "hktCutoff": metrics["hkt_cutoff"],
        "validationStatus": "validated_apples_to_apples",
        "tradeable": True,
        "leakageSafe": True,
        "sourceContract": "Info.gov LOCAL WEATHER FORECAST, target T, issue_at <= T-1 cutoff",
        "asOfRule": f"Use latest eligible lead-1 local forecast issued at or before T-1 {metrics['hkt_cutoff']} HKT.",
        "warning": metrics["risk_label"],
        "qualityScore": metrics["quality_score"],
        "operational": metrics["operational"],
        "strictCommon": metrics["strict_common"],
    }


def profile_catalog() -> list[dict[str, Any]]:
    return [profile_metadata(profile) for profile in VALIDATED_PROFILE_ORDER] + [profile_metadata("live_now")]


def cutoff_utc_for_profile(target_date: date, as_of_profile: str) -> datetime:
    profile = normalize_as_of_profile(as_of_profile)
    if profile not in PROFILE_METRICS:
        raise ForecastUnavailable(f"Profile {as_of_profile} does not have a fixed HKT cutoff")
    hkt_cutoff = PROFILE_METRICS[profile]["hkt_cutoff"]
    hour, minute = [int(part) for part in hkt_cutoff.split(":")]
    cutoff_hkt = datetime.combine(
        target_date - timedelta(days=1),
        time(hour=hour, minute=minute),
        tzinfo=HKT,
    )
    return cutoff_hkt.astimezone(UTC)


def _fetch_json(url: str) -> Any:
    headers = {"User-Agent": "HKG-Tmax-Demo-Backtester/0.1", "Accept": "application/json"}
    with httpx.Client(timeout=20.0, follow_redirects=True, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


def _parse_hko_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(HKT)


def _parse_flw_period_date(value: str) -> date | None:
    match = re.search(r"\((?:[A-Za-z]+,\s*)?(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\)", value)
    if not match:
        return None
    return date(int(match.group(3)), MONTHS[match.group(2).lower()], int(match.group(1)))


def _infer_flw_target_date(forecast_period: str, update_time_hkt: datetime) -> date | None:
    explicit = _parse_flw_period_date(forecast_period)
    if explicit is not None:
        return explicit
    normalized = forecast_period.lower()
    if "tomorrow" in normalized:
        return update_time_hkt.date() + timedelta(days=1)
    if "today" in normalized:
        return update_time_hkt.date()
    return None


def _parse_flw_temperatures(description: str) -> tuple[float | None, float | None]:
    min_c = None
    max_c = None
    range_match = re.search(
        r"temperatures?\s+(?:will\s+)?(?:range|ranging)\s+between\s+"
        r"(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s+degrees",
        description,
        re.I,
    )
    if range_match:
        return float(range_match.group(1)), float(range_match.group(2))
    min_match = re.search(
        r"minimum temperature (?:will be )?(?:is )?(?:about|around)?\s*(\d+(?:\.\d+)?)\s*degrees",
        description,
        re.I,
    )
    if min_match:
        min_c = float(min_match.group(1))
    max_match = re.search(
        r"maximum temperature (?:will be )?(?:is )?(?:of )?(?:about|around)?\s*(\d+(?:\.\d+)?)\s*degrees",
        description,
        re.I,
    )
    if max_match:
        max_c = float(max_match.group(1))
    return min_c, max_c


def _forecast_candidate_sort_key(candidate: ForecastSnapshot) -> tuple[datetime, int]:
    local_forecast_priority = 1 if candidate.source == LOCAL_FORECAST_SOURCE else 0
    return candidate.update_time_hkt, local_forecast_priority


def _live_hko_forecast_candidates(target_date: date, as_of_profile: str = "live_now") -> list[ForecastSnapshot]:
    candidates: list[ForecastSnapshot] = []
    try:
        flw = _fetch_json(HKO_FLW_URL)
        update_time_hkt = _parse_hko_time(str(flw["updateTime"]))
        flw_date = _infer_flw_target_date(str(flw.get("forecastPeriod", "")), update_time_hkt)
        if flw_date == target_date:
            min_c, max_c = _parse_flw_temperatures(str(flw.get("forecastDesc", "")))
            if max_c is not None:
                candidates.append(
                    ForecastSnapshot(
                        source=LOCAL_FORECAST_SOURCE,
                        update_time_hkt=update_time_hkt,
                        target_date=target_date,
                        forecast_min_c=min_c,
                        forecast_max_c=max_c,
                        as_of_profile=as_of_profile,
                        raw=flw,
                    )
                )
    except Exception:
        pass

    try:
        fnd = _fetch_json(HKO_FND_URL)
        for item in fnd.get("weatherForecast", []):
            forecast_date = datetime.strptime(str(item["forecastDate"]), "%Y%m%d").date()
            if forecast_date != target_date:
                continue
            max_c = item.get("forecastMaxtemp", {}).get("value")
            if max_c is None:
                continue
            min_c = item.get("forecastMintemp", {}).get("value")
            candidates.append(
                ForecastSnapshot(
                    source=FND_FORECAST_SOURCE,
                    update_time_hkt=_parse_hko_time(str(fnd["updateTime"])),
                    target_date=target_date,
                    forecast_min_c=None if min_c is None else float(min_c),
                    forecast_max_c=float(max_c),
                    as_of_profile=as_of_profile,
                    raw={
                        "source_payload": item,
                        "updateTime": fnd.get("updateTime"),
                        "generalSituation": fnd.get("generalSituation"),
                    },
                )
            )
    except Exception:
        pass

    return candidates


def latest_hko_forecast(target_date: date) -> ForecastSnapshot:
    candidates = _live_hko_forecast_candidates(target_date, "live_now")
    if not candidates:
        raise ForecastUnavailable(f"No HKO flw/fnd forecast candidate found for {target_date.isoformat()}")
    return sorted(candidates, key=_forecast_candidate_sort_key)[-1]


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _today_hkt() -> date:
    return _now_utc().astimezone(HKT).date()


def live_cutoff_forecast(target_date: date, as_of_profile: str) -> ForecastSnapshot:
    profile = normalize_as_of_profile(as_of_profile)
    cutoff_utc = cutoff_utc_for_profile(target_date, profile)
    now_utc = _now_utc()
    if cutoff_utc > now_utc:
        raise ForecastUnavailable(
            f"Validated profile {profile} is not available yet; cutoff is {cutoff_utc.isoformat()}"
        )
    if target_date < _today_hkt():
        raise ForecastUnavailable(
            "Live HKO fallback is only allowed until the target date in HKT; "
            f"{target_date.isoformat()} is already past in HKT"
        )

    candidates = [
        candidate
        for candidate in _live_hko_forecast_candidates(target_date, profile)
        if candidate.source == LOCAL_FORECAST_SOURCE
    ]
    eligible = [
        candidate
        for candidate in candidates
        if candidate.update_time_hkt.astimezone(UTC) <= cutoff_utc
    ]
    if not eligible:
        latest_update = (
            max((candidate.update_time_hkt.astimezone(UTC) for candidate in candidates), default=None)
        )
        latest_text = "none" if latest_update is None else latest_update.isoformat()
        raise ForecastUnavailable(
            "No live HKO local forecast update is eligible for "
            f"{target_date.isoformat()} profile {profile} before {cutoff_utc.isoformat()} "
            f"(latest live update: {latest_text})"
        )

    selected = sorted(eligible, key=_forecast_candidate_sort_key)[-1]
    return ForecastSnapshot(
        source=f"{selected.source} live cutoff fetch",
        update_time_hkt=selected.update_time_hkt,
        target_date=selected.target_date,
        forecast_min_c=selected.forecast_min_c,
        forecast_max_c=selected.forecast_max_c,
        as_of_profile=profile,
        raw={
            "live_cutoff_fetch": True,
            "cutoff_utc": cutoff_utc,
            "selected_update_utc": selected.update_time_hkt.astimezone(UTC),
            "profile": profile_metadata(profile),
            "source_payload": selected.raw,
        },
    )


def _stored_live_forecast_sql() -> str:
    return """
        SELECT
            id,
            as_of_profile,
            forecast_source,
            forecast_update_time_hkt,
            forecast_min_c,
            forecast_max_c,
            snapshot_json->'forecast' AS forecast
        FROM demo_trading.market_snapshot
        WHERE target_date = %(target_date)s
          AND status = 'ok'
          AND forecast_source = %(forecast_source)s
          AND forecast_update_time_hkt IS NOT NULL
          AND forecast_update_time_hkt <= %(cutoff_utc)s
          AND forecast_max_c IS NOT NULL
        ORDER BY forecast_update_time_hkt DESC, created_at_utc DESC, id DESC
        LIMIT 1
    """


def stored_live_cutoff_forecast(connection: Any, target_date: date, as_of_profile: str) -> ForecastSnapshot:
    profile = normalize_as_of_profile(as_of_profile)
    cutoff_utc = cutoff_utc_for_profile(target_date, profile)
    if cutoff_utc > _now_utc():
        raise ForecastUnavailable(
            f"Validated profile {profile} is not available yet; cutoff is {cutoff_utc.isoformat()}"
        )

    with connection.cursor() as cursor:
        cursor.execute(
            _stored_live_forecast_sql(),
            {
                "target_date": target_date,
                "cutoff_utc": cutoff_utc,
                "forecast_source": f"{LOCAL_FORECAST_SOURCE} live cutoff fetch",
            },
        )
        row = cursor.fetchone()
    if not row:
        raise ForecastUnavailable(
            "No stored live HKO local forecast snapshot is eligible for "
            f"{target_date.isoformat()} profile {profile} before {cutoff_utc.isoformat()}"
        )

    update_time = row["forecast_update_time_hkt"]
    if update_time.tzinfo is None:
        update_time = update_time.replace(tzinfo=UTC)
    forecast_payload = row["forecast"] if isinstance(row.get("forecast"), dict) else {}
    raw = forecast_payload.get("raw") if isinstance(forecast_payload, dict) else {}
    raw = dict(raw) if isinstance(raw, dict) else {}
    source_cutoff_utc = raw.get("cutoff_utc")

    return ForecastSnapshot(
        source=f"{LOCAL_FORECAST_SOURCE} live cutoff fetch",
        update_time_hkt=update_time.astimezone(HKT),
        target_date=target_date,
        forecast_min_c=None if row["forecast_min_c"] is None else float(row["forecast_min_c"]),
        forecast_max_c=float(row["forecast_max_c"]),
        as_of_profile=profile,
        raw={
            **raw,
            "live_cutoff_fetch": True,
            "stored_cutoff_reuse": True,
            "stored_snapshot_id": row["id"],
            "stored_snapshot_profile": row["as_of_profile"],
            "stored_source_cutoff_utc": source_cutoff_utc,
            "cutoff_utc": cutoff_utc,
            "selected_update_utc": update_time.astimezone(UTC),
            "profile": profile_metadata(profile),
        },
    )


def _historical_forecast_sql() -> str:
    return """
        SELECT
            source,
            product_type,
            title,
            issue_at_hkt,
            issue_at_utc,
            target_date,
            forecast_min_c,
            forecast_max_c,
            forecast_range_c,
            row_quality_status,
            raw_sha256,
            source_url
        FROM public.hko_historical_forecasts_2000_2026
        WHERE target_date = %(target_date)s
          AND issue_at_utc <= %(cutoff_utc)s
          AND usable_local_tmax_forecast
          AND forecast_max_c IS NOT NULL
        ORDER BY issue_at_utc DESC
        LIMIT 1
    """


def historical_cutoff_forecast(connection: Any, target_date: date, as_of_profile: str) -> ForecastSnapshot:
    profile = normalize_as_of_profile(as_of_profile)
    cutoff_utc = cutoff_utc_for_profile(target_date, profile)
    with connection.cursor() as cursor:
        cursor.execute(
            _historical_forecast_sql(),
            {"target_date": target_date, "cutoff_utc": cutoff_utc},
        )
        row = cursor.fetchone()
    if not row:
        try:
            return live_cutoff_forecast(target_date, profile)
        except ForecastUnavailable as live_exc:
            try:
                return stored_live_cutoff_forecast(connection, target_date, profile)
            except ForecastUnavailable as stored_exc:
                raise ForecastUnavailable(
                    "No validated Info.gov local forecast anchor found for "
                    f"{target_date.isoformat()} profile {profile} before {cutoff_utc.isoformat()}; "
                    f"live fallback unavailable: {live_exc}; "
                    f"stored fallback unavailable: {stored_exc}"
                ) from stored_exc
    issue_at_utc = row["issue_at_utc"]
    if issue_at_utc.tzinfo is None:
        issue_at_utc = issue_at_utc.replace(tzinfo=UTC)
    return ForecastSnapshot(
        source="public.hko_historical_forecasts_2000_2026",
        update_time_hkt=issue_at_utc.astimezone(HKT),
        target_date=target_date,
        forecast_min_c=None if row["forecast_min_c"] is None else float(row["forecast_min_c"]),
        forecast_max_c=float(row["forecast_max_c"]),
        as_of_profile=profile,
        raw={
            **dict(row),
            "cutoff_utc": cutoff_utc,
            "profile": profile_metadata(profile),
        },
    )


def choose_forecast(connection: Any, target_date: date, as_of_profile: str | None = None) -> ForecastSnapshot:
    profile = normalize_as_of_profile(as_of_profile)
    if not profile:
        profile = "live_now" if target_date >= _today_hkt() else DEFAULT_VALIDATED_PROFILE
    if profile == "live_now":
        return latest_hko_forecast(target_date)
    return historical_cutoff_forecast(connection, target_date, profile)


def _decimal_tenths(value: float) -> int:
    quantized = Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return int(quantized * 10)


def _decimal_round_int(value: float) -> int:
    return int(Decimal(str(value)).to_integral_value(rounding=ROUND_HALF_UP))


@lru_cache(maxsize=4)
def _modeling_table(repo_root: str, database_url: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(repo_root)
    config_path = root / "config" / "experiments" / "hkg_tmax" / "probability_bucket_v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    modeling, _selected, _eligible, row_audit = build_modeling_table(config, database_url=database_url)
    modeling["target_date"] = pd.to_datetime(modeling["target_date"])
    return modeling, row_audit


def compute_b4_probabilities(
    *,
    repo_root: Path,
    database_url: str,
    target_date: date,
    forecast_min_c: float | None,
    forecast_max_c: float,
    cutoff_profile: str | None = None,
) -> dict[str, Any]:
    requested_profile = normalize_as_of_profile(cutoff_profile)
    model_profile = DEFAULT_VALIDATED_PROFILE if requested_profile in {"", "live_now"} else requested_profile
    modeling, row_audit = _modeling_table(str(repo_root.resolve()), database_url)
    if "cutoff_profile" in modeling.columns:
        train = modeling[
            (modeling["cutoff_profile"] == model_profile)
            & (modeling["target_date"] < pd.Timestamp(target_date))
        ].copy()
    else:
        train = modeling[
            (modeling["is_primary_cutoff"]) & (modeling["target_date"] < pd.Timestamp(target_date))
        ].copy()
    if train.empty:
        raise ForecastUnavailable(
            f"No B4 training rows for {model_profile} before {target_date.isoformat()}"
        )
    train = train.reset_index(drop=True)
    config_path = repo_root / "config" / "experiments" / "hkg_tmax" / "probability_bucket_v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    month_alpha, cell_alpha, alpha_details = select_b4_alphas(train, config)
    validation = pd.DataFrame(
        [
            {
                "target_date": pd.Timestamp(target_date),
                "target_month": target_date.month,
                "forecast_max_c": float(forecast_max_c),
                "forecast_min_c": None if forecast_min_c is None else float(forecast_min_c),
                "forecast_range_c": None if forecast_min_c is None else float(forecast_max_c - forecast_min_c),
                "forecast_max_tenths": _decimal_tenths(forecast_max_c),
                "official_max_round": _decimal_round_int(forecast_max_c),
            }
        ]
    )
    probabilities = hierarchical_month_forecast_pmf_predict(train, validation, month_alpha, cell_alpha)[0]
    return {
        "method": "B4_hierarchical_residual_pmf",
        "bucket_keys": list(BUCKET_KEYS),
        "probabilities": {bucket: float(prob) for bucket, prob in zip(BUCKET_KEYS, probabilities, strict=True)},
        "train_rows": int(len(train)),
        "train_start": str(train["target_date"].min().date()),
        "train_end": str(train["target_date"].max().date()),
        "month_alpha": float(month_alpha),
        "cell_alpha": float(cell_alpha),
        "alpha_selection": alpha_details,
        "row_audit": row_audit,
        "cutoff_profile": model_profile,
        "profile": profile_metadata(model_profile),
        "live_forecast_profile": None if requested_profile != "live_now" else profile_metadata("live_now"),
    }


def build_probability_snapshot(
    *,
    connection: Any,
    repo_root: Path,
    database_url: str,
    target_date: date,
    as_of_profile: str | None = None,
) -> ProbabilitySnapshot:
    forecast = choose_forecast(connection, target_date, as_of_profile)
    model = compute_b4_probabilities(
        repo_root=repo_root,
        database_url=database_url,
        target_date=target_date,
        forecast_min_c=forecast.forecast_min_c,
        forecast_max_c=forecast.forecast_max_c,
        cutoff_profile=forecast.as_of_profile,
    )
    return ProbabilitySnapshot(forecast=forecast, model=model)
