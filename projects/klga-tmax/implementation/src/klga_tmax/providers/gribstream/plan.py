from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from klga_tmax.constants import TARGET_TZ
from klga_tmax.ingestion.hash_keys import canonical_json, sha256_hex
from klga_tmax.providers.gribstream.models import (
    GribStreamChunk,
    GribStreamJobPlan,
    GribStreamModelSpec,
    ResolvedSelector,
)
from klga_tmax.registry.materialize_targets import iter_dates
from klga_tmax.registry.station_universe import coordinate_tier


DEFAULT_CUTOFF_ID = "T_MINUS_1_2045UTC"
T1245_CUTOFF_ID = "T_1245UTC"
CUTOFF_ID = DEFAULT_CUTOFF_ID
CUTOFF_UTC_TIME = "20:45:00"
DEFAULT_END_DATE = date(2026, 6, 28)
REQUEST_ENDPOINT = "/timeseries"
RUNS_REQUEST_ENDPOINT = "/runs"
TMAX_THIN_FEATURE_PROFILE = "TMAX_THIN_V1"
TMAX_THIN_PERSISTENCE_MODE = "gold_only"
SILVER_ATOMIC_PERSISTENCE_MODE = "silver_atomic"
TMAX_THIN_JOB_ID = "klga_t1245utc_tmax_thin_backfill_v1"


@dataclass(frozen=True)
class GribStreamCutoffProfile:
    cutoff_id: str
    cutoff_utc_time: time
    target_day_offset: int
    archive_start_offset_days: int
    nbmqmd_max18_valid_hour_utc: int

    @property
    def cutoff_utc_time_label(self) -> str:
        return self.cutoff_utc_time.isoformat()


CUTOFF_PROFILES: tuple[GribStreamCutoffProfile, ...] = (
    GribStreamCutoffProfile(
        cutoff_id=DEFAULT_CUTOFF_ID,
        cutoff_utc_time=time(20, 45),
        target_day_offset=-1,
        archive_start_offset_days=1,
        nbmqmd_max18_valid_hour_utc=0,
    ),
    GribStreamCutoffProfile(
        cutoff_id=T1245_CUTOFF_ID,
        cutoff_utc_time=time(12, 45),
        target_day_offset=0,
        archive_start_offset_days=0,
        nbmqmd_max18_valid_hour_utc=6,
    ),
)

CUTOFF_PROFILE_BY_ID = {profile.cutoff_id: profile for profile in CUTOFF_PROFILES}


MODEL_SPECS: tuple[GribStreamModelSpec, ...] = (
    GribStreamModelSpec("T1", "hrrr", date(2014, 7, 30), "hourly_peak", 80, 348080, "hourly_8", timedelta(hours=2, minutes=15), "18Z T-1 extended", default_chunk_days=31),
    GribStreamModelSpec("T1", "rtma", date(2018, 1, 1), "rtma_latest", 4, 12400, "rtma_4", timedelta(hours=1), "latest safe analysis", default_chunk_days=62),
    GribStreamModelSpec("T1", "urma", date(2024, 4, 30), "urma_peak_temp", 10, 7890, "temp_only", None, "retrospective target-day support only", default_chunk_days=62),
    GribStreamModelSpec("T1", "nbm", date(2020, 9, 29), "hourly_peak", 80, 167840, "nbm_8", timedelta(hours=1, minutes=45), "latest safe hourly", default_chunk_days=31),
    GribStreamModelSpec("T1", "gfs", date(2021, 3, 22), "hourly_peak", 80, 153920, "hourly_8", timedelta(hours=4), "12Z T-1", default_chunk_days=31),
    GribStreamModelSpec("T1", "rap", date(2021, 2, 22), "hourly_peak", 80, 156160, "hourly_8", timedelta(hours=1, minutes=45), "19Z if available, else 18Z", default_chunk_days=31),
    GribStreamModelSpec("T1", "gefsatmosmean", date(2020, 10, 1), "synoptic", 2, 4192, "temp_only", timedelta(hours=4), "12Z T-1", default_chunk_days=62),
    GribStreamModelSpec("T1", "gefsatmos", date(2020, 10, 1), "synoptic", 62, 129952, "temp_only", timedelta(hours=4), "12Z T-1", member_mode="gefs_31", expected_members=31, default_chunk_days=14),
    GribStreamModelSpec("T1", "ifsoper", date(2024, 2, 28), "synoptic", 14, 11914, "ecmwf_7", timedelta(hours=3), "12Z T-1", default_chunk_days=31),
    GribStreamModelSpec("T1", "ifsenfo", date(2024, 3, 1), "synoptic", 102, 86598, "temp_only", timedelta(hours=3), "12Z T-1", member_mode="ecmwf_51", expected_members=51, default_chunk_days=10),
    GribStreamModelSpec("T2", "nbmqmd", date(2026, 1, 31), "nbmqmd_max18", 21, 3108, "nbmqmd_percentiles", timedelta(hours=1, minutes=45), "latest safe hourly", default_chunk_days=31),
    GribStreamModelSpec("T2", "aifsoper", date(2025, 2, 25), "synoptic", 14, 6832, "ecmwf_7", timedelta(hours=3, minutes=30), "12Z T-1", default_chunk_days=31),
    GribStreamModelSpec("T2", "aifsenfo", date(2025, 7, 2), "synoptic", 102, 36822, "temp_only", timedelta(hours=3, minutes=30), "12Z T-1", member_mode="ecmwf_51", expected_members=51, default_chunk_days=10),
    GribStreamModelSpec("T2", "aigefssfc", date(2025, 6, 1), "synoptic", 62, 24304, "temp_only", timedelta(hours=4), "12Z T-1", member_mode="gefs_31", expected_members=31, default_chunk_days=14),
    GribStreamModelSpec("T2", "aigfssfc", date(2026, 4, 16), "synoptic", 2, 146, "temp_only", timedelta(hours=4), "12Z T-1", default_chunk_days=62),
)

TMAX_THIN_MODEL_SPECS: tuple[GribStreamModelSpec, ...] = (
    GribStreamModelSpec("T1", "hrrr", date(2014, 7, 30), "hourly_peak", 10, 43520, "temperature_peak_only", timedelta(hours=2, minutes=15), "T_1245UTC 08Z/10Z HRRR peak-window temperature", default_chunk_days=62),
    GribStreamModelSpec("T1", "rtma", date(2018, 1, 1), "rtma_latest", 3, 9303, "rtma_current_state_thin", timedelta(hours=1), "latest safe current-state analysis near cutoff", default_chunk_days=180),
    GribStreamModelSpec("T1", "nbm", date(2020, 9, 29), "hourly_peak", 10, 20990, "temperature_peak_only", timedelta(hours=1, minutes=45), "fallback NBM 2m-temperature peak curve after native Tmax empty pilot", default_chunk_days=62),
    GribStreamModelSpec("T1", "gfs", date(2021, 3, 22), "hourly_peak", 10, 19250, "temperature_peak_only", timedelta(hours=4), "06Z/12Z-safe GFS temperature peak curve", default_chunk_days=62),
    GribStreamModelSpec("T1", "rap", date(2021, 2, 22), "hourly_peak", 10, 19530, "temperature_peak_only", timedelta(hours=1, minutes=45), "08Z RAP peak-window temperature", default_chunk_days=62),
    GribStreamModelSpec("T1", "gefsatmosmean", date(2020, 10, 1), "synoptic", 2, 4194, "ensemble_temperature_only", timedelta(hours=4), "06Z GEFS mean synoptic temperature", default_chunk_days=180),
    GribStreamModelSpec("T1", "gefsatmos", date(2020, 10, 1), "synoptic", 62, 130014, "ensemble_temperature_only", timedelta(hours=4), "06Z GEFS member synoptic temperature", member_mode="gefs_31", expected_members=31, default_chunk_days=45),
    GribStreamModelSpec("T1", "ifsoper", date(2024, 2, 28), "synoptic", 2, 1704, "ecmwf_temperature_only", timedelta(hours=3), "00Z IFS synoptic temperature", default_chunk_days=90),
    GribStreamModelSpec("T1", "ifsenfo", date(2024, 3, 1), "synoptic", 102, 86700, "ecmwf_temperature_only", timedelta(hours=3), "00Z ENS member synoptic temperature", member_mode="ecmwf_51", expected_members=51, default_chunk_days=31),
    GribStreamModelSpec("T2", "nbmqmd", date(2026, 1, 31), "nbmqmd_max18", 21, 3129, "nbmqmd_percentiles", timedelta(hours=1, minutes=45), "QMD max-18h percentile package", default_chunk_days=180),
    GribStreamModelSpec("T2", "aifsoper", date(2025, 2, 25), "synoptic", 2, 978, "ecmwf_temperature_only", timedelta(hours=3, minutes=30), "00Z AIFS deterministic temperature", default_chunk_days=90),
    GribStreamModelSpec("T2", "aifsenfo", date(2025, 7, 2), "synoptic", 102, 36924, "ecmwf_temperature_only", timedelta(hours=3, minutes=30), "00Z AIFS ensemble member temperature", member_mode="ecmwf_51", expected_members=51, default_chunk_days=31),
    GribStreamModelSpec("T2", "aigefssfc", date(2025, 6, 1), "synoptic", 62, 24366, "ensemble_temperature_only", timedelta(hours=4), "06Z AI-GEFS member synoptic temperature", member_mode="gefs_31", expected_members=31, default_chunk_days=45),
    GribStreamModelSpec("T2", "aigfssfc", date(2026, 4, 16), "synoptic", 2, 148, "ensemble_temperature_only", timedelta(hours=4), "06Z AI-GFS synoptic temperature", default_chunk_days=180),
)

TMAX_THIN_COORDINATE_TIER_BY_MODEL: dict[str, str] = {
    "hrrr": "B",
    "rap": "B",
    "gfs": "B",
    "nbm": "B",
    "rtma": "KLGA",
    "gefsatmosmean": "KLGA",
    "gefsatmos": "KLGA",
    "ifsoper": "KLGA",
    "ifsenfo": "KLGA",
    "nbmqmd": "KLGA",
    "aifsoper": "KLGA",
    "aifsenfo": "KLGA",
    "aigefssfc": "KLGA",
    "aigfssfc": "KLGA",
}

TMAX_THIN_EXECUTION_ORDER: tuple[str, ...] = (
    "nbm",
    "nbmqmd",
    "hrrr",
    "rap",
    "gfs",
    "gefsatmosmean",
    "gefsatmos",
    "ifsoper",
    "ifsenfo",
    "aifsoper",
    "aifsenfo",
    "aigefssfc",
    "aigfssfc",
    "rtma",
)


def model_spec_by_id(model_id: str) -> GribStreamModelSpec:
    for spec in MODEL_SPECS:
        if spec.model_id == model_id:
            return spec
    raise KeyError(f"unknown GribStream model_id {model_id}")


def tmax_thin_model_spec_by_id(model_id: str) -> GribStreamModelSpec:
    for spec in TMAX_THIN_MODEL_SPECS:
        if spec.model_id == model_id:
            return spec
    raise KeyError(f"unknown Tmax-thin GribStream model_id {model_id}")


def cutoff_profile_by_id(cutoff_id: str) -> GribStreamCutoffProfile:
    try:
        return CUTOFF_PROFILE_BY_ID[cutoff_id]
    except KeyError as exc:
        raise KeyError(f"unknown GribStream cutoff_id {cutoff_id}") from exc


def supported_cutoff_ids() -> tuple[str, ...]:
    return tuple(profile.cutoff_id for profile in CUTOFF_PROFILES)


def default_members(spec: GribStreamModelSpec) -> tuple[int, ...]:
    if spec.member_mode == "gefs_31":
        return tuple(range(31))
    if spec.member_mode == "ecmwf_51":
        return tuple(range(51))
    return ()


def cutoff_utc(target_date: date, *, cutoff_id: str = DEFAULT_CUTOFF_ID) -> datetime:
    profile = cutoff_profile_by_id(cutoff_id)
    cutoff_date = target_date + timedelta(days=profile.target_day_offset)
    return datetime.combine(cutoff_date, profile.cutoff_utc_time, timezone.utc)


def as_of_utc(target_date: date, spec: GribStreamModelSpec, *, cutoff_id: str = DEFAULT_CUTOFF_ID) -> datetime | None:
    if spec.buffer is None:
        return None
    return cutoff_utc(target_date, cutoff_id=cutoff_id) - spec.buffer


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def floor_to_hour(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def floor_to_cycle(value: datetime, cycle_hours: tuple[int, ...]) -> datetime:
    value = value.astimezone(timezone.utc)
    for hour in sorted(cycle_hours, reverse=True):
        candidate = value.replace(hour=hour, minute=0, second=0, microsecond=0)
        if candidate <= value:
            return candidate
    previous = value - timedelta(days=1)
    return previous.replace(hour=max(cycle_hours), minute=0, second=0, microsecond=0)


def runs_model_time_for_target(
    spec: GribStreamModelSpec,
    target_date: date,
    *,
    cutoff_id: str = DEFAULT_CUTOFF_ID,
) -> datetime:
    if spec.fetch_shape == "urma_peak_temp":
        raise ValueError("URMA uses valid-time run anchors, not one model cycle per target date")
    target_as_of = as_of_utc(target_date, spec, cutoff_id=cutoff_id)
    if target_as_of is None:
        raise ValueError(f"{spec.model_id} has no asOf-backed run cycle")
    if spec.model_id in {"gfs", "gefsatmos", "gefsatmosmean", "aigefssfc", "aigfssfc"}:
        return floor_to_cycle(target_as_of, (0, 6, 12, 18))
    if spec.model_id in {"ifsoper", "ifsenfo", "aifsoper", "aifsenfo"}:
        return floor_to_cycle(target_as_of, (0, 12))
    if spec.model_id == "rap":
        # Live T_1245UTC smoke checks showed the 08Z/09Z RAP cycles carry the
        # full NY peak window at the archive edge; 11Z is too short.
        return floor_to_hour(target_as_of - timedelta(hours=3))
    if spec.model_id == "nbmqmd":
        # The max-18h valid at T+1 06Z resolves to the 06Z NBM QMD package in
        # live checks, not the 11Z hourly cutoff floor.
        return floor_to_hour(target_as_of - timedelta(hours=5))
    return floor_to_hour(target_as_of)


def effective_target_start(spec: GribStreamModelSpec, *, cutoff_id: str = DEFAULT_CUTOFF_ID) -> date:
    profile = cutoff_profile_by_id(cutoff_id)
    return spec.catalog_archive_start + timedelta(days=profile.archive_start_offset_days)


def hourly_peak_times_utc(target_date: date) -> tuple[datetime, ...]:
    target_zone = ZoneInfo(TARGET_TZ)
    return tuple(
        datetime.combine(target_date, time(hour, 0), target_zone).astimezone(timezone.utc)
        for hour in range(12, 22)
    )


def synoptic_peak_times_utc(target_date: date) -> tuple[datetime, ...]:
    return (
        datetime.combine(target_date, time(18, 0), timezone.utc),
        datetime.combine(target_date + timedelta(days=1), time(0, 0), timezone.utc),
    )


def nbmqmd_max18_time_utc(target_date: date, *, cutoff_id: str = DEFAULT_CUTOFF_ID) -> tuple[datetime, ...]:
    profile = cutoff_profile_by_id(cutoff_id)
    return (
        datetime.combine(
            target_date + timedelta(days=1),
            time(profile.nbmqmd_max18_valid_hour_utc, 0),
            timezone.utc,
        ),
    )


def rtma_latest_time_utc(target_date: date, *, cutoff_id: str = DEFAULT_CUTOFF_ID) -> tuple[datetime, ...]:
    effective = cutoff_utc(target_date, cutoff_id=cutoff_id) - timedelta(hours=1)
    return (effective.replace(minute=0, second=0, microsecond=0),)


def valid_times_for_target(
    spec: GribStreamModelSpec,
    target_date: date,
    *,
    cutoff_id: str = DEFAULT_CUTOFF_ID,
) -> tuple[datetime, ...]:
    if spec.fetch_shape == "hourly_peak":
        return hourly_peak_times_utc(target_date)
    if spec.fetch_shape == "rtma_latest":
        return rtma_latest_time_utc(target_date, cutoff_id=cutoff_id)
    if spec.fetch_shape == "urma_peak_temp":
        return hourly_peak_times_utc(target_date)
    if spec.fetch_shape == "synoptic":
        return synoptic_peak_times_utc(target_date)
    if spec.fetch_shape == "nbmqmd_max18":
        return nbmqmd_max18_time_utc(target_date, cutoff_id=cutoff_id)
    if spec.fetch_shape == "nbm_tmax_native":
        return nbmqmd_max18_time_utc(target_date, cutoff_id=cutoff_id)
    raise ValueError(f"unsupported fetch shape {spec.fetch_shape}")


def target_date_for_valid_time(valid_time_utc: datetime) -> date:
    return valid_time_utc.astimezone(ZoneInfo(TARGET_TZ)).date()


def request_payload_for_chunk(
    *,
    spec: GribStreamModelSpec,
    target_start_date: date,
    target_end_date: date,
    coordinate_tier_name: str,
    selectors: tuple[ResolvedSelector, ...],
    members: tuple[int, ...],
    cutoff_id: str = DEFAULT_CUTOFF_ID,
) -> dict[str, object]:
    points = coordinate_tier(coordinate_tier_name)
    if spec.buffer is not None and target_start_date != target_end_date:
        raise ValueError("asOf-backed GribStream chunks must cover exactly one target date")
    valid_times: list[str] = []
    for target_date in iter_dates(target_start_date, target_end_date):
        valid_times.extend(
            valid.isoformat().replace("+00:00", "Z")
            for valid in valid_times_for_target(spec, target_date, cutoff_id=cutoff_id)
        )
    variables: list[dict[str, object]] = []
    expressions: list[dict[str, object]] = []
    seen_variables: set[str] = set()
    seen_expressions: set[str] = set()
    for selector in selectors:
        for variable in selector.request_variables:
            key = canonical_json(variable)
            if key not in seen_variables:
                variables.append(dict(variable))
                seen_variables.add(key)
        for expression in selector.request_expressions:
            key = canonical_json(expression)
            if key not in seen_expressions:
                expressions.append(dict(expression))
                seen_expressions.add(key)
    payload: dict[str, object] = {
        "coordinates": [
            {
                "lat": point.lat,
                "lon": point.lon,
                "name": point.grid_point_id,
            }
            for point in points
        ],
        "timesList": valid_times,
        "variables": variables,
        "includeMetadata": ["index_updated_at"],
    }
    if expressions:
        payload["expressions"] = expressions
    start_as_of = as_of_utc(target_start_date, spec, cutoff_id=cutoff_id)
    if start_as_of is not None:
        payload["asOf"] = start_as_of.isoformat().replace("+00:00", "Z")
    if members:
        payload["members"] = list(members)
    return payload


def estimate_credits(
    *,
    valid_times_count: int,
    selector_count: int,
    coordinate_count: int,
    member_count: int,
) -> int:
    coordinate_bucket = max(1, (coordinate_count + 499) // 500)
    return valid_times_count * selector_count * coordinate_bucket * max(1, member_count)


def _request_hash_payload(
    *,
    model_id: str,
    endpoint: str,
    payload: dict[str, object],
    feature_profile: str,
    persistence_mode: str,
) -> dict[str, object]:
    hash_payload: dict[str, object] = {
        "model_id": model_id,
        "endpoint": endpoint,
        "payload": payload,
    }
    if feature_profile != "BROAD_V1" or persistence_mode != SILVER_ATOMIC_PERSISTENCE_MODE:
        hash_payload["feature_profile"] = feature_profile
        hash_payload["persistence_mode"] = persistence_mode
    return hash_payload


def build_chunk(
    *,
    spec: GribStreamModelSpec,
    target_start_date: date,
    target_end_date: date,
    coordinate_tier_name: str,
    selectors: tuple[ResolvedSelector, ...],
    members: tuple[int, ...] | None = None,
    cutoff_id: str = DEFAULT_CUTOFF_ID,
    feature_profile: str = "BROAD_V1",
    persistence_mode: str = SILVER_ATOMIC_PERSISTENCE_MODE,
) -> GribStreamChunk:
    profile = cutoff_profile_by_id(cutoff_id)
    chunk_members = default_members(spec) if members is None else members
    valid_times: list[datetime] = []
    for target_date in iter_dates(target_start_date, target_end_date):
        valid_times.extend(valid_times_for_target(spec, target_date, cutoff_id=cutoff_id))
    payload = request_payload_for_chunk(
        spec=spec,
        target_start_date=target_start_date,
        target_end_date=target_end_date,
        coordinate_tier_name=coordinate_tier_name,
        selectors=selectors,
        members=chunk_members,
        cutoff_id=cutoff_id,
    )
    request_sha256 = sha256_hex(
        canonical_json(
            _request_hash_payload(
                model_id=spec.model_id,
                endpoint=REQUEST_ENDPOINT,
                payload=payload,
                feature_profile=feature_profile,
                persistence_mode=persistence_mode,
            )
        )
    )
    chunk_identity = {
        "endpoint": REQUEST_ENDPOINT,
        "model_id": spec.model_id,
        "target_start_date": target_start_date.isoformat(),
        "target_end_date": target_end_date.isoformat(),
        "cutoff_id": cutoff_id,
        "coordinate_tier": coordinate_tier_name.upper(),
        "request_sha256": request_sha256,
    }
    return GribStreamChunk(
        model_id=spec.model_id,
        target_start_date=target_start_date,
        target_end_date=target_end_date,
        cutoff_id=cutoff_id,
        cutoff_utc_time=profile.cutoff_utc_time_label,
        coordinate_tier=coordinate_tier_name.upper(),
        as_of_utc=as_of_utc(target_start_date, spec, cutoff_id=cutoff_id),
        valid_times_utc=tuple(valid_times),
        selectors=selectors,
        members=chunk_members,
        request_payload=payload,
        request_sha256=request_sha256,
        estimated_credits=estimate_credits(
            valid_times_count=len(valid_times),
            selector_count=len(selectors),
            coordinate_count=len(coordinate_tier(coordinate_tier_name)),
            member_count=len(chunk_members) if chunk_members else 1,
        ),
        chunk_id=f"gs_chunk_{sha256_hex(canonical_json(chunk_identity))[:32]}",
        fetch_shape=spec.fetch_shape,
        feature_profile=feature_profile,
        persistence_mode=persistence_mode,
        endpoint_type="timeseries",
    )


def runs_group_count(spec: GribStreamModelSpec) -> int:
    if spec.fetch_shape == "synoptic":
        return 2
    return 1


def _runs_base_payload(
    *,
    spec: GribStreamModelSpec,
    target_date: date,
    coordinate_tier_name: str,
    selectors: tuple[ResolvedSelector, ...],
    members: tuple[int, ...],
    cutoff_id: str,
) -> dict[str, object]:
    base = request_payload_for_chunk(
        spec=spec,
        target_start_date=target_date,
        target_end_date=target_date,
        coordinate_tier_name=coordinate_tier_name,
        selectors=selectors,
        members=members,
        cutoff_id=cutoff_id,
    )
    base.pop("asOf", None)
    base.pop("timesList", None)
    return base


def _runs_times_for_target(
    spec: GribStreamModelSpec,
    target_date: date,
    *,
    cutoff_id: str,
    group_index: int | None,
) -> tuple[tuple[datetime, datetime], ...]:
    valid_times = valid_times_for_target(spec, target_date, cutoff_id=cutoff_id)
    if group_index is not None:
        valid_times = (valid_times[group_index],)
    pairs: list[tuple[datetime, datetime]] = []
    for valid_time in valid_times:
        if spec.fetch_shape == "urma_peak_temp":
            run_time = valid_time
        else:
            run_time = runs_model_time_for_target(spec, target_date, cutoff_id=cutoff_id)
        pairs.append((run_time, valid_time))
    return tuple(pairs)


def build_runs_chunk(
    *,
    spec: GribStreamModelSpec,
    target_start_date: date,
    target_end_date: date,
    coordinate_tier_name: str,
    selectors: tuple[ResolvedSelector, ...],
    members: tuple[int, ...] | None = None,
    cutoff_id: str = DEFAULT_CUTOFF_ID,
    group_index: int | None = None,
    feature_profile: str = "BROAD_V1",
    persistence_mode: str = SILVER_ATOMIC_PERSISTENCE_MODE,
) -> GribStreamChunk:
    profile = cutoff_profile_by_id(cutoff_id)
    chunk_members = default_members(spec) if members is None else members
    pairs: list[tuple[datetime, datetime]] = []
    for target_date in iter_dates(target_start_date, target_end_date):
        pairs.extend(
            _runs_times_for_target(
                spec,
                target_date,
                cutoff_id=cutoff_id,
                group_index=group_index,
            )
        )
    if not pairs:
        raise ValueError(f"no run/valid pairs built for {spec.model_id}")
    run_times = tuple(dict.fromkeys(run_time for run_time, _ in pairs))
    valid_times = tuple(valid_time for _, valid_time in pairs)
    lead_hours = [
        int(round((valid_time - run_time).total_seconds() / 3600.0))
        for run_time, valid_time in pairs
    ]
    payload = _runs_base_payload(
        spec=spec,
        target_date=target_start_date,
        coordinate_tier_name=coordinate_tier_name,
        selectors=selectors,
        members=chunk_members,
        cutoff_id=cutoff_id,
    )
    payload["timesList"] = [iso_z(run_time) for run_time in run_times]
    payload["minLeadTime"] = f"{min(lead_hours)}h"
    payload["maxLeadTime"] = f"{max(lead_hours)}h"
    request_sha256 = sha256_hex(
        canonical_json(
            _request_hash_payload(
                model_id=spec.model_id,
                endpoint=RUNS_REQUEST_ENDPOINT,
                payload=payload,
                feature_profile=feature_profile,
                persistence_mode=persistence_mode,
            )
        )
    )
    lead_count = max(lead_hours) - min(lead_hours) + 1
    chunk_identity = {
        "endpoint": RUNS_REQUEST_ENDPOINT,
        "model_id": spec.model_id,
        "target_start_date": target_start_date.isoformat(),
        "target_end_date": target_end_date.isoformat(),
        "cutoff_id": cutoff_id,
        "coordinate_tier": coordinate_tier_name.upper(),
        "group_index": group_index,
        "request_sha256": request_sha256,
    }
    return GribStreamChunk(
        model_id=spec.model_id,
        target_start_date=target_start_date,
        target_end_date=target_end_date,
        cutoff_id=cutoff_id,
        cutoff_utc_time=profile.cutoff_utc_time_label,
        coordinate_tier=coordinate_tier_name.upper(),
        as_of_utc=as_of_utc(target_start_date, spec, cutoff_id=cutoff_id),
        valid_times_utc=valid_times,
        selectors=selectors,
        members=chunk_members,
        request_payload=payload,
        request_sha256=request_sha256,
        estimated_credits=estimate_credits(
            valid_times_count=len(run_times) * lead_count,
            selector_count=len(selectors),
            coordinate_count=len(coordinate_tier(coordinate_tier_name)),
            member_count=len(chunk_members) if chunk_members else 1,
        ),
        chunk_id=f"gs_chunk_{sha256_hex(canonical_json(chunk_identity))[:32]}",
        fetch_shape=spec.fetch_shape,
        feature_profile=feature_profile,
        persistence_mode=persistence_mode,
        endpoint_type="runs",
        expected_run_valid_pairs_utc=tuple(pairs),
    )


def chunk_date_ranges(start_date: date, end_date: date, chunk_days: int) -> tuple[tuple[date, date], ...]:
    ranges: list[tuple[date, date]] = []
    cursor = start_date
    while cursor <= end_date:
        window_end = min(end_date, cursor + timedelta(days=chunk_days - 1))
        ranges.append((cursor, window_end))
        cursor = window_end + timedelta(days=1)
    return tuple(ranges)


def build_job_plan(
    *,
    job_id: str,
    end_date: date,
    coordinate_tier_name: str,
    selectors_by_model: dict[str, tuple[ResolvedSelector, ...]],
    model_ids: tuple[str, ...] | None = None,
    start_date_override: date | None = None,
    chunk_days_override: int | None = None,
    cutoff_id: str = DEFAULT_CUTOFF_ID,
) -> GribStreamJobPlan:
    cutoff_profile_by_id(cutoff_id)
    selected_specs = [model_spec_by_id(model_id) for model_id in model_ids] if model_ids else list(MODEL_SPECS)
    chunks: list[GribStreamChunk] = []
    selector_gaps: list[dict[str, object]] = []
    for spec in selected_specs:
        spec_target_start = effective_target_start(spec, cutoff_id=cutoff_id)
        target_start = max(spec_target_start, start_date_override or spec_target_start)
        if target_start > end_date:
            selector_gaps.append(
                {
                    "model_id": spec.model_id,
                    "gap_type": "date_range_empty",
                    "gap_reason": f"effective target start {target_start.isoformat()} is after end {end_date.isoformat()}",
                }
            )
            continue
        selectors = selectors_by_model.get(spec.model_id, ())
        if not selectors:
            selector_gaps.append(
                {
                    "model_id": spec.model_id,
                    "gap_type": "selector_missing",
                    "gap_reason": "no GribStream selectors resolved for model",
                }
            )
            continue
        effective_chunk_days = 1 if spec.buffer is not None else (chunk_days_override or spec.default_chunk_days)
        for chunk_start, chunk_end in chunk_date_ranges(target_start, end_date, effective_chunk_days):
            chunks.append(
                build_chunk(
                    spec=spec,
                    target_start_date=chunk_start,
                    target_end_date=chunk_end,
                    coordinate_tier_name=coordinate_tier_name,
                    selectors=selectors,
                    cutoff_id=cutoff_id,
                )
            )
    return GribStreamJobPlan(
        job_id=job_id,
        cutoff_id=cutoff_id,
        start_date=min((chunk.target_start_date for chunk in chunks), default=end_date),
        end_date=end_date,
        coordinate_tier=coordinate_tier_name.upper(),
        chunks=tuple(chunks),
        selector_gaps=tuple(selector_gaps),
    )


def build_runs_job_plan(
    *,
    job_id: str,
    end_date: date,
    coordinate_tier_name: str,
    selectors_by_model: dict[str, tuple[ResolvedSelector, ...]],
    model_ids: tuple[str, ...] | None = None,
    start_date_override: date | None = None,
    chunk_days_override: int | None = None,
    cutoff_id: str = T1245_CUTOFF_ID,
) -> GribStreamJobPlan:
    cutoff_profile_by_id(cutoff_id)
    selected_specs = [model_spec_by_id(model_id) for model_id in model_ids] if model_ids else list(MODEL_SPECS)
    chunks: list[GribStreamChunk] = []
    selector_gaps: list[dict[str, object]] = []
    for spec in selected_specs:
        spec_target_start = effective_target_start(spec, cutoff_id=cutoff_id)
        target_start = max(spec_target_start, start_date_override or spec_target_start)
        if target_start > end_date:
            selector_gaps.append(
                {
                    "model_id": spec.model_id,
                    "gap_type": "date_range_empty",
                    "gap_reason": f"effective target start {target_start.isoformat()} is after end {end_date.isoformat()}",
                }
            )
            continue
        selectors = selectors_by_model.get(spec.model_id, ())
        if not selectors:
            selector_gaps.append(
                {
                    "model_id": spec.model_id,
                    "gap_type": "selector_missing",
                    "gap_reason": "no GribStream selectors resolved for model",
                }
            )
            continue
        effective_chunk_days = chunk_days_override or spec.default_chunk_days
        for chunk_start, chunk_end in chunk_date_ranges(target_start, end_date, effective_chunk_days):
            for group_index in range(runs_group_count(spec)):
                chunks.append(
                    build_runs_chunk(
                        spec=spec,
                        target_start_date=chunk_start,
                        target_end_date=chunk_end,
                        coordinate_tier_name=coordinate_tier_name,
                        selectors=selectors,
                        cutoff_id=cutoff_id,
                        group_index=group_index if runs_group_count(spec) > 1 else None,
                    )
                )
    return GribStreamJobPlan(
        job_id=job_id,
        cutoff_id=cutoff_id,
        start_date=min((chunk.target_start_date for chunk in chunks), default=end_date),
        end_date=end_date,
        coordinate_tier=coordinate_tier_name.upper(),
        chunks=tuple(chunks),
        selector_gaps=tuple(selector_gaps),
    )


def build_tmax_thin_runs_job_plan(
    *,
    job_id: str,
    end_date: date,
    selectors_by_model: dict[str, tuple[ResolvedSelector, ...]],
    model_ids: tuple[str, ...] | None = None,
    start_date_override: date | None = None,
    chunk_days_override: int | None = None,
    cutoff_id: str = T1245_CUTOFF_ID,
) -> GribStreamJobPlan:
    cutoff_profile_by_id(cutoff_id)
    if cutoff_id != T1245_CUTOFF_ID:
        raise ValueError("TMAX_THIN_V1 is currently defined only for T_1245UTC")
    selected_specs = (
        [tmax_thin_model_spec_by_id(model_id) for model_id in model_ids]
        if model_ids
        else [tmax_thin_model_spec_by_id(model_id) for model_id in TMAX_THIN_EXECUTION_ORDER]
    )
    chunks: list[GribStreamChunk] = []
    selector_gaps: list[dict[str, object]] = []
    for spec in selected_specs:
        spec_target_start = effective_target_start(spec, cutoff_id=cutoff_id)
        target_start = max(spec_target_start, start_date_override or spec_target_start)
        if target_start > end_date:
            selector_gaps.append(
                {
                    "model_id": spec.model_id,
                    "gap_type": "date_range_empty",
                    "gap_reason": f"effective target start {target_start.isoformat()} is after end {end_date.isoformat()}",
                    "feature_profile": TMAX_THIN_FEATURE_PROFILE,
                }
            )
            continue
        selectors = selectors_by_model.get(spec.model_id, ())
        if not selectors:
            selector_gaps.append(
                {
                    "model_id": spec.model_id,
                    "gap_type": "selector_missing",
                    "gap_reason": "no GribStream selectors resolved for model",
                    "feature_profile": TMAX_THIN_FEATURE_PROFILE,
                    "variable_group": spec.variable_group,
                }
            )
            continue
        coordinate_tier_name = TMAX_THIN_COORDINATE_TIER_BY_MODEL[spec.model_id]
        effective_chunk_days = chunk_days_override or spec.default_chunk_days
        for chunk_start, chunk_end in chunk_date_ranges(target_start, end_date, effective_chunk_days):
            group_count = runs_group_count(spec)
            for group_index in range(group_count):
                chunks.append(
                    build_runs_chunk(
                        spec=spec,
                        target_start_date=chunk_start,
                        target_end_date=chunk_end,
                        coordinate_tier_name=coordinate_tier_name,
                        selectors=selectors,
                        cutoff_id=cutoff_id,
                        group_index=group_index if group_count > 1 else None,
                        feature_profile=TMAX_THIN_FEATURE_PROFILE,
                        persistence_mode=TMAX_THIN_PERSISTENCE_MODE,
                    )
                )
    return GribStreamJobPlan(
        job_id=job_id,
        cutoff_id=cutoff_id,
        start_date=min((chunk.target_start_date for chunk in chunks), default=end_date),
        end_date=end_date,
        coordinate_tier="MIXED_TMAX_THIN",
        chunks=tuple(chunks),
        selector_gaps=tuple(selector_gaps),
    )


def tmax_thin_spec_summary_rows(*, end_date: date = DEFAULT_END_DATE, cutoff_id: str = T1245_CUTOFF_ID) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in TMAX_THIN_MODEL_SPECS:
        target_start = effective_target_start(spec, cutoff_id=cutoff_id)
        days = max(0, (end_date - target_start).days + 1)
        rows.append(
            {
                "feature_profile": TMAX_THIN_FEATURE_PROFILE,
                "cutoff_id": cutoff_id,
                "tier": spec.tier,
                "model_id": spec.model_id,
                "catalog_archive_start": spec.catalog_archive_start.isoformat(),
                "effective_target_from": target_start.isoformat(),
                "target_days": days,
                "fetch_shape": spec.fetch_shape,
                "variable_group": spec.variable_group,
                "coordinate_tier": TMAX_THIN_COORDINATE_TIER_BY_MODEL[spec.model_id],
                "expected_members": spec.expected_members,
                "expected_credits_per_day": spec.expected_credits_per_day,
                "expected_total_credits": days * spec.expected_credits_per_day,
                "buffer_minutes": int(spec.buffer.total_seconds() // 60) if spec.buffer else None,
                "intended_latest_cycle": spec.intended_latest_cycle,
                "default_chunk_days": spec.default_chunk_days,
            }
        )
    return rows
