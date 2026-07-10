from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from klga_tmax.ingestion.hash_keys import canonical_json, sha256_hex
from klga_tmax.providers.gribstream.config import GribStreamSettings
from klga_tmax.providers.gribstream.models import GribStreamModelSpec, ResolvedSelector
from klga_tmax.providers.gribstream.plan import DEFAULT_CUTOFF_ID, MODEL_SPECS, effective_target_start


@dataclass(frozen=True)
class CatalogSnapshot:
    model_id: str
    catalog_kind: str
    catalog_url: str
    payload_sha256: str
    payload_json: dict[str, Any]
    retrieved_at_utc: datetime
    status: str = "ok"
    error_message: str | None = None


class CatalogResolver:
    def __init__(self, settings: GribStreamSettings) -> None:
        self.settings = settings
        self._cache: dict[str, dict[str, Any]] = {}
        self.snapshots: list[CatalogSnapshot] = []

    def _get_json(self, path: str, *, params: dict[str, object] | None = None) -> dict[str, Any]:
        base_url = f"{self.settings.base_url.rstrip('/')}/{path.lstrip('/')}"
        query = urlencode(params or {})
        url = f"{base_url}?{query}" if query else base_url
        cache_key = f"{url}?{canonical_json(params or {})}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        headers = {"Accept": "application/json", "User-Agent": self.settings.user_agent}
        if self.settings.api_token:
            headers["Authorization"] = f"Bearer {self.settings.api_token}"
        request = Request(url, method="GET", headers=headers)
        with urlopen(request, timeout=self.settings.timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
            payload = json.loads(body)
            response_url = response.geturl()
        if not isinstance(payload, dict):
            raise ValueError(f"catalog endpoint {path} returned non-object JSON")
        self._cache[cache_key] = payload
        self.snapshots.append(
            CatalogSnapshot(
                model_id=str((params or {}).get("dataset") or path.rsplit("/", 1)[-1]),
                catalog_kind=path.strip("/"),
                catalog_url=response_url,
                payload_sha256=sha256_hex(json.dumps(payload, sort_keys=True, default=str)),
                payload_json=payload,
                retrieved_at_utc=datetime.now(timezone.utc),
            )
        )
        return payload

    def dataset(self, model_id: str) -> dict[str, Any]:
        return self._get_json(f"catalog/datasets/{model_id}")

    def parameters(self, model_id: str) -> dict[str, Any]:
        return self._get_json(f"catalog/datasets/{model_id}/parameters")

    def parameter_detail(self, model_id: str, parameter: str) -> dict[str, Any]:
        return self._get_json(f"catalog/datasets/{model_id}/parameters/{parameter}")

    def shared_parameter(self, model_id: str, code: str, *, alias: str) -> dict[str, Any]:
        return self._get_json(f"catalog/shared-parameters/{code}", params={"dataset": model_id, "alias": alias})


def _find_catalog_value(payload: Any, predicate) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        if predicate(payload):
            return payload
        for value in payload.values():
            found = _find_catalog_value(value, predicate)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_catalog_value(item, predicate)
            if found is not None:
                return found
    return None


def _payload_mentions(payload: Any, needle: str) -> bool:
    return needle.lower() in json.dumps(payload, sort_keys=True, default=str).lower()


def _shared_selector(resolver: CatalogResolver, model_id: str, code: str, alias: str) -> ResolvedSelector:
    payload = resolver.shared_parameter(model_id, code, alias=alias)
    resolved_request = _find_catalog_value(
        payload,
        lambda node: isinstance(node.get("resolved_request"), dict),
    )
    request = payload.get("resolved_request") if isinstance(payload.get("resolved_request"), dict) else None
    if request is None and resolved_request is not None:
        request = resolved_request.get("resolved_request")
    if not isinstance(request, dict):
        raise ValueError(f"shared parameter {code} for {model_id} did not return resolved_request")
    variables = tuple(dict(item) for item in request.get("variables", []) if isinstance(item, dict))
    expressions = tuple(dict(item) for item in request.get("expressions", []) if isinstance(item, dict))
    if not variables:
        raise ValueError(f"shared parameter {code} for {model_id} resolved without variables")
    if not _payload_mentions(request, alias):
        raise ValueError(f"shared parameter {code} for {model_id} did not include requested alias {alias}")
    return ResolvedSelector(
        alias=alias,
        request_variables=variables,
        variable_name=code,
        shared_parameter=code,
        request_expressions=expressions,
        unit_hint=_unit_hint(code),
    )


def _native_selector(
    resolver: CatalogResolver,
    model_id: str,
    *,
    parameter: str,
    alias: str,
    level_contains: str | None = None,
    info_contains: str | None = None,
    exact_info: str | None = None,
) -> ResolvedSelector:
    payload = resolver.parameter_detail(model_id, parameter)

    def predicate(node: dict[str, Any]) -> bool:
        node_text = json.dumps(node, sort_keys=True, default=str).lower()
        if parameter.lower() not in node_text:
            return False
        if level_contains and level_contains.lower() not in node_text:
            return False
        if info_contains and info_contains.lower() not in node_text:
            return False
        if exact_info is not None:
            info = str(node.get("info", node.get("parameterInfo", ""))).strip().lower()
            if info != exact_info.lower():
                return False
        return True

    found = _find_catalog_value(payload, lambda node: isinstance(node.get("selector"), dict) and predicate(node))
    if found is not None:
        variable = dict(found["selector"])
    else:
        found = _find_catalog_value(payload, predicate)
        variable = {
            "name": str(found.get("name", parameter)) if found is not None else parameter,
            "level": str(found.get("level", level_contains or "")) if found is not None else str(level_contains or ""),
        }
        info = found.get("info", exact_info if exact_info is not None else info_contains) if found is not None else exact_info or info_contains
        if info is not None:
            variable["info"] = str(info)
    if found is None or not variable.get("level"):
        raise ValueError(
            f"catalog selector not found: model={model_id} parameter={parameter} "
            f"level_contains={level_contains} info_contains={info_contains} exact_info={exact_info}"
        )
    variable["alias"] = alias
    return ResolvedSelector(
        alias=alias,
        request_variables=(variable,),
        variable_name=str(variable["name"]),
        variable_level=str(variable["level"]),
        variable_info=str(variable.get("info")) if variable.get("info") is not None else None,
        unit_hint=_unit_hint(alias),
    )


def _expression_selector(alias: str, expression: str, *, unit_hint: str | None = None) -> ResolvedSelector:
    return ResolvedSelector(
        alias=alias,
        request_variables=(),
        variable_name=f"expression:{alias}",
        variable_level="derived",
        variable_info=expression,
        request_expressions=({"expression": expression, "alias": alias},),
        unit_hint=unit_hint,
    )


def _unit_hint(code: str) -> str | None:
    key = code.lower()
    if "temperature" in key or key in {"tmp", "tmax", "tmin", "2t", "2d", "dew_point_2m"}:
        return "K"
    if "humidity" in key:
        return "%"
    if "wind" in key or key in {"10u", "10v"}:
        return "m/s"
    if "pressure" in key or key == "msl":
        return "Pa"
    if "precip" in key or key == "tp":
        return "m"
    return None


def _resolve_hourly_8(resolver: CatalogResolver, model_id: str) -> tuple[ResolvedSelector, ...]:
    selectors = [
        _shared_selector(resolver, model_id, "temperature_2m", "temperature_2m"),
        _shared_selector(resolver, model_id, "dew_point_2m", "dew_point_2m"),
        _shared_selector(resolver, model_id, "relative_humidity_2m", "relative_humidity_2m"),
        _shared_selector(resolver, model_id, "u_wind_10m", "u_wind_10m"),
        _shared_selector(resolver, model_id, "v_wind_10m", "v_wind_10m"),
        _shared_selector(resolver, model_id, "wind_speed_10m", "wind_speed_10m"),
        _shared_selector(resolver, model_id, "total_precipitation", "total_precipitation"),
    ]
    try:
        selectors.append(_shared_selector(resolver, model_id, "cloud_cover_total", "cloud_cover_total"))
    except Exception:
        selectors.append(_shared_selector(resolver, model_id, "mean_sea_level_pressure", "mean_sea_level_pressure"))
    return tuple(selectors)


def _resolve_rtma_4(resolver: CatalogResolver, model_id: str) -> tuple[ResolvedSelector, ...]:
    selectors = [
        _shared_selector(resolver, model_id, "temperature_2m", "temperature_2m"),
        _shared_selector(resolver, model_id, "dew_point_2m", "dew_point_2m"),
        _shared_selector(resolver, model_id, "relative_humidity_2m", "relative_humidity_2m"),
    ]
    try:
        selectors.append(_shared_selector(resolver, model_id, "wind_gust", "wind_gust"))
    except Exception:
        selectors.append(_shared_selector(resolver, model_id, "wind_speed_10m", "wind_speed_10m"))
    return tuple(selectors)


def _resolve_nbm_8(resolver: CatalogResolver, model_id: str) -> tuple[ResolvedSelector, ...]:
    return (
        _native_selector(resolver, model_id, parameter="TMP", alias="temperature_2m", level_contains="2 m", exact_info=""),
        _native_selector(resolver, model_id, parameter="DPT", alias="dew_point_2m", level_contains="2 m", exact_info=""),
        _shared_selector(resolver, model_id, "wind_speed_10m", "wind_speed_10m"),
        _shared_selector(resolver, model_id, "wind_gust", "wind_gust"),
        _shared_selector(resolver, model_id, "total_precipitation", "total_precipitation"),
        _native_selector(resolver, model_id, parameter="TMAX", alias="tmax_2m", level_contains="2 m", exact_info=""),
        _native_selector(resolver, model_id, parameter="TMP", alias="temperature_2m_ens_stddev", level_contains="2 m", info_contains="ens std dev"),
        _native_selector(resolver, model_id, parameter="TMAX", alias="tmax_2m_ens_stddev", level_contains="2 m", info_contains="ens std dev"),
    )


def _resolve_ecmwf_7(resolver: CatalogResolver, model_id: str) -> tuple[ResolvedSelector, ...]:
    return (
        _native_selector(resolver, model_id, parameter="2t", alias="temperature_2m"),
        _native_selector(resolver, model_id, parameter="2d", alias="dew_point_2m"),
        _native_selector(resolver, model_id, parameter="10u", alias="u_wind_10m"),
        _native_selector(resolver, model_id, parameter="10v", alias="v_wind_10m"),
        _native_selector(resolver, model_id, parameter="msl", alias="mean_sea_level_pressure"),
        _native_selector(resolver, model_id, parameter="tp", alias="total_precipitation"),
        _native_selector(resolver, model_id, parameter="tcc", alias="cloud_cover_total"),
    )


def _resolve_nbmqmd(resolver: CatalogResolver, model_id: str) -> tuple[ResolvedSelector, ...]:
    percentiles = (1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99)
    selectors: list[ResolvedSelector] = []
    for percentile in percentiles:
        selectors.append(
            _native_selector(
                resolver,
                model_id,
                parameter="TMP",
                alias=f"tmp_max18_p{percentile:02d}",
                level_contains="2 m",
                info_contains=f"{percentile}% level | max-18h",
            )
        )
    return tuple(selectors)


def _resolve_temperature_peak_only(resolver: CatalogResolver, model_id: str) -> tuple[ResolvedSelector, ...]:
    return (_shared_selector(resolver, model_id, "temperature_2m", "temperature_2m"),)


def _resolve_native_tmax_core(resolver: CatalogResolver, model_id: str) -> tuple[ResolvedSelector, ...]:
    return (
        _native_selector(
            resolver,
            model_id,
            parameter="TMAX",
            alias="tmax_2m",
            level_contains="2 m",
            exact_info="",
        ),
        _native_selector(
            resolver,
            model_id,
            parameter="TMAX",
            alias="tmax_2m_ens_stddev",
            level_contains="2 m",
            info_contains="ens std dev",
        ),
    )


def _resolve_ecmwf_temperature_only(resolver: CatalogResolver, model_id: str) -> tuple[ResolvedSelector, ...]:
    try:
        return (_native_selector(resolver, model_id, parameter="2t", alias="temperature_2m"),)
    except Exception:
        return (_shared_selector(resolver, model_id, "temperature_2m", "temperature_2m"),)


def _resolve_rtma_current_state_thin(resolver: CatalogResolver, model_id: str) -> tuple[ResolvedSelector, ...]:
    selectors = [
        _shared_selector(resolver, model_id, "temperature_2m", "temperature_2m"),
        _shared_selector(resolver, model_id, "dew_point_2m", "dew_point_2m"),
    ]
    try:
        selectors.append(_shared_selector(resolver, model_id, "wind_speed_10m", "wind_speed_10m"))
    except Exception:
        selectors.append(_shared_selector(resolver, model_id, "wind_gust", "wind_gust"))
    try:
        selectors.append(_shared_selector(resolver, model_id, "wind_gust", "wind_gust"))
    except Exception:
        pass
    return tuple(selectors)


def resolve_selectors_for_model(
    resolver: CatalogResolver,
    spec: GribStreamModelSpec,
) -> tuple[ResolvedSelector, ...]:
    if spec.variable_group == "hourly_8":
        return _resolve_hourly_8(resolver, spec.model_id)
    if spec.variable_group == "rtma_4":
        return _resolve_rtma_4(resolver, spec.model_id)
    if spec.variable_group == "nbm_8":
        return _resolve_nbm_8(resolver, spec.model_id)
    if spec.variable_group == "ecmwf_7":
        return _resolve_ecmwf_7(resolver, spec.model_id)
    if spec.variable_group == "nbmqmd_percentiles":
        return _resolve_nbmqmd(resolver, spec.model_id)
    if spec.variable_group == "temp_only":
        return (_shared_selector(resolver, spec.model_id, "temperature_2m", "temperature_2m"),)
    if spec.variable_group == "temperature_peak_only":
        return _resolve_temperature_peak_only(resolver, spec.model_id)
    if spec.variable_group == "native_tmax_core":
        return _resolve_native_tmax_core(resolver, spec.model_id)
    if spec.variable_group == "ecmwf_temperature_only":
        return _resolve_ecmwf_temperature_only(resolver, spec.model_id)
    if spec.variable_group == "ensemble_temperature_only":
        return _resolve_temperature_peak_only(resolver, spec.model_id)
    if spec.variable_group == "rtma_current_state_thin":
        return _resolve_rtma_current_state_thin(resolver, spec.model_id)
    raise ValueError(f"unsupported GribStream selector group {spec.variable_group}")


def resolve_all_selectors(
    settings: GribStreamSettings,
    *,
    model_ids: tuple[str, ...] | None = None,
    model_specs: tuple[GribStreamModelSpec, ...] = MODEL_SPECS,
) -> tuple[dict[str, tuple[ResolvedSelector, ...]], tuple[dict[str, Any], ...], tuple[CatalogSnapshot, ...]]:
    resolver = CatalogResolver(settings)
    selected_specs = [spec for spec in model_specs if model_ids is None or spec.model_id in model_ids]
    selectors: dict[str, tuple[ResolvedSelector, ...]] = {}
    gaps: list[dict[str, Any]] = []
    for spec in selected_specs:
        try:
            resolver.dataset(spec.model_id)
            selectors[spec.model_id] = resolve_selectors_for_model(resolver, spec)
        except Exception as exc:
            gaps.append(
                {
                    "model_id": spec.model_id,
                    "gap_type": "selector_resolution_failed",
                    "gap_reason": str(exc),
                    "variable_group": spec.variable_group,
                }
            )
    return selectors, tuple(gaps), tuple(resolver.snapshots)


def spec_summary_rows(*, end_date: date = date(2026, 6, 28), cutoff_id: str = DEFAULT_CUTOFF_ID) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in MODEL_SPECS:
        target_start = effective_target_start(spec, cutoff_id=cutoff_id)
        days = max(0, (end_date - target_start).days + 1)
        rows.append(
            {
                "cutoff_id": cutoff_id,
                "tier": spec.tier,
                "model_id": spec.model_id,
                "catalog_archive_start": spec.catalog_archive_start.isoformat(),
                "effective_target_from": target_start.isoformat(),
                "target_days": days,
                "fetch_shape": spec.fetch_shape,
                "variable_group": spec.variable_group,
                "expected_members": spec.expected_members,
                "expected_credits_per_day": spec.expected_credits_per_day,
                "expected_total_credits": days * spec.expected_credits_per_day,
                "buffer_minutes": int(spec.buffer.total_seconds() // 60) if spec.buffer else None,
                "intended_latest_cycle": spec.intended_latest_cycle,
                "default_chunk_days": spec.default_chunk_days,
            }
        )
    return rows
