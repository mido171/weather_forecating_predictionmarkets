from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable


@dataclass(frozen=True)
class VariableSpec:
    name: str
    level: str
    info: str = ""

    @property
    def header(self) -> str:
        return f"{self.name}|{self.level}|{self.info}"


@dataclass(frozen=True)
class NativeTmaxRule:
    available_from: date
    var: VariableSpec


@dataclass(frozen=True)
class ModelSpec:
    model_code: str
    family: str
    role: str
    archive_start: date
    snapshot_var: VariableSpec
    native_tmax_var: VariableSpec | None = None
    native_tmax_rules: tuple[NativeTmaxRule, ...] = ()
    ensemble_members: tuple[int, ...] | None = None
    notes: str | None = None

    @property
    def enabled_backtest(self) -> int:
        return int(self.role in {"backtest", "backtest_partial"})

    @property
    def enabled_live(self) -> int:
        return 1

    def native_tmax_for_date(self, settlement_date_local: date) -> VariableSpec | None:
        if self.native_tmax_var is not None:
            return self.native_tmax_var
        for rule in self.native_tmax_rules:
            if settlement_date_local >= rule.available_from:
                return rule.var
        return None

    def native_tmax_available_from(self) -> date | None:
        if self.native_tmax_var is not None:
            return self.archive_start
        if self.native_tmax_rules:
            return min(rule.available_from for rule in self.native_tmax_rules)
        return None


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_code="hrrr",
        family="regional_noaa_short",
        role="backtest",
        archive_start=date(2014, 7, 30),
        snapshot_var=VariableSpec("TMP", "2 m above ground", ""),
    ),
    ModelSpec(
        model_code="rap",
        family="regional_noaa_short",
        role="backtest",
        archive_start=date(2021, 2, 22),
        snapshot_var=VariableSpec("TMP", "2 m above ground", ""),
    ),
    ModelSpec(
        model_code="gfs",
        family="noaa_global_blend",
        role="backtest",
        archive_start=date(2021, 3, 22),
        snapshot_var=VariableSpec("TMP", "2 m above ground", ""),
        native_tmax_var=VariableSpec("TMAX", "2 m above ground", ""),
    ),
    ModelSpec(
        model_code="nbm",
        family="noaa_global_blend",
        role="backtest",
        archive_start=date(2020, 9, 29),
        snapshot_var=VariableSpec("TMP", "2 m above ground", ""),
        native_tmax_var=VariableSpec("TMAX", "2 m above ground", ""),
    ),
    ModelSpec(
        model_code="gefsatmosmean",
        family="noaa_global_blend",
        role="backtest",
        archive_start=date(2020, 10, 1),
        snapshot_var=VariableSpec("TMP", "2 m above ground", "ens mean"),
        native_tmax_var=VariableSpec("TMAX", "2 m above ground", "ens mean"),
    ),
    ModelSpec(
        model_code="gefsatmos",
        family="noaa_global_blend",
        role="backtest",
        archive_start=date(2020, 10, 1),
        snapshot_var=VariableSpec("TMP", "2 m above ground", ""),
        native_tmax_var=VariableSpec("TMAX", "2 m above ground", ""),
        ensemble_members=(0,),
        notes="Control member only for V1.",
    ),
    ModelSpec(
        model_code="ifsoper",
        family="ecmwf_physics",
        role="backtest_partial",
        archive_start=date(2024, 3, 1),
        snapshot_var=VariableSpec("2t", "sfc", ""),
        native_tmax_rules=(
            NativeTmaxRule(
                available_from=date(2024, 9, 17),
                var=VariableSpec("mx2t3", "sfc", ""),
            ),
        ),
    ),
    ModelSpec(
        model_code="ifsenfo",
        family="ecmwf_physics",
        role="backtest_partial",
        archive_start=date(2024, 3, 1),
        snapshot_var=VariableSpec("2t", "sfc", ""),
        native_tmax_rules=(
            NativeTmaxRule(
                available_from=date(2024, 11, 12),
                var=VariableSpec("mx2t3", "sfc", ""),
            ),
        ),
        ensemble_members=(0,),
        notes="Control member only for V1.",
    ),
    ModelSpec(
        model_code="graphcast",
        family="ai_global",
        role="backtest_partial",
        archive_start=date(2024, 4, 25),
        snapshot_var=VariableSpec("TMP", "2 m above ground", ""),
    ),
    ModelSpec(
        model_code="fourcastnetgfs",
        family="ai_global",
        role="backtest_partial",
        archive_start=date(2024, 5, 2),
        snapshot_var=VariableSpec("TMP", "2 m above ground", ""),
    ),
    ModelSpec(
        model_code="hrrrsubh",
        family="regional_noaa_short",
        role="live_only",
        archive_start=date(2026, 2, 4),
        snapshot_var=VariableSpec("TMP", "2 m above ground", ""),
    ),
    ModelSpec(
        model_code="rrfsprslev",
        family="regional_noaa_short",
        role="live_only",
        archive_start=date(2026, 1, 24),
        snapshot_var=VariableSpec("TMP", "2 m above ground", ""),
        native_tmax_var=VariableSpec("TMAX", "2 m above ground", ""),
    ),
    ModelSpec(
        model_code="refsprslev",
        family="regional_noaa_short",
        role="live_only",
        archive_start=date(2026, 2, 25),
        snapshot_var=VariableSpec("TMP", "2 m above ground", ""),
        native_tmax_var=VariableSpec("TMAX", "2 m above ground", ""),
        ensemble_members=(0,),
        notes="Control member only for V1.",
    ),
    ModelSpec(
        model_code="aifsoper",
        family="ecmwf_ai",
        role="live_only",
        archive_start=date(2025, 2, 25),
        snapshot_var=VariableSpec("2t", "sfc", ""),
    ),
    ModelSpec(
        model_code="aifsenfo",
        family="ecmwf_ai",
        role="live_only",
        archive_start=date(2025, 7, 2),
        snapshot_var=VariableSpec("2t", "sfc", ""),
        ensemble_members=(0,),
        notes="Control member only for V1.",
    ),
)


MODEL_BY_CODE = {spec.model_code: spec for spec in MODEL_SPECS}


def get_model_spec(model_code: str) -> ModelSpec:
    try:
        return MODEL_BY_CODE[model_code]
    except KeyError as exc:
        raise KeyError(f"Unknown model_code: {model_code}") from exc


def historical_model_specs() -> tuple[ModelSpec, ...]:
    return tuple(spec for spec in MODEL_SPECS if spec.role in {"backtest", "backtest_partial"})


def live_capable_model_specs() -> tuple[ModelSpec, ...]:
    return MODEL_SPECS


def eligible_specs_for_date(target_date_local: date, include_live_only: bool) -> tuple[ModelSpec, ...]:
    roles = {"backtest", "backtest_partial"}
    if include_live_only:
        roles.add("live_only")
    return tuple(
        spec
        for spec in MODEL_SPECS
        if spec.role in roles and spec.archive_start <= target_date_local
    )


def model_catalog_rows(created_at_utc: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in MODEL_SPECS:
        native_tmax = spec.native_tmax_for_date(spec.native_tmax_available_from() or spec.archive_start)
        rows.append(
            {
                "model_code": spec.model_code,
                "family": spec.family,
                "role": spec.role,
                "archive_start": spec.archive_start.isoformat(),
                "snapshot_var_name": spec.snapshot_var.name,
                "snapshot_var_level": spec.snapshot_var.level,
                "snapshot_var_info": spec.snapshot_var.info,
                "native_tmax_var_name": native_tmax.name if native_tmax else None,
                "native_tmax_var_level": native_tmax.level if native_tmax else None,
                "native_tmax_var_info": native_tmax.info if native_tmax else None,
                "native_tmax_available_from": (
                    spec.native_tmax_available_from().isoformat()
                    if spec.native_tmax_available_from() is not None
                    else None
                ),
                "ensemble_members_json": (
                    "[" + ",".join(str(member) for member in spec.ensemble_members) + "]"
                    if spec.ensemble_members is not None
                    else None
                ),
                "enabled_backtest": spec.enabled_backtest,
                "enabled_live": spec.enabled_live,
                "notes": spec.notes,
                "created_at_utc": created_at_utc,
            }
        )
    return rows


def model_codes(specs: Iterable[ModelSpec]) -> tuple[str, ...]:
    return tuple(spec.model_code for spec in specs)
