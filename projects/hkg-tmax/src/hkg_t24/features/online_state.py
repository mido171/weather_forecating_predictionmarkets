"""Causal online residual-memory state for Jira 002."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

from hkg_t24.constants import ONLINE_HALF_LIVES, ONLINE_SOURCE_SCOPES


@dataclass(frozen=True)
class ResidualObservation:
    """A settled prediction/label pair for online-state replay."""

    target_date_hkt: date
    source_key: str
    state_scope: str
    prediction_tmax_c: float
    target_tmax_c: float

    @property
    def residual_c(self) -> float:
        return self.target_tmax_c - self.prediction_tmax_c


@dataclass(frozen=True)
class OnlineState:
    """Causal online state row before conversion to feature names."""

    target_date_hkt: date
    source_key: str
    state_scope: str
    n_prior_rows: int
    warmup_status: str
    state_available: bool
    features: dict[str, float | int | None]


def ewma_alpha(half_life: int) -> float:
    """Return alpha for the contract half-life definition."""
    return 1.0 - math.exp(-math.log(2.0) / half_life)


def warmup_status(n_prior_rows: int) -> tuple[str, bool]:
    """Return the Jira 002 warmup status and availability flag."""
    if n_prior_rows == 0:
        return "NO_HISTORY", False
    if n_prior_rows < 5:
        return "COLD_START", True
    if n_prior_rows < 20:
        return "WARMING", True
    return "READY", True


def _ewma(values: Sequence[float], half_life: int) -> float | None:
    if not values:
        return None
    alpha = ewma_alpha(half_life)
    current = values[0]
    for value in values[1:]:
        current = alpha * value + (1.0 - alpha) * current
    return current


def _streaks(residuals: Sequence[float]) -> tuple[int, int, int]:
    over = 0
    under = 0
    neutral = 0
    for residual in reversed(residuals):
        if residual <= -0.05:
            if under or neutral:
                break
            over += 1
            continue
        if residual >= 0.05:
            if over or neutral:
                break
            under += 1
            continue
        if over or under:
            break
        neutral += 1
    return over, under, neutral


def online_feature_prefix(source_key: str, state_scope: str) -> str:
    return f"online__{source_key}__{state_scope}"


def online_state_feature_names(source_key: str, state_scope: str) -> tuple[str, ...]:
    """Return the deterministic feature names emitted for one source/scope state."""
    prefix = online_feature_prefix(source_key, state_scope)
    names = [
        f"{prefix}__n_prior_rows",
        f"{prefix}__state_available",
        f"{prefix}__warmup_status_code",
        f"{prefix}__overforecast_streak_days",
        f"{prefix}__underforecast_streak_days",
        f"{prefix}__neutral_streak_days",
    ]
    for half_life in ONLINE_HALF_LIVES:
        names.extend(
            [
                f"{prefix}__ewma_bias_h{half_life}_c",
                f"{prefix}__ewma_abs_error_h{half_life}_c",
                f"{prefix}__ewma_sq_error_h{half_life}_c2",
                f"{prefix}__error_volatility_h{half_life}_c",
            ]
        )
    names.extend(
        [
            f"{prefix}__correction_bias_h20_shrunk_c",
            f"{prefix}__correction_bias_h20_capped_c",
            f"{prefix}__expected_abs_error_h20_shrunk_c",
        ]
    )
    return tuple(names)


def all_online_state_feature_names() -> tuple[str, ...]:
    return tuple(
        name
        for source_key, state_scope in ONLINE_SOURCE_SCOPES
        for name in online_state_feature_names(source_key, state_scope)
    )


def build_online_state(
    *,
    target_date_hkt: date,
    source_key: str,
    state_scope: str,
    observations: Iterable[ResidualObservation],
) -> OnlineState:
    """Build state for one target date using only observations with date < target."""
    prior = sorted(
        [
            observation
            for observation in observations
            if observation.source_key == source_key
            and observation.state_scope == state_scope
            and observation.target_date_hkt < target_date_hkt
        ],
        key=lambda observation: observation.target_date_hkt,
    )
    residuals = [observation.residual_c for observation in prior]
    abs_errors = [abs(value) for value in residuals]
    sq_errors = [value * value for value in residuals]
    status, available = warmup_status(len(prior))
    status_code = {"NO_HISTORY": 0, "COLD_START": 1, "WARMING": 2, "READY": 3}[status]
    prefix = online_feature_prefix(source_key, state_scope)
    features: dict[str, float | int | None] = {
        f"{prefix}__n_prior_rows": len(prior),
        f"{prefix}__state_available": int(available),
        f"{prefix}__warmup_status_code": status_code,
    }
    over, under, neutral = _streaks(residuals)
    features[f"{prefix}__overforecast_streak_days"] = over
    features[f"{prefix}__underforecast_streak_days"] = under
    features[f"{prefix}__neutral_streak_days"] = neutral
    ewma_bias_h20: float | None = None
    ewma_abs_h20: float | None = None
    for half_life in ONLINE_HALF_LIVES:
        bias = _ewma(residuals, half_life)
        abs_error = _ewma(abs_errors, half_life)
        sq_error = _ewma(sq_errors, half_life)
        volatility = None
        if bias is not None and sq_error is not None:
            volatility = math.sqrt(max(0.0, sq_error - bias * bias))
        features[f"{prefix}__ewma_bias_h{half_life}_c"] = bias
        features[f"{prefix}__ewma_abs_error_h{half_life}_c"] = abs_error
        features[f"{prefix}__ewma_sq_error_h{half_life}_c2"] = sq_error
        features[f"{prefix}__error_volatility_h{half_life}_c"] = volatility
        if half_life == 20:
            ewma_bias_h20 = bias
            ewma_abs_h20 = abs_error

    if len(prior) == 0 or ewma_bias_h20 is None or ewma_abs_h20 is None:
        features[f"{prefix}__correction_bias_h20_shrunk_c"] = None
        features[f"{prefix}__correction_bias_h20_capped_c"] = None
        features[f"{prefix}__expected_abs_error_h20_shrunk_c"] = None
    else:
        shrink = len(prior) / (len(prior) + 40.0)
        shrunk = shrink * ewma_bias_h20
        features[f"{prefix}__correction_bias_h20_shrunk_c"] = shrunk
        features[f"{prefix}__correction_bias_h20_capped_c"] = max(-0.40, min(0.40, shrunk))
        features[f"{prefix}__expected_abs_error_h20_shrunk_c"] = max(
            0.20,
            shrink * ewma_abs_h20 + (1.0 - shrink) * 0.90,
        )

    return OnlineState(
        target_date_hkt=target_date_hkt,
        source_key=source_key,
        state_scope=state_scope,
        n_prior_rows=len(prior),
        warmup_status=status,
        state_available=available,
        features=features,
    )


def replay_online_states(
    *,
    target_dates: Sequence[date],
    observations: Sequence[ResidualObservation],
    source_scopes: Sequence[tuple[str, str]] = ONLINE_SOURCE_SCOPES,
) -> list[OnlineState]:
    """Replay all online states causally for each target date."""
    states: list[OnlineState] = []
    for target_date_hkt in sorted(target_dates):
        for source_key, state_scope in source_scopes:
            states.append(
                build_online_state(
                    target_date_hkt=target_date_hkt,
                    source_key=source_key,
                    state_scope=state_scope,
                    observations=observations,
                )
            )
    return states
