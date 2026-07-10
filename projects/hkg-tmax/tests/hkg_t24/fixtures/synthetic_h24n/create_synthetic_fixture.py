from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from hkg_t24.features.nwp_daily import (
    build_gefs_ensemble_features,
    build_gefs_mean_features,
    build_gfs_daily_features,
)
from hkg_t24.features.official_anchor import OfficialForecastRow, official_feature_map
from hkg_t24.features.online_state import ResidualObservation, build_online_state
from hkg_t24.features.target_memory import build_target_memory_features
from hkg_t24.timeutils import operational_freeze_utc


def synthetic_labels(days: int = 80) -> list[tuple[date, float]]:
    start = date(2021, 1, 1)
    return [(start + timedelta(days=offset), 25.0 + (offset % 10) * 0.2) for offset in range(days)]


def synthetic_feature_families() -> tuple[
    list[date],
    dict[date, float],
    dict[date, dict[str, object]],
    dict[date, dict[str, object]],
    dict[date, dict[str, object]],
    dict[date, dict[str, object]],
]:
    labels = synthetic_labels()
    label_map = {target_date: value for target_date, value in labels}
    dates = [target_date for target_date, _ in labels[40:70]]
    target_memory = {
        target_date: dict(features)
        for target_date, features in build_target_memory_features(labels).items()
        if target_date in dates
    }
    official: dict[date, dict[str, object]] = {}
    nwp: dict[date, dict[str, object]] = {}
    online: dict[date, dict[str, object]] = {}
    observations: list[ResidualObservation] = []
    for target_date in dates:
        freeze = operational_freeze_utc(target_date)
        issue = datetime.combine(target_date - timedelta(days=1), datetime.min.time(), tzinfo=UTC)
        official[target_date] = official_feature_map(
            [
                OfficialForecastRow(
                    issue_at_utc=issue,
                    forecast_min_c=23.0,
                    forecast_max_c=label_map[target_date] - 0.2,
                    forecast_text="Fine and hot",
                )
            ],
            operational_freeze_utc=freeze,
        )
        loc_values = {
            "center": label_map[target_date] - 0.3,
            "inland_nw": label_map[target_date] + 0.1,
            "marine_s": label_map[target_date] - 0.6,
            "local_n": label_map[target_date] - 0.1,
            "local_s": label_map[target_date] - 0.4,
            "local_e": label_map[target_date] - 0.2,
            "local_w": label_map[target_date] - 0.5,
        }
        nwp[target_date] = {}
        nwp[target_date].update(build_gfs_daily_features(location_tmax_c=loc_values))
        nwp[target_date].update(build_gefs_mean_features(location_tmax_c=loc_values))
        nwp[target_date].update(
            build_gefs_ensemble_features([label_map[target_date] - 0.6, label_map[target_date] - 0.2, label_map[target_date]])
        )
        observations.append(
            ResidualObservation(
                target_date_hkt=target_date,
                source_key="official_raw",
                state_scope="global",
                prediction_tmax_c=label_map[target_date] - 0.2,
                target_tmax_c=label_map[target_date],
            )
        )
        state = build_online_state(
            target_date_hkt=target_date,
            source_key="official_raw",
            state_scope="global",
            observations=observations,
        )
        online[target_date] = state.features
    return dates, label_map, target_memory, official, nwp, online
