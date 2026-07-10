"""Leakage-safe finalized-target memory features for H24N."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from statistics import mean, pstdev

from hkg_t24.constants import (
    TARGET_MEMORY_FEATURE_WHITELIST,
    TARGET_MEMORY_MISSING_INDICATOR_FEATURES,
    assert_no_forbidden_target_memory_names,
)

NumericFeature = float | int | None


def _numeric_window(values: Sequence[float | None], start: int, end: int) -> list[float] | None:
    if start < 0:
        return None
    window = values[start:end]
    if len(window) != end - start or any(value is None for value in window):
        return None
    return [float(value) for value in window if value is not None]


def _ols_slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("OLS slope requires at least two values")
    x_mean = (len(values) - 1) / 2.0
    y_mean = mean(values)
    numerator = sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values))
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    return numerator / denominator


def _spell_length(
    values: Sequence[float | None],
    end_index: int,
    *,
    hot: bool,
) -> int | None:
    if end_index < 0:
        return None
    count = 0
    threshold = 30.0 if hot else 15.0
    for index in range(end_index, -1, -1):
        value = values[index]
        if value is None:
            break
        if hot and float(value) >= threshold:
            count += 1
            continue
        if not hot and float(value) <= threshold:
            count += 1
            continue
        break
    return count


def _circular_doy_distance(left: int, right: int) -> int:
    raw = abs(left - right)
    return min(raw, 366 - raw)


def _causal_climatology(
    labels: Sequence[tuple[date, float | None]],
    target_date_hkt: date,
) -> tuple[float | None, float | None]:
    cutoff_date = target_date_hkt - timedelta(days=2)
    target_doy = target_date_hkt.timetuple().tm_yday
    earliest_year = target_date_hkt.year - 30
    sample = [
        float(value)
        for local_date, value in labels
        if value is not None
        and earliest_year <= local_date.year < target_date_hkt.year
        and local_date <= cutoff_date
        and _circular_doy_distance(local_date.timetuple().tm_yday, target_doy) <= 15
    ]
    if len(sample) < 10:
        return None, None
    return mean(sample), pstdev(sample)


def _annual_mean_slope(
    labels: Sequence[tuple[date, float | None]],
    target_date_hkt: date,
) -> float | None:
    cutoff_date = target_date_hkt - timedelta(days=2)
    by_year: dict[int, list[float]] = {year: [] for year in range(target_date_hkt.year - 10, target_date_hkt.year)}
    for local_date, value in labels:
        if value is None or local_date > cutoff_date or local_date.year not in by_year:
            continue
        by_year[local_date.year].append(float(value))
    annual_means = [
        (year, mean(values))
        for year, values in sorted(by_year.items())
        if len(values) >= 300
    ]
    if len(annual_means) < 8:
        return None
    first_year = annual_means[0][0]
    x = [float(year - first_year) for year, _ in annual_means]
    y = [value for _, value in annual_means]
    x_mean = mean(x)
    y_mean = mean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator == 0:
        return None
    numerator = sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(x, y, strict=True))
    return numerator / denominator


def _base_feature_map() -> dict[str, NumericFeature]:
    return {feature_name: None for feature_name in TARGET_MEMORY_FEATURE_WHITELIST}


def _with_missing_indicators(features: dict[str, NumericFeature]) -> dict[str, NumericFeature]:
    for feature_name in TARGET_MEMORY_FEATURE_WHITELIST:
        if feature_name == "target__year_index":
            continue
        features[f"{feature_name}__is_missing"] = int(features[feature_name] is None)
    missing_indicator_gap = sorted(set(TARGET_MEMORY_MISSING_INDICATOR_FEATURES) - set(features))
    if missing_indicator_gap:
        raise ValueError("Missing target-memory indicators: " + ", ".join(missing_indicator_gap))
    return features


def build_target_memory_features(
    labels: Sequence[tuple[date, float | None]],
    *,
    selected_dates: set[date] | None = None,
) -> dict[date, dict[str, NumericFeature]]:
    """Build canonical Jira 002 target-memory features for selected label dates.

    Every finalized target value used by this builder is at least two local
    calendar days older than the target date being featurized.
    """
    assert_no_forbidden_target_memory_names(TARGET_MEMORY_FEATURE_WHITELIST)
    ordered = sorted(labels, key=lambda item: item[0])
    dates = [item[0] for item in ordered]
    values = [item[1] for item in ordered]
    output: dict[date, dict[str, NumericFeature]] = {}

    for index, target_date_hkt in enumerate(dates):
        if selected_dates is not None and target_date_hkt not in selected_dates:
            continue
        features = _base_feature_map()
        for lag in (2, 3, 7, 14, 30, 60, 365):
            source_index = index - lag
            if source_index >= 0:
                lag_value = values[source_index]
                if lag_value is not None:
                    features[f"target__lag{lag}_tmax_c"] = float(lag_value)

        roll_means: dict[int, float] = {}
        slopes: dict[int, float] = {}
        for window in (7, 14, 30, 60, 365):
            numeric_window = _numeric_window(values, index - window - 1, index - 1)
            if numeric_window is None:
                continue
            roll_mean = mean(numeric_window)
            roll_means[window] = roll_mean
            features[f"target__roll{window}_mean_lag2_c"] = roll_mean
            if window in {7, 14, 30}:
                features[f"target__roll{window}_std_lag2_c"] = pstdev(numeric_window)
            if window in {7, 14}:
                features[f"target__range{window}_lag2_c"] = max(numeric_window) - min(numeric_window)
            if window in {7, 30}:
                slope = _ols_slope(numeric_window)
                slopes[window] = slope
                features[f"target__slope{window}_lag2_c_per_day"] = slope

        lag2 = features["target__lag2_tmax_c"]
        if lag2 is not None and 7 in roll_means:
            features["target__lag2_minus_roll7_c"] = float(lag2) - roll_means[7]
        if lag2 is not None and 30 in roll_means:
            features["target__lag2_minus_roll30_c"] = float(lag2) - roll_means[30]
        if 7 in roll_means and 30 in roll_means:
            features["target__roll7_minus_roll30_c"] = roll_means[7] - roll_means[30]
        if 7 in slopes and 30 in slopes:
            features["target__slope7_minus_slope30_lag2_c_per_day"] = slopes[7] - slopes[30]

        lag2_index = index - 2
        features["target__hot_spell_length_lag2_days"] = _spell_length(values, lag2_index, hot=True)
        features["target__cool_spell_length_lag2_days"] = _spell_length(values, lag2_index, hot=False)

        clim_mean, clim_std = _causal_climatology(ordered, target_date_hkt)
        features["target__clim30_mean_c"] = clim_mean
        features["target__clim30_std_c"] = clim_std
        if lag2 is not None and clim_mean is not None:
            features["target__lag2_minus_clim30_c"] = float(lag2) - clim_mean

        features["target__warming_trend_10y_c_per_year"] = _annual_mean_slope(ordered, target_date_hkt)
        features["target__year_index"] = target_date_hkt.year - 2000
        output[target_date_hkt] = _with_missing_indicators(features)

    return output


def assert_target_year_index_matches_calendar(
    target_features: Mapping[str, object],
    calendar_features: dict[str, float | int],
) -> None:
    """Fail closed if the target and calendar year-index features diverge."""
    target_year_index = target_features.get("target__year_index")
    calendar_year_index = calendar_features.get("calendar__year_index")
    if target_year_index != calendar_year_index:
        raise ValueError(
            "target__year_index must equal calendar__year_index; "
            f"target={target_year_index}, calendar={calendar_year_index}"
        )
