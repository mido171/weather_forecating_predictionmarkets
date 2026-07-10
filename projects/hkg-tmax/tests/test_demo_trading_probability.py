from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
import pytest
from hkg_tmax_demo_trading import probability as probability_module
from hkg_tmax_demo_trading.probability import (
    DEFAULT_DATABASE_URL,
    ForecastSnapshot,
    ForecastUnavailable,
    compute_b4_probabilities,
    historical_cutoff_forecast,
    latest_hko_forecast,
    live_cutoff_forecast,
)
from hkg_tmax_probability.bucket_rules import BUCKET_KEYS


def _write_minimal_probability_config(root: Path) -> None:
    config_path = root / "config" / "experiments" / "hkg_tmax" / "probability_bucket_v1.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
models:
  B4_hierarchical_residual_pmf:
    alpha_grid:
      month_alpha: [20.0]
      cell_alpha: [10.0]
""".lstrip(),
        encoding="utf-8",
    )


def test_latest_hko_forecast_parses_flw_tonight_tomorrow_range_and_prefers_local_forecast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch_json(url: str) -> dict[str, Any]:
        if "dataType=flw" in url:
            return {
                "forecastPeriod": "Weather forecast for tonight and tomorrow",
                "forecastDesc": "Temperatures will range between 26 and 30 degrees.",
                "updateTime": "2026-07-07T18:45:00+08:00",
            }
        return {
            "updateTime": "2026-07-07T18:45:00+08:00",
            "weatherForecast": [
                {
                    "forecastDate": "20260708",
                    "forecastMaxtemp": {"value": 31},
                    "forecastMintemp": {"value": 26},
                }
            ],
        }

    monkeypatch.setattr(probability_module, "_fetch_json", fake_fetch_json)

    forecast = latest_hko_forecast(date(2026, 7, 8))

    assert forecast.source == "HKO flw local forecast"
    assert forecast.update_time_hkt.isoformat() == "2026-07-07T18:45:00+08:00"
    assert forecast.forecast_min_c == 26.0
    assert forecast.forecast_max_c == 30.0


def test_live_cutoff_forecast_uses_only_current_updates_inside_elapsed_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = ForecastSnapshot(
        source="HKO flw local forecast",
        update_time_hkt=datetime.fromisoformat("2026-07-07T18:45:00+08:00"),
        target_date=date(2026, 7, 8),
        forecast_min_c=26.0,
        forecast_max_c=30.0,
        as_of_profile="t_minus_1_1900_hkt",
        raw={"fixture": True},
    )
    fnd_candidate = ForecastSnapshot(
        source="HKO fnd 9-day forecast",
        update_time_hkt=datetime.fromisoformat("2026-07-07T16:30:00+08:00"),
        target_date=date(2026, 7, 8),
        forecast_min_c=26.0,
        forecast_max_c=30.0,
        as_of_profile="t_minus_1_1800_hkt",
        raw={"fixture": "fnd"},
    )
    monkeypatch.setattr(
        probability_module,
        "_live_hko_forecast_candidates",
        lambda _target_date, _profile: [candidate, fnd_candidate],
    )
    monkeypatch.setattr(
        probability_module,
        "_now_utc",
        lambda: datetime(2026, 7, 7, 11, 20, tzinfo=UTC),
    )

    forecast = live_cutoff_forecast(date(2026, 7, 8), "t_minus_1_1900_hkt")

    assert forecast.source == "HKO flw local forecast live cutoff fetch"
    assert forecast.as_of_profile == "t_minus_1_1900_hkt"
    assert forecast.forecast_max_c == 30.0
    assert forecast.raw["live_cutoff_fetch"] is True

    with pytest.raises(ForecastUnavailable, match="No live HKO local forecast update is eligible"):
        live_cutoff_forecast(date(2026, 7, 8), "t_minus_1_1800_hkt")

    with pytest.raises(ForecastUnavailable, match="not available yet"):
        live_cutoff_forecast(date(2026, 7, 8), "t_minus_1_2000_hkt")


def test_live_cutoff_forecast_allows_target_day_after_hkt_date_roll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = ForecastSnapshot(
        source="HKO flw local forecast",
        update_time_hkt=datetime.fromisoformat("2026-07-08T23:45:00+08:00"),
        target_date=date(2026, 7, 9),
        forecast_min_c=27.0,
        forecast_max_c=31.0,
        as_of_profile="t_minus_1_2359_hkt",
        raw={"fixture": True},
    )
    monkeypatch.setattr(
        probability_module,
        "_live_hko_forecast_candidates",
        lambda _target_date, _profile: [candidate],
    )
    monkeypatch.setattr(
        probability_module,
        "_now_utc",
        lambda: datetime(2026, 7, 8, 20, 20, tzinfo=UTC),
    )

    forecast = live_cutoff_forecast(date(2026, 7, 9), "t_minus_1_2359_hkt")

    assert forecast.source == "HKO flw local forecast live cutoff fetch"
    assert forecast.as_of_profile == "t_minus_1_2359_hkt"
    assert forecast.raw["selected_update_utc"].isoformat() == "2026-07-08T15:45:00+00:00"

    with pytest.raises(ForecastUnavailable, match="already past in HKT"):
        live_cutoff_forecast(date(2026, 7, 8), "t_minus_1_2359_hkt")


class _EmptyHistoricalCursor:
    def __enter__(self) -> _EmptyHistoricalCursor:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def execute(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def fetchone(self) -> None:
        return None


class _EmptyHistoricalConnection:
    def cursor(self) -> _EmptyHistoricalCursor:
        return _EmptyHistoricalCursor()


class _StoredLiveSnapshotCursor:
    def __init__(self) -> None:
        self.last_sql = ""
        self.last_params: dict[str, Any] = {}

    def __enter__(self) -> _StoredLiveSnapshotCursor:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> None:
        self.last_sql = sql
        self.last_params = params or {}

    def fetchone(self) -> dict[str, Any] | None:
        if "demo_trading.market_snapshot" not in self.last_sql:
            return None
        return {
            "id": 42,
            "as_of_profile": "t_minus_1_1900_hkt",
            "forecast_source": "HKO flw local forecast live cutoff fetch",
            "forecast_update_time_hkt": datetime.fromisoformat("2026-07-08T18:45:00+08:00"),
            "forecast_min_c": 27.0,
            "forecast_max_c": 32.0,
            "forecast": {
                "raw": {
                    "live_cutoff_fetch": True,
                    "cutoff_utc": "2026-07-08T11:00:00+00:00",
                    "source_payload": {"fixture": True},
                }
            },
        }


class _StoredLiveSnapshotConnection:
    def __init__(self) -> None:
        self.cursor_instance = _StoredLiveSnapshotCursor()

    def cursor(self) -> _StoredLiveSnapshotCursor:
        return self.cursor_instance


def test_historical_cutoff_forecast_falls_back_to_safe_live_forecast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback = ForecastSnapshot(
        source="HKO flw local forecast live cutoff fetch",
        update_time_hkt=datetime.fromisoformat("2026-07-07T18:45:00+08:00"),
        target_date=date(2026, 7, 8),
        forecast_min_c=26.0,
        forecast_max_c=30.0,
        as_of_profile="t_minus_1_1900_hkt",
        raw={"live_cutoff_fetch": True},
    )
    calls: list[tuple[date, str]] = []

    def fake_live_cutoff(target_date: date, as_of_profile: str) -> ForecastSnapshot:
        calls.append((target_date, as_of_profile))
        return fallback

    monkeypatch.setattr(probability_module, "live_cutoff_forecast", fake_live_cutoff)

    forecast = historical_cutoff_forecast(
        _EmptyHistoricalConnection(),
        date(2026, 7, 8),
        "t_minus_1_1900_hkt",
    )

    assert forecast is fallback
    assert calls == [(date(2026, 7, 8), "t_minus_1_1900_hkt")]


def test_historical_cutoff_forecast_reuses_stored_live_forecast_after_live_api_moves_past_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_live_cutoff(_target_date: date, _profile: str) -> ForecastSnapshot:
        raise ForecastUnavailable("No live HKO local forecast update is eligible")

    connection = _StoredLiveSnapshotConnection()
    monkeypatch.setattr(probability_module, "live_cutoff_forecast", unavailable_live_cutoff)
    monkeypatch.setattr(
        probability_module,
        "_now_utc",
        lambda: datetime(2026, 7, 8, 20, 20, tzinfo=UTC),
    )

    forecast = historical_cutoff_forecast(
        connection,
        date(2026, 7, 9),
        "t_minus_1_2359_hkt",
    )

    assert forecast.source == "HKO flw local forecast live cutoff fetch"
    assert forecast.as_of_profile == "t_minus_1_2359_hkt"
    assert forecast.update_time_hkt.isoformat() == "2026-07-08T18:45:00+08:00"
    assert forecast.forecast_min_c == 27.0
    assert forecast.forecast_max_c == 32.0
    assert forecast.raw["stored_cutoff_reuse"] is True
    assert forecast.raw["stored_snapshot_id"] == 42
    assert forecast.raw["stored_snapshot_profile"] == "t_minus_1_1900_hkt"
    assert forecast.raw["stored_source_cutoff_utc"] == "2026-07-08T11:00:00+00:00"
    assert forecast.raw["cutoff_utc"].isoformat() == "2026-07-08T15:59:00+00:00"
    assert connection.cursor_instance.last_params["cutoff_utc"].isoformat() == "2026-07-08T15:59:00+00:00"


def test_compute_b4_probabilities_uses_only_prior_primary_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_minimal_probability_config(tmp_path)
    modeling = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2026-07-04", "2026-07-05", "2026-07-06"]),
            "is_primary_cutoff": [False, True, True],
            "bucket_index": [6, 7, 8],
        }
    )
    captured: dict[str, Any] = {}

    def fake_modeling_table(_repo_root: str, _database_url: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        return modeling.copy(), {"bucket_rule_examples": {"31.9": "31", "32.0": "32"}}

    def fake_select_b4_alphas(train: pd.DataFrame, _config: dict[str, Any]) -> tuple[float, float, dict[str, Any]]:
        captured["train"] = train.copy()
        return 20.0, 10.0, {"selection": "test"}

    def fake_predict(
        train: pd.DataFrame,
        validation: pd.DataFrame,
        month_alpha: float,
        cell_alpha: float,
    ) -> np.ndarray:
        captured["predict_train"] = train.copy()
        captured["validation"] = validation.copy()
        captured["alphas"] = (month_alpha, cell_alpha)
        probabilities = np.array([0.0, 0.0, 0.0, 0.01, 0.04, 0.10, 0.25, 0.30, 0.20, 0.08, 0.02])
        return np.array([probabilities])

    monkeypatch.setattr(probability_module, "_modeling_table", fake_modeling_table)
    monkeypatch.setattr(probability_module, "select_b4_alphas", fake_select_b4_alphas)
    monkeypatch.setattr(probability_module, "hierarchical_month_forecast_pmf_predict", fake_predict)

    model = compute_b4_probabilities(
        repo_root=tmp_path,
        database_url="postgresql://example",
        target_date=date(2026, 7, 6),
        forecast_min_c=27.0,
        forecast_max_c=31.95,
    )

    assert captured["train"]["target_date"].dt.date.tolist() == [date(2026, 7, 5)]
    assert captured["predict_train"]["target_date"].dt.date.tolist() == [date(2026, 7, 5)]
    validation_row = captured["validation"].iloc[0]
    assert validation_row["target_month"] == 7
    assert validation_row["forecast_max_tenths"] == 320
    assert validation_row["official_max_round"] == 32
    assert validation_row["forecast_range_c"] == pytest.approx(4.95)
    assert captured["alphas"] == (20.0, 10.0)
    assert model["bucket_keys"] == list(BUCKET_KEYS)
    assert list(model["probabilities"]) == list(BUCKET_KEYS)
    assert sum(model["probabilities"].values()) == pytest.approx(1.0)
    assert all(0.0 <= value <= 1.0 for value in model["probabilities"].values())
    forbidden_trading_fields = {"market", "edge", "ev", "price"}
    assert forbidden_trading_fields.isdisjoint(model)
    assert model["cutoff_profile"] == "t_minus_1_2359_hkt"


def test_compute_b4_probabilities_filters_training_rows_by_selected_cutoff_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_minimal_probability_config(tmp_path)
    modeling = pd.DataFrame(
        {
            "target_date": pd.to_datetime(
                [
                    "2026-07-04",
                    "2026-07-04",
                    "2026-07-05",
                    "2026-07-05",
                    "2026-07-06",
                ]
            ),
            "is_primary_cutoff": [False, True, False, True, False],
            "cutoff_profile": [
                "t_minus_1_2100_hkt",
                "t_minus_1_2359_hkt",
                "t_minus_1_2100_hkt",
                "t_minus_1_2359_hkt",
                "t_minus_1_2100_hkt",
            ],
            "bucket_index": [6, 7, 8, 9, 10],
        }
    )
    captured: dict[str, Any] = {}

    def fake_modeling_table(_repo_root: str, _database_url: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        return modeling.copy(), {}

    def fake_select_b4_alphas(train: pd.DataFrame, _config: dict[str, Any]) -> tuple[float, float, dict[str, Any]]:
        captured["train"] = train.copy()
        return 20.0, 10.0, {}

    def fake_predict(
        train: pd.DataFrame,
        validation: pd.DataFrame,
        month_alpha: float,
        cell_alpha: float,
    ) -> np.ndarray:
        captured["predict_train"] = train.copy()
        probabilities = np.array([0.0, 0.0, 0.0, 0.01, 0.04, 0.10, 0.25, 0.30, 0.20, 0.08, 0.02])
        return np.array([probabilities])

    monkeypatch.setattr(probability_module, "_modeling_table", fake_modeling_table)
    monkeypatch.setattr(probability_module, "select_b4_alphas", fake_select_b4_alphas)
    monkeypatch.setattr(probability_module, "hierarchical_month_forecast_pmf_predict", fake_predict)

    model = compute_b4_probabilities(
        repo_root=tmp_path,
        database_url="postgresql://example",
        target_date=date(2026, 7, 6),
        forecast_min_c=27.0,
        forecast_max_c=31.0,
        cutoff_profile="t_minus_1_2100_hkt",
    )

    assert model["cutoff_profile"] == "t_minus_1_2100_hkt"
    assert captured["train"]["cutoff_profile"].tolist() == [
        "t_minus_1_2100_hkt",
        "t_minus_1_2100_hkt",
    ]
    assert captured["predict_train"]["target_date"].dt.date.tolist() == [
        date(2026, 7, 4),
        date(2026, 7, 5),
    ]


def test_compute_b4_probabilities_rejects_missing_prior_training_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_minimal_probability_config(tmp_path)
    modeling = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2026-07-06"]),
            "is_primary_cutoff": [True],
            "bucket_index": [8],
        }
    )
    monkeypatch.setattr(
        probability_module,
        "_modeling_table",
        lambda _repo_root, _database_url: (modeling.copy(), {}),
    )

    with pytest.raises(ForecastUnavailable, match="No B4 training rows for t_minus_1_2359_hkt before 2026-07-06"):
        compute_b4_probabilities(
            repo_root=tmp_path,
            database_url="postgresql://example",
            target_date=date(2026, 7, 6),
            forecast_min_c=27.0,
            forecast_max_c=31.0,
        )


def test_real_db_b4_probabilities_match_july6_edge_artifact(repo_root: Path) -> None:
    artifact_path = repo_root / "experiments" / "hkg_tmax_live_market_edges" / "latest_edge_report.json"
    if not artifact_path.exists():
        pytest.skip("July 6 live edge artifact is unavailable")

    try:
        with psycopg.connect(DEFAULT_DATABASE_URL, connect_timeout=3) as connection, connection.cursor() as cursor:
            cursor.execute("SELECT 1")
    except psycopg.Error as exc:
        pytest.skip(f"Postgres unavailable for real probability sanity test: {exc}")

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    probability_module._modeling_table.cache_clear()
    model = compute_b4_probabilities(
        repo_root=repo_root,
        database_url=DEFAULT_DATABASE_URL,
        target_date=date.fromisoformat(artifact["event"]["target_date"]),
        forecast_min_c=artifact["forecast"]["forecast_min_c"],
        forecast_max_c=artifact["forecast"]["forecast_max_c"],
    )

    assert model["method"] == "B4_hierarchical_residual_pmf"
    assert model["train_rows"] == artifact["model"]["train_rows"] == 9644
    assert model["train_start"] == artifact["model"]["train_start"] == "2000-01-02"
    assert model["train_end"] == artifact["model"]["train_end"] == "2026-05-31"
    assert list(model["probabilities"]) == list(BUCKET_KEYS)
    assert sum(model["probabilities"].values()) == pytest.approx(1.0, abs=1e-12)
    assert all(0.0 <= value <= 1.0 for value in model["probabilities"].values())
    for bucket in BUCKET_KEYS:
        assert model["probabilities"][bucket] == pytest.approx(
            artifact["model"]["probabilities"][bucket],
            abs=1e-12,
        )
    assert model["probabilities"]["32"] == pytest.approx(0.22826367229339364, abs=1e-12)
