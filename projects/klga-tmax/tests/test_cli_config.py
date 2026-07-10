from __future__ import annotations

from typer.testing import CliRunner

from klga_tmax.cli import app
from klga_tmax.constants import EXIT_CONFIG_ERROR


def test_help_does_not_require_database_url(monkeypatch) -> None:
    monkeypatch.delenv("KLGA_DB_URL", raising=False)
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0


def test_db_command_without_database_url_exits_10(monkeypatch) -> None:
    monkeypatch.delenv("KLGA_DB_URL", raising=False)
    result = CliRunner().invoke(app, ["db", "inspect-contract"])
    assert result.exit_code == EXIT_CONFIG_ERROR
    assert "KLGA_DB_URL is required" in result.stderr


def test_wunderground_inspect_config_does_not_require_database_or_api_key(monkeypatch) -> None:
    monkeypatch.delenv("KLGA_DB_URL", raising=False)
    monkeypatch.delenv("WUNDERGROUND_API_KEY", raising=False)
    monkeypatch.delenv("WEATHERCOM_API_KEY", raising=False)
    result = CliRunner().invoke(app, ["wunderground", "inspect-config"])
    assert result.exit_code == 0
    assert '"api_key_present": false' in result.stdout


def test_strategy_feature_materialize_help_exposes_feature_version(monkeypatch) -> None:
    monkeypatch.delenv("KLGA_DB_URL", raising=False)
    result = CliRunner().invoke(app, ["features", "materialize", "--help"])
    assert result.exit_code == 0
    assert "--feature-version" in result.stdout


def test_settlement_update_is_nested_command(monkeypatch) -> None:
    monkeypatch.delenv("KLGA_DB_URL", raising=False)
    result = CliRunner().invoke(app, ["settlement", "update", "--help"])
    assert result.exit_code == 0
    assert "--start-date" in result.stdout
