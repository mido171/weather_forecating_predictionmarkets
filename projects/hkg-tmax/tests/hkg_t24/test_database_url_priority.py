from __future__ import annotations

from pathlib import Path

import pytest
from hkg_t24.cli import main
from hkg_t24.constants import DUAL_DSN_WARNING, MISSING_DSN_ERROR
from hkg_t24.db.connection import DatabaseConfigError, get_database_url


def test_database_url_wins_over_fallback_dsn() -> None:
    messages: list[str] = []

    result = get_database_url(
        {
            "HKG_TMAX_DATABASE_URL": "postgresql://primary.example/db",
            "HKG_TMAX_DB_DSN": "postgresql://fallback.example/db",
        },
        message_sink=messages.append,
    )

    assert result == "postgresql://primary.example/db"
    assert messages == [DUAL_DSN_WARNING]


def test_fallback_dsn_used_when_primary_absent() -> None:
    assert (
        get_database_url({"HKG_TMAX_DB_DSN": "postgresql://fallback.example/db"})
        == "postgresql://fallback.example/db"
    )


def test_missing_database_dsn_error_is_exact() -> None:
    try:
        get_database_url({})
    except DatabaseConfigError as exc:
        assert str(exc) == MISSING_DSN_ERROR
    else:  # pragma: no cover
        raise AssertionError("missing DSN did not raise")


def test_cli_fails_closed_without_dsn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from hkg_t24 import cli

    monkeypatch.delenv("HKG_TMAX_DATABASE_URL", raising=False)
    monkeypatch.delenv("HKG_TMAX_DB_DSN", raising=False)
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)

    assert main(["phase0-preflight"]) == 1
    captured = capsys.readouterr()
    assert MISSING_DSN_ERROR in captured.err
    assert (tmp_path / "reports" / "phase0_preflight_report.md").exists()
    assert (tmp_path / "reports" / "jira_001_contract_coverage.md").exists()
