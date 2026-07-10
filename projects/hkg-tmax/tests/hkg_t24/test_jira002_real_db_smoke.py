from __future__ import annotations

import os
from pathlib import Path

import pytest
from hkg_t24.cli import main


def _skip_without_db() -> None:
    if not os.environ.get("HKG_TMAX_DATABASE_URL") and not os.environ.get("HKG_TMAX_DB_DSN"):
        pytest.skip("SKIPPED_REAL_DB_NO_DATABASE_URL")


def test_real_db_jira002_feature_build_scopes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _skip_without_db()
    from hkg_t24 import cli

    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    for scope in ("strict", "proxy", "live_shadow"):
        assert (
            main(
                [
                    "build-features",
                    "--scope",
                    scope,
                    "--from-date",
                    "2021-04-14",
                    "--to-date",
                    "2021-05-31",
                    "--smoke",
                ]
            )
            == 0
        )


def test_real_db_jira002_expert_oof_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _skip_without_db()
    from hkg_t24 import cli

    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    assert (
        main(
            [
                "build-features",
                "--scope",
                "strict",
                "--from-date",
                "2021-04-14",
                "--to-date",
                "2021-05-31",
                "--smoke",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "train-experts",
                "--scope",
                "strict-pre2024",
                "--smoke",
                "--from-date",
                "2021-04-14",
                "--to-date",
                "2021-05-31",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "generate-oof",
                "--scope",
                "strict-pre2024",
                "--smoke",
                "--from-date",
                "2021-04-14",
                "--to-date",
                "2021-05-31",
            ]
        )
        == 0
    )
