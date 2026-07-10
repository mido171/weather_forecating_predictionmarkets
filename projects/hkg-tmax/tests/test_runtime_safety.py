from __future__ import annotations

from datetime import UTC, datetime

import pytest

from hkg_tmax import cli, collector
from hkg_tmax.collector import CollectorError
from hkg_tmax.config import load_yaml


@pytest.mark.parametrize(
    "arguments,patched_name",
    [
        (
            ["sources", "fetch", "--id", "hko_latest_1min_temperature"],
            "fetch_sources",
        ),
        (
            [
                "acquisition",
                "collect",
                "--source-id",
                "hko_latest_1min_temperature",
            ],
            "collect_source_ids",
        ),
        (['acquisition', 'run-due'], "run_due_schedules"),
        (['acquisition', 'hko-backfill', '--batch', 'daily-extract'], "run_hko_backfill_batch"),
    ],
)
def test_network_commands_require_execute(repo_root, monkeypatch, arguments, patched_name) -> None:
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("network operation must not be called")

    monkeypatch.setattr(cli, patched_name, forbidden)
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--root", str(repo_root), *arguments])

    assert exc_info.value.code == 2
    assert called is False


def test_collector_schedule_is_globally_disabled(repo_root, monkeypatch) -> None:
    monkeypatch.setenv("HKG_TMAX_ENABLE_COLLECTORS", "1")
    monkeypatch.setattr(
        collector,
        "ensure_data_root",
        lambda *_: pytest.fail("disabled schedule must not initialize storage"),
    )

    assert collector.due_schedules(repo_root, now_utc=datetime.now(UTC)) == []
    with pytest.raises(CollectorError, match="fail-closed"):
        collector.run_due_schedules(repo_root)


def test_every_collector_source_defaults_disabled(repo_root) -> None:
    config = load_yaml(
        repo_root / "config" / "acquisition" / "collector_schedules.yaml"
    )

    assert config["policy"]["enabled"] is False
    assert config["sources"]
    assert all(source["enabled"] is False for source in config["sources"])


def test_worker_environment_defaults_are_bounded(repo_root) -> None:
    text = (repo_root / ".env.example").read_text(encoding="utf-8")

    assert "HKG_TMAX_COLLECTOR_MAX_SOURCES=1" in text
    assert "HKG_TMAX_COLLECTOR_MAX_REQUESTS=1" in text
    assert "OMP_NUM_THREADS=1" in text
    assert "MKL_NUM_THREADS=1" in text
