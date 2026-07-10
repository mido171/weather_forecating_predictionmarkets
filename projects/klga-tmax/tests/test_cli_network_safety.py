from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import klga_tmax.cli as cli
from klga_tmax.providers.gribstream.config import GribStreamSettings


RUNNER = CliRunner()


def _normalized_output(result) -> str:
    box_characters = "│─┌┐└┘╭╮╰╯"
    without_box = result.output.translate(str.maketrans({char: " " for char in box_characters}))
    return " ".join(without_box.split())


def _unexpected_call(name: str):
    def fail(*args, **kwargs):
        pytest.fail(f"{name} was called before the CLI safety boundary")

    return fail


def _gribstream_run_args() -> list[str]:
    return [
        "gribstream",
        "run",
        "--start-date",
        "2026-07-01",
        "--end-date",
        "2026-07-01",
        "--models",
        "gfs",
        "--cutoff-id",
        "T_MINUS_1_2045UTC",
        "--coordinate-tier",
        "B",
    ]


def test_gribstream_run_requires_execute_before_settings_or_database(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_gribstream_settings_or_exit", _unexpected_call("settings"))
    monkeypatch.setattr(cli, "_run_audited", _unexpected_call("database audit"))

    result = RUNNER.invoke(cli.app, _gribstream_run_args())

    assert result.exit_code == 2
    assert "re-run with --execute" in result.output


def test_gribstream_plan_dry_run_makes_zero_external_calls(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_gribstream_settings_or_exit", _unexpected_call("settings"))
    monkeypatch.setattr(cli, "_run_audited", _unexpected_call("database audit"))
    monkeypatch.setattr(cli, "prepare_gribstream_plan", _unexpected_call("provider plan"))

    result = RUNNER.invoke(
        cli.app,
        [
            "gribstream",
            "plan",
            "--start-date",
            "2026-07-01",
            "--end-date",
            "2026-07-01",
            "--models",
            "gfs",
            "--cutoff-id",
            "T_MINUS_1_2045UTC",
            "--coordinate-tier",
            "B",
            "--execute",
        ],
    )

    assert result.exit_code == 0
    assert '"dry_run": true' in result.stdout
    assert '"provider_calls": 0' in result.stdout
    assert '"database_calls": 0' in result.stdout


def test_gribstream_execute_defaults_to_one_chunk_and_twelve_second_spacing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = GribStreamSettings(
        api_token="dummy-token",
        base_url="https://example.invalid/api/v2",
        artifact_root=tmp_path,
        timeout_seconds=1,
        spacing_seconds=12,
        max_retries=0,
        user_agent="test",
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(cli, "_gribstream_settings_or_exit", lambda **kwargs: settings)
    monkeypatch.setattr(cli, "_run_audited", lambda **kwargs: captured.update(kwargs))

    result = RUNNER.invoke(cli.app, [*_gribstream_run_args(), "--execute"])

    assert result.exit_code == 0
    command_args = captured["command_args"]
    assert isinstance(command_args, dict)
    assert command_args["max_chunks"] == 1
    assert command_args["max_models"] == 1
    assert command_args["spacing_seconds"] == 12.0


def test_gribstream_catalog_and_data_requests_share_twelve_second_limiter(
    monkeypatch,
) -> None:
    limiter_events: list[object] = []

    class FakeLimiter:
        def __init__(self, *, spacing_seconds: float) -> None:
            limiter_events.append(("spacing", spacing_seconds))

        def acquire(self) -> None:
            limiter_events.append("acquire")

    def catalog_open(*args, **kwargs):
        return "catalog"

    def client_open(*args, **kwargs):
        return "client"

    monkeypatch.setattr(cli, "OneThreadRateLimiter", FakeLimiter)
    monkeypatch.setattr(cli.gribstream_catalog, "urlopen", catalog_open)
    monkeypatch.setattr(cli.gribstream_client, "urlopen", client_open)

    with cli._spaced_gribstream_requests(12.0):
        assert cli.gribstream_catalog.urlopen("catalog-request") == "catalog"
        assert cli.gribstream_client.urlopen("data-request") == "client"

    assert limiter_events == [("spacing", 12.0), "acquire", "acquire"]
    assert cli.gribstream_catalog.urlopen is catalog_open
    assert cli.gribstream_client.urlopen is client_open


@pytest.mark.parametrize(
    "extra_args, expected_text",
    [
        (["--execute", "--spacing-seconds", "11.9"], "12"),
        (["--execute", "--models", "all"], "must list explicit model IDs"),
        (
            ["--execute", "--end-date", "2026-07-02"],
            "date scope contains 2 target days but --max-chunks is 1",
        ),
    ],
)
def test_gribstream_rejects_unsafe_scope_before_settings(
    monkeypatch,
    extra_args: list[str],
    expected_text: str,
) -> None:
    monkeypatch.setattr(cli, "_gribstream_settings_or_exit", _unexpected_call("settings"))
    args = _gribstream_run_args()
    for option in {"--models", "--end-date"}:
        if option in extra_args:
            existing_index = args.index(option)
            del args[existing_index : existing_index + 2]

    result = RUNNER.invoke(cli.app, [*args, *extra_args])

    assert result.exit_code == 2
    assert expected_text in _normalized_output(result)


def test_polymarket_requires_execute_before_analysis(monkeypatch) -> None:
    monkeypatch.setattr(cli, "run_cutoff_analysis", _unexpected_call("Polymarket analysis"))

    result = RUNNER.invoke(
        cli.app,
        [
            "polymarket",
            "cutoff-analysis",
            "--start-date",
            "2026-07-01",
            "--end-date",
            "2026-07-01",
        ],
    )

    assert result.exit_code == 2
    assert "re-run with --execute" in result.output


def test_polymarket_default_artifacts_use_external_settings_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "_offline_settings_or_exit",
        lambda: SimpleNamespace(artifact_root=tmp_path),
    )
    monkeypatch.setattr(
        cli,
        "run_cutoff_analysis",
        lambda **kwargs: captured.update(kwargs) or {"ok": True},
    )

    result = RUNNER.invoke(
        cli.app,
        [
            "polymarket",
            "cutoff-analysis",
            "--start-date",
            "2026-07-01",
            "--end-date",
            "2026-07-01",
            "--max-requests",
            "1",
            "--execute",
        ],
    )

    assert result.exit_code == 0
    assert captured["artifact_root"] == tmp_path / "polymarket_cutoff_analysis"


def test_polymarket_http_attempt_budget_is_enforced(
    monkeypatch,
    tmp_path: Path,
) -> None:
    outbound_calls: list[str] = []

    def fake_request(session, method, url, **kwargs):
        outbound_calls.append(str(url))
        return object()

    def fake_analysis(**kwargs):
        session = cli.requests.Session()
        session.request("GET", "https://example.invalid/first")
        session.request("GET", "https://example.invalid/second")
        return {"ok": True}

    monkeypatch.setattr(cli.requests.sessions.Session, "request", fake_request)
    monkeypatch.setattr(cli, "run_cutoff_analysis", fake_analysis)
    monkeypatch.setattr(
        cli,
        "_offline_settings_or_exit",
        lambda: SimpleNamespace(artifact_root=tmp_path),
    )

    result = RUNNER.invoke(
        cli.app,
        [
            "polymarket",
            "cutoff-analysis",
            "--start-date",
            "2026-07-01",
            "--end-date",
            "2026-07-01",
            "--max-requests",
            "1",
            "--execute",
        ],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert "request budget exhausted after 1" in str(result.exception)
    assert outbound_calls == ["https://example.invalid/first"]


def test_wunderground_smoke_dry_run_never_loads_credentials_or_calls_java(
    monkeypatch,
) -> None:
    monkeypatch.setattr(cli, "_wunderground_settings_or_exit", _unexpected_call("WU settings"))
    monkeypatch.setattr(cli, "_run_audited", _unexpected_call("database audit"))
    monkeypatch.setattr(cli, "run_java_wu_truth", _unexpected_call("Java provider runner"))

    result = RUNNER.invoke(
        cli.app,
        [
            "wunderground",
            "smoke",
            "--station-id",
            "KLGA",
            "--local-date",
            "2026-07-01",
            "--execute",
        ],
    )

    assert result.exit_code == 0
    assert '"provider_calls": 0' in result.stdout
    assert '"database_calls": 0' in result.stdout


def test_wunderground_persist_requires_execute_before_settings(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_wunderground_settings_or_exit", _unexpected_call("WU settings"))
    monkeypatch.setattr(cli, "_run_audited", _unexpected_call("database audit"))

    result = RUNNER.invoke(
        cli.app,
        [
            "wunderground",
            "fetch-day",
            "--station-id",
            "KLGA",
            "--local-date",
            "2026-07-01",
            "--persist",
        ],
    )

    assert result.exit_code == 2
    assert "re-run with --execute" in result.output


def test_wunderground_single_day_rejects_unbounded_station_scope(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_wunderground_settings_or_exit", _unexpected_call("WU settings"))

    result = RUNNER.invoke(
        cli.app,
        [
            "wunderground",
            "smoke",
            "--station-id",
            "all",
            "--local-date",
            "2026-07-01",
            "--persist",
            "--execute",
        ],
    )

    assert result.exit_code == 2
    assert "must list explicit station IDs" in _normalized_output(result)


def test_wunderground_backfill_dry_run_is_bounded_and_zero_call(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_wunderground_settings_or_exit", _unexpected_call("WU settings"))
    monkeypatch.setattr(cli, "_run_audited", _unexpected_call("database audit"))

    result = RUNNER.invoke(
        cli.app,
        [
            "wunderground",
            "backfill",
            "--start-date",
            "2026-07-01",
            "--end-date",
            "2026-07-31",
            "--stations",
            "KLGA",
        ],
    )

    assert result.exit_code == 0
    assert '"planned_requests": 1' in result.stdout
    assert '"provider_calls": 0' in result.stdout


def test_wunderground_long_range_requires_separate_bounded_override(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_wunderground_settings_or_exit", _unexpected_call("WU settings"))
    base_args = [
        "wunderground",
        "backfill",
        "--start-date",
        "2026-07-01",
        "--end-date",
        "2026-08-01",
        "--stations",
        "KLGA",
    ]

    without_override = RUNNER.invoke(cli.app, base_args)
    bounded_override = RUNNER.invoke(
        cli.app,
        [*base_args, "--allow-long-date-range", "--max-requests", "2"],
    )

    assert without_override.exit_code == 2
    assert "default maximum is 31" in without_override.output
    assert bounded_override.exit_code == 0
    assert '"planned_requests": 2' in bounded_override.stdout
    assert '"provider_calls": 0' in bounded_override.stdout


def test_wunderground_resume_budgets_fragmented_missing_days_at_worst_case(
    monkeypatch,
) -> None:
    monkeypatch.setattr(cli, "_wunderground_settings_or_exit", _unexpected_call("WU settings"))
    args = [
        "wunderground",
        "backfill",
        "--start-date",
        "2026-07-01",
        "--end-date",
        "2026-07-03",
        "--stations",
        "KLGA",
        "--resume",
    ]

    under_budget = RUNNER.invoke(cli.app, args)
    bounded = RUNNER.invoke(cli.app, [*args, "--max-requests", "3"])

    assert under_budget.exit_code == 2
    assert "requires 3 provider request windows" in _normalized_output(under_budget)
    assert bounded.exit_code == 0
    assert '"planned_requests": 3' in bounded.stdout


def test_wunderground_workers_are_capped_at_two(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_wunderground_settings_or_exit", _unexpected_call("WU settings"))

    result = RUNNER.invoke(
        cli.app,
        [
            "wu",
            "rebuild",
            "--start-date",
            "2026-07-01",
            "--end-date",
            "2026-07-01",
            "--stations",
            "KLGA",
            "--workers",
            "3",
            "--execute",
        ],
    )

    assert result.exit_code == 2
    assert "not in the range" in result.output


def test_java_wu_acknowledgement_precedes_settings_and_database(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        cli,
        "_wunderground_settings_or_exit",
        lambda **kwargs: calls.append("settings") or object(),
    )
    monkeypatch.setattr(
        cli,
        "_run_audited",
        lambda **kwargs: calls.append("database-audit"),
    )

    denied = RUNNER.invoke(
        cli.app,
        ["wu", "audit-day", "--station", "KLGA", "--date", "2026-07-01"],
    )
    allowed = RUNNER.invoke(
        cli.app,
        [
            "wu",
            "audit-day",
            "--station",
            "KLGA",
            "--date",
            "2026-07-01",
            "--execute",
        ],
    )

    assert denied.exit_code == 2
    assert calls == ["settings", "database-audit"]
    assert allowed.exit_code == 0


def test_wu_validate_sample_cannot_exceed_request_budget(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_wunderground_settings_or_exit", _unexpected_call("WU settings"))

    result = RUNNER.invoke(
        cli.app,
        [
            "wu",
            "validate-sample",
            "--sample-size",
            "2",
            "--max-requests",
            "1",
            "--execute",
        ],
    )

    assert result.exit_code == 2
    assert "--sample-size is 2 but --max-requests is 1" in _normalized_output(result)
