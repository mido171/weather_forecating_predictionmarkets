from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "scripts/run_t07_t13_gribstream_backfill.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_t07_t13_gribstream_backfill", RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_expression_backed_shared_parameter_is_recorded_as_derived_selector() -> None:
    runner = load_runner()
    fragment = {
        "variables": [
            {"name": "TMP", "level": "2 m above ground", "alias": "temperature_2m", "hidden": True},
            {"name": "DPT", "level": "2 m above ground", "alias": "dew_point_2m", "hidden": True},
        ],
        "expressions": [
            {"alias": "relative_humidity_2m", "expression": "rh(TMP,DPT)", "hidden": False},
        ],
    }

    selector = runner.selector_from_fragment(
        dataset="gfs",
        shared_parameter="relative_humidity_2m",
        fragment=fragment,
        source_payload={"resolved_request": fragment},
        unit="%",
        retrieved_at_utc="2026-06-24T00:00:00Z",
    )

    assert selector.native_name == "EXPR:relative_humidity_2m"
    assert selector.native_level == "derived"
    assert selector.alias == "relative_humidity_2m"


def test_credit_estimate_includes_runs_leads_parameters_members_and_coordinate_blocks() -> None:
    runner = load_runner()
    payload = {
        "forecastedFrom": "2026-06-23T00:00:00Z",
        "forecastedUntil": "2026-06-23T18:00:00Z",
        "minLeadTime": "0h",
        "maxLeadTime": "84h",
        "variables": [{"name": "TMP"}, {"name": "DPT"}],
        "expressions": [{"alias": "relative_humidity_2m"}],
        "members": [0, 1, 2, 3, 4],
    }

    assert runner.estimate_credits(payload, coordinate_count=132) == 4 * 85 * 3 * 5
    assert runner.estimate_credits(payload, coordinate_count=501) == 4 * 85 * 3 * 5 * 2


def test_member_chunking_and_task_windows_are_explicit() -> None:
    runner = load_runner()

    assert runner.chunks(list(range(12)), 5) == [(0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (10, 11)]
    assert runner.DATASETS["gfs"]["archive_start"] == "2021-03-22"
    assert runner.DATASETS["gefsatmos"]["archive_start"] == "2020-10-01"
    assert runner.DATASETS["ifsoper"]["archive_start"] == "2024-02-28"
    assert runner.DATASETS["cwawrf15"]["archive_start"] == "2026-06-21"
    assert runner.DATASETS["cwawrf15"]["archive_end"] == "2026-06-23"


def test_t13_is_not_treated_as_a_gribstream_dataset() -> None:
    runner = load_runner()

    assert "hko_arwf" not in runner.DATASETS
    assert "T13" not in {config["task_id"] for config in runner.DATASETS.values()}


def test_project_credential_file_beats_legacy_token_env(tmp_path: Path, monkeypatch) -> None:
    runner = load_runner()
    credential_file = tmp_path / "gribstream.env"
    credential_file.write_text("GRIBSTREAM_API_KEY=file-token-123\n", encoding="utf-8")
    monkeypatch.setattr(runner, "SECRET_FILE", credential_file)
    monkeypatch.delenv("GRIBSTREAM_API_KEY", raising=False)
    monkeypatch.setenv("GRIBSTREAM_API_TOKEN", "stale-env-token")

    assert runner.load_gribstream_token() == "file-token-123"
