from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_script(script_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script_name), *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_iem_mos_materializer_requires_execute_before_database() -> None:
    result = _run_script(
        "materialize_iem_mos_features.py",
        "--start-date",
        "2026-07-01",
        "--through-date",
        "2026-07-01",
    )

    assert result.returncode == 2
    assert "re-run with --execute" in result.stderr


def test_iem_mos_materializer_rejects_date_scope_over_budget() -> None:
    result = _run_script(
        "materialize_iem_mos_features.py",
        "--start-date",
        "2026-07-01",
        "--through-date",
        "2026-08-01",
        "--max-days",
        "31",
        "--db-url",
        "postgresql://placeholder.invalid/db",
        "--execute",
    )

    assert result.returncode == 2
    assert "date scope contains 32 days" in result.stderr


def test_gribstream_fast_backfill_requires_execute_before_external_state() -> None:
    result = _run_script(
        "run_gribstream_t1245_runs_fast_backfill.py",
        "--start-date",
        "2026-07-01",
        "--end-date",
        "2026-07-01",
        "--models",
        "gfs",
        "--coordinate-tier",
        "B",
    )

    assert result.returncode == 2
    assert "re-run with --execute" in result.stderr


def test_gribstream_fast_backfill_rejects_spacing_below_floor() -> None:
    result = _run_script(
        "run_gribstream_t1245_runs_fast_backfill.py",
        "--start-date",
        "2026-07-01",
        "--end-date",
        "2026-07-01",
        "--models",
        "gfs",
        "--coordinate-tier",
        "B",
        "--spacing-seconds",
        "11.9",
        "--database-url",
        "postgresql://placeholder.invalid/db",
        "--execute",
    )

    assert result.returncode == 2
    assert "--spacing-seconds must be >= 12" in result.stderr


def test_gribstream_fastpath_probe_requires_execute_before_credentials() -> None:
    result = _run_script(
        "probe_gribstream_t1245_runs_fastpath.py",
        "--start-date",
        "2026-07-01",
        "--end-date",
        "2026-07-01",
        "--models",
        "gfs",
    )

    assert result.returncode == 2
    assert "re-run with --execute" in result.stderr


def test_gribstream_fastpath_probe_rejects_all_models_before_credentials() -> None:
    result = _run_script(
        "probe_gribstream_t1245_runs_fastpath.py",
        "--start-date",
        "2026-07-01",
        "--end-date",
        "2026-07-01",
        "--models",
        "all",
        "--execute",
    )

    assert result.returncode == 2
    assert "must list explicit model IDs" in result.stderr


def test_gribstream_bias_backtest_requires_execute_before_database() -> None:
    result = _run_script("run_gribstream_bias_backtest.py")

    assert result.returncode == 2
    assert "re-run with --execute" in result.stderr


def test_all_model_experiment_requires_execute_before_database(tmp_path: Path) -> None:
    result = _run_script(
        "../experiments/klga_all_model_strategy_suite_eval.py",
        "--output-dir",
        str(tmp_path),
    )

    assert result.returncode == 2
    assert "re-run with --execute" in result.stderr


def test_nbm_residual_experiment_requires_execute_before_database() -> None:
    result = _run_script("../experiments/nbm_residual_stacker_eval.py")

    assert result.returncode == 2
    assert "re-run with --execute" in result.stderr
