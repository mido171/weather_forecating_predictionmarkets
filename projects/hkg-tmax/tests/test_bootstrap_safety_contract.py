from pathlib import Path

import pytest


@pytest.mark.parametrize("script_name", ["bootstrap.sh", "bootstrap.ps1"])
def test_bootstrap_fails_closed_on_repository_identity_and_git_safety(
    repo_root: Path,
    script_name: str,
) -> None:
    text = (repo_root / "scripts" / script_name).read_text(encoding="utf-8")
    lowered = text.lower()

    assert "git init" not in lowered
    assert "codex_start_here.md" not in lowered
    assert "first_goals.md" not in lowered
    assert "weather_data_extraction" in text
    assert "rev-parse" in text
    assert "--show-toplevel" in text
    assert "core.fsmonitor" in text
    assert "false" in lowered
    assert ".git" in text


@pytest.mark.parametrize("script_name", ["bootstrap.sh", "bootstrap.ps1"])
def test_bootstrap_uses_serial_bounded_verification_without_manifest_churn(
    repo_root: Path,
    script_name: str,
) -> None:
    text = (repo_root / "scripts" / script_name).read_text(encoding="utf-8")

    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert variable in text
    for focused_test in (
        "tests/test_bootstrap_safety_contract.py",
        "tests/test_config_and_sources.py",
        "tests/test_experiments.py",
        "tests/test_validation.py",
        "tests/test_hko_backfill.py",
        "tests/hkg_t24/test_h24n_contract_policy.py",
        "tests/hkg_t24/test_schema_sql_contract.py",
        "tests/test_demo_trading_migration.py",
    ):
        assert focused_test in text
    assert "pytest -n" not in text
    assert "pip install --upgrade pip" not in text
    assert "hkg_tmax manifest" not in text
    assert "tools/repo/doctor.py" in text.replace("\\", "/")
    assert "manage_campaign_documentation.py" in text


def test_pyproject_is_the_single_python_dependency_authority(repo_root: Path) -> None:
    assert not (repo_root / "requirements.txt").exists()
    assert not (repo_root / "requirements-dev.txt").exists()

    dockerfile = (repo_root / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY pyproject.toml README.md ./" in dockerfile
    assert "requirements.txt" not in dockerfile
    assert "requirements-dev.txt" not in dockerfile
    assert "pip install --no-cache-dir --upgrade pip" not in dockerfile


def test_generated_milestones_link_the_current_production_gate(repo_root: Path) -> None:
    renderer = (repo_root / "src" / "hkg_tmax" / "milestones.py").read_text(encoding="utf-8")

    assert "docs/operations/PRODUCTION_GATE.md" in renderer
    assert "docs/07_PRODUCTION_GATE.md" not in renderer
