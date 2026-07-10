from __future__ import annotations

import subprocess
from pathlib import Path

import hkg_tmax.doctor as doctor_module


def _write_minimal_project(root: Path) -> None:
    (root / "config" / "project").mkdir(parents=True)
    (root / "config" / "sources").mkdir(parents=True)
    (root / "experiments").mkdir()
    (root / "src" / "hkg_tmax").mkdir(parents=True)
    (root / "config" / "project" / "project.yaml").write_text("name: test\n")
    (root / "config" / "sources" / "data_sources.yaml").write_text("sources: []\n")


def test_doctor_accepts_containing_monorepo(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "projects" / "hkg-tmax"
    _write_minimal_project(project_root)
    monkeypatch.setattr(doctor_module.shutil, "which", lambda command: "/usr/bin/git")
    monkeypatch.setattr(
        doctor_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout=f"{tmp_path}\n", stderr=""
        ),
    )

    checks, warnings = doctor_module.doctor(project_root)

    assert f"Git repository detected: {tmp_path}" in checks
    assert not any("Git repository not detected" in warning for warning in warnings)


def test_doctor_warns_when_project_is_outside_git(tmp_path: Path, monkeypatch) -> None:
    _write_minimal_project(tmp_path)
    monkeypatch.setattr(doctor_module.shutil, "which", lambda command: "/usr/bin/git")
    monkeypatch.setattr(
        doctor_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=128, stdout="", stderr="not a git repository"
        ),
    )

    _, warnings = doctor_module.doctor(tmp_path)

    assert "Git repository not detected; initialize before accepted experiments." in warnings


def test_doctor_warns_when_git_probe_times_out(tmp_path: Path, monkeypatch) -> None:
    _write_minimal_project(tmp_path)
    monkeypatch.setattr(doctor_module.shutil, "which", lambda command: "/usr/bin/git")

    def time_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=5)

    monkeypatch.setattr(doctor_module.subprocess, "run", time_out)

    _, warnings = doctor_module.doctor(tmp_path)

    assert "Git repository not detected; initialize before accepted experiments." in warnings
