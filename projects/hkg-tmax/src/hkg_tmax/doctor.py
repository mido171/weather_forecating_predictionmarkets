from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from .config import SourceCatalog, load_yaml


def doctor(root: Path) -> tuple[list[str], list[str]]:
    ok: list[str] = []
    warnings: list[str] = []

    ok.append(f"Python: {platform.python_version()}")

    for command in ("git",):
        resolved = shutil.which(command)
        if resolved:
            ok.append(f"{command}: {resolved}")
        else:
            warnings.append(f"{command} not found")

    _ = load_yaml(root / "config" / "project" / "project.yaml")
    ok.append("project config readable")
    catalog = SourceCatalog.from_path(root / "config" / "sources" / "data_sources.yaml")
    ok.append(f"source catalog readable: {len(catalog.sources)} sources")

    for relative in ("config", "experiments", "src/hkg_tmax"):
        path = root / relative
        if not path.exists():
            raise RuntimeError(f"Missing required path: {path}")
        ok.append(f"path exists: {relative}")

    if not os.getenv("HKG_TMAX_USER_AGENT"):
        warnings.append(
            "HKG_TMAX_USER_AGENT is unset; copy .env.example to .env and add contact details."
        )
    if resolved := shutil.which("git"):
        try:
            result = subprocess.run(
                [resolved, "-C", str(root), "rev-parse", "--show-toplevel"],
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            warnings.append("Git repository not detected; initialize before accepted experiments.")
        else:
            if result.returncode == 0 and result.stdout.strip():
                ok.append(f"Git repository detected: {result.stdout.strip()}")
            else:
                warnings.append("Git repository not detected; initialize before accepted experiments.")

    return ok, warnings
