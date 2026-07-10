#!/usr/bin/env python3
"""Check optional project dependencies used by analysis utilities."""
from __future__ import annotations

import importlib
import json
import platform
import sys

PACKAGES = ["numpy", "pandas", "pyarrow"]


def main() -> int:
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {},
    }
    missing = []
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            result["packages"][name] = {
                "available": True,
                "version": getattr(module, "__version__", "unknown"),
            }
        except Exception as exc:
            result["packages"][name] = {
                "available": False,
                "error": str(exc),
            }
            missing.append(name)
    print(json.dumps(result, indent=2))
    if missing:
        print(
            "\nMissing optional analysis packages: " + ", ".join(missing)
            + "\nInstall them in the project's existing environment before running "
              "Parquet census or scoreboard utilities."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
