from __future__ import annotations

from pathlib import Path


_BASE = Path(__file__).resolve().parent
_PYTHON_ROOT = _BASE / "python"

if _PYTHON_ROOT.exists():
    __path__.append(str(_PYTHON_ROOT))
