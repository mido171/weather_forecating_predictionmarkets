from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_manifest(path: str | Path, payload: Dict[str, Any]) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(out)
