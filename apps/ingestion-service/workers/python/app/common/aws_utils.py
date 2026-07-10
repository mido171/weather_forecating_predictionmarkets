from __future__ import annotations

from typing import Any, Dict


def not_implemented_payload(service: str, request: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "NOT_IMPLEMENTED",
        "service": service,
        "request": request,
        "note": "Worker scaffold created; source-specific extraction still pending.",
    }
