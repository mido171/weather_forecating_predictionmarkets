from __future__ import annotations

from typing import Dict, List


def parse_lines(text: str) -> Dict[str, object]:
    lines: List[str] = [line.rstrip() for line in text.splitlines() if line.strip()]
    return {
        "line_count": len(lines),
        "head": lines[:10],
    }


def run(request: dict) -> dict:
    text = str(request.get("text", ""))
    return {
        "status": "NOT_IMPLEMENTED",
        "service": "nbm_text",
        "parsed_preview": parse_lines(text),
        "request": request,
    }
