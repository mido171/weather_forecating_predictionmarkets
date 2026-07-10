from __future__ import annotations

from app.extract.extract_ndfd import run as run_direct_grib_extract


def run(request: dict) -> dict:
    request = dict(request)
    request.setdefault("model_name", "ruc")
    return run_direct_grib_extract(request)
