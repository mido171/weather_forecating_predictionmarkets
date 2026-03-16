from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path

from app.common.logging_utils import configure_logging, log_json


WORKER_MODULES = {
    "ndfd": "app.extract.extract_ndfd",
    "gefs_reforecast": "app.extract.extract_gefs_reforecast",
    "gefs_operational": "app.extract.extract_gefs_operational",
    "nbm_grid": "app.extract.extract_nbm_grid",
    "hrrr": "app.extract.extract_hrrr",
    "rap": "app.extract.extract_rap",
    "ruc": "app.extract.extract_ruc",
    "gfs": "app.extract.extract_gfs",
    "nam": "app.extract.extract_nam",
    "nbm_text": "app.source.parse_nbm_text",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pilot heavy-model worker scaffold")
    parser.add_argument("--worker", required=True, choices=sorted(WORKER_MODULES))
    parser.add_argument("--request-json", required=True, help="JSON string or path to JSON request payload")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_request(raw: str) -> dict:
    candidate = Path(raw)
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return json.loads(raw)


def main() -> int:
    args = parse_args()
    logger = configure_logging(args.log_level)
    request = load_request(args.request_json)
    module = import_module(WORKER_MODULES[args.worker])
    result = module.run(request)
    log_json(logger, action="worker_run", worker=args.worker, status=result.get("status"), request=request)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
