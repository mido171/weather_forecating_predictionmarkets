"""Discover MOS JSON keys for GFS/NAM from IEM mos.py."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover MOS variable keys.")
    parser.add_argument(
        "--app-config",
        default=str(
            REPO_ROOT
            / "ingestion-service"
            / "src"
            / "main"
            / "resources"
            / "application.yml"
        ),
        help="Path to application.yml",
    )
    parser.add_argument("--station-id", help="Station ID (default: first configured station).")
    parser.add_argument(
        "--models",
        default="GFS,NAM",
        help="Comma-separated MOS models (default: GFS,NAM).",
    )
    parser.add_argument(
        "--hours-back",
        type=int,
        default=48,
        help="Hours to look back (default: 48).",
    )
    parser.add_argument(
        "--hours-forward",
        type=int,
        default=6,
        help="Hours to look forward (default: 6).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "artifacts" / "discovery"),
        help="Output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(Path(args.app_config))
    base_url = (config.get("iem") or {}).get("base-url", "https://mesonet.agron.iastate.edu")
    station_id = resolve_station_id(config, args.station_id)
    models = [m.strip().upper() for m in args.models.split(",") if m.strip()]
    if not models:
        raise ValueError("No MOS models provided.")

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=args.hours_back)
    end = now + timedelta(hours=args.hours_forward)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        payload = fetch_mos_payload(base_url, station_id, model, start, end)
        summary = summarize_payload(payload)
        out_path = output_dir / f"iem_mos_keys_{station_id}_{model}.json"
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[MOS] {model} keys={len(summary['keys'])} output={out_path}")
    return 0


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_station_id(config: dict, station_id: str | None) -> str:
    if station_id:
        return station_id.strip().upper()
    stations = (config.get("gribstream") or {}).get("stations") or []
    if stations:
        return stations[0]["stationId"].strip().upper()
    raise ValueError("No station_id provided and gribstream.stations is empty.")


def fetch_mos_payload(
    base_url: str, station_id: str, model: str, start: datetime, end: datetime
) -> list[dict[str, Any]]:
    params = {
        "station": station_id,
        "model": model,
        "sts": start.strftime("%Y-%m-%dT%H:%MZ"),
        "ets": end.strftime("%Y-%m-%dT%H:%MZ"),
        "format": "json",
    }
    url = f"{base_url}/cgi-bin/request/mos.py?{urlencode(params)}"
    with urlopen(url, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("MOS response is not a JSON array.")
    return payload


def summarize_payload(payload: list[dict[str, Any]]) -> dict[str, Any]:
    key_info: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            info = key_info.setdefault(
                key,
                {
                    "count": 0,
                    "non_null": 0,
                    "types": {},
                    "sample_values": [],
                },
            )
            info["count"] += 1
            if value is not None:
                info["non_null"] += 1
                tname = type(value).__name__
                info["types"][tname] = info["types"].get(tname, 0) + 1
                if len(info["sample_values"]) < 5:
                    info["sample_values"].append(value)
    return {
        "rows": len(payload),
        "keys": sorted(key_info.keys()),
        "key_summary": key_info,
    }


if __name__ == "__main__":
    raise SystemExit(main())
