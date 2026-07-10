from __future__ import annotations

import bz2
import hashlib
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from hkg_tmax.evaluation.reporting import (
    demote_markdown_headings,
    write_bounded_readme_section,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "0005_public_gfs_gefs_himawari_fetch_smoke_20260708"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "campaigns" / "hkg-tmax" / EXPERIMENT_ID
USER_AGENT = "weather-markets-hkg-fetch-smoke/1.0"
README_RESULTS_START = "<!-- BEGIN GENERATED PUBLIC FETCH RESULT -->"
README_RESULTS_END = "<!-- END GENERATED PUBLIC FETCH RESULT -->"

HKG_BBOX = {
    "leftlon": "113.0",
    "rightlon": "115.5",
    "toplat": "23.5",
    "bottomlat": "21.5",
}


@dataclass
class FetchRecord:
    source: str
    status: str
    issued_at_utc: str | None
    valid_at_utc: str | None
    retrieved_at_utc: str
    available_at_method: str
    url: str
    output_path: str | None
    bytes: int
    sha256: str | None
    content_type: str | None
    notes: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def request_bytes(url: str, timeout: int = 90) -> tuple[bytes, dict[str, str]]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as response:
        headers = {k.lower(): v for k, v in response.headers.items()}
        data = response.read()
    return data, headers


def safe_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_path = path
    if sys.platform.startswith("win"):
        write_path = Path("\\\\?\\" + str(path.resolve()))
    write_path.write_bytes(data)


def safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_path = path
    if sys.platform.startswith("win"):
        write_path = Path("\\\\?\\" + str(path.resolve()))
    write_path.write_text(text, encoding="utf-8")


def recent_cycles(max_hours_back: int = 96) -> list[datetime]:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    cycles: list[datetime] = []
    seen: set[tuple[str, int]] = set()
    for hours_back in range(0, max_hours_back + 1, 6):
        candidate = now - timedelta(hours=hours_back)
        cycle_hour = (candidate.hour // 6) * 6
        cycle = candidate.replace(hour=cycle_hour)
        key = (cycle.strftime("%Y%m%d"), cycle.hour)
        if key not in seen:
            cycles.append(cycle)
            seen.add(key)
    return cycles


def build_gfs_url(cycle: datetime, lead_hour: int = 24) -> str:
    cc = f"{cycle.hour:02d}"
    params = {
        "dir": f"/gfs.{cycle:%Y%m%d}/{cc}/atmos",
        "file": f"gfs.t{cc}z.pgrb2.0p25.f{lead_hour:03d}",
        "lev_2_m_above_ground": "on",
        "var_TMP": "on",
        "var_DPT": "on",
        "var_TMAX": "on",
        "var_TMIN": "on",
        "lev_10_m_above_ground": "on",
        "var_UGRD": "on",
        "var_VGRD": "on",
        "lev_mean_sea_level": "on",
        "var_PRMSL": "on",
        **HKG_BBOX,
    }
    return "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?" + urlencode(params)


def build_gfs_index_url(cycle: datetime, lead_hour: int = 24) -> str:
    cc = f"{cycle.hour:02d}"
    return (
        f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"
        f"gfs.{cycle:%Y%m%d}/{cc}/atmos/gfs.t{cc}z.pgrb2.0p25.f{lead_hour:03d}.idx"
    )


def build_gefs_url(cycle: datetime, lead_hour: int = 24) -> str:
    cc = f"{cycle.hour:02d}"
    params = {
        "dir": f"/gefs.{cycle:%Y%m%d}/{cc}/atmos/pgrb2sp25",
        "file": f"gec00.t{cc}z.pgrb2s.0p25.f{lead_hour:03d}",
        "lev_2_m_above_ground": "on",
        "var_TMP": "on",
        "var_DPT": "on",
        "var_TMAX": "on",
        "var_TMIN": "on",
        "lev_10_m_above_ground": "on",
        "var_UGRD": "on",
        "var_VGRD": "on",
        "lev_mean_sea_level": "on",
        "var_PRMSL": "on",
        **HKG_BBOX,
    }
    return "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl?" + urlencode(params)


def build_gefs_index_url(cycle: datetime, lead_hour: int = 24) -> str:
    cc = f"{cycle.hour:02d}"
    return (
        f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/"
        f"gefs.{cycle:%Y%m%d}/{cc}/atmos/pgrb2sp25/gec00.t{cc}z.pgrb2s.0p25.f{lead_hour:03d}.idx"
    )


def fetch_latest_grib(
    source: str,
    url_builder: Any,
    index_url_builder: Any,
    output_stem: str,
    lead_hour: int = 24,
) -> tuple[FetchRecord, dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for cycle in recent_cycles():
        url = url_builder(cycle, lead_hour=lead_hour)
        attempts.append({"cycle_utc": cycle.isoformat(), "url": url})
        try:
            data, headers = request_bytes(url)
        except Exception as exc:
            attempts[-1]["error"] = f"{type(exc).__name__}: {exc}"
            continue

        if not data.startswith(b"GRIB"):
            attempts[-1]["error"] = "response did not start with GRIB magic bytes"
            attempts[-1]["first_200_bytes"] = data[:200].decode("utf-8", errors="replace")
            continue

        out_path = (
            EXPERIMENT_DIR
            / "raw"
            / source
            / f"{output_stem}_{cycle:%Y%m%d}_{cycle:%H}z_f{lead_hour:03d}_hkg_bbox.grib2"
        )
        safe_write_bytes(out_path, data)

        idx_url = index_url_builder(cycle, lead_hour=lead_hour)
        idx_record: dict[str, Any] = {"url": idx_url, "status": "not_fetched"}
        try:
            idx_data, idx_headers = request_bytes(idx_url, timeout=30)
            idx_path = (
                EXPERIMENT_DIR
                / "raw"
                / source
                / f"{output_stem}_{cycle:%Y%m%d}_{cycle:%H}z_f{lead_hour:03d}.idx"
            )
            safe_write_bytes(idx_path, idx_data)
            idx_record = {
                "url": idx_url,
                "status": "fetched",
                "path": str(idx_path.relative_to(EXPERIMENT_DIR)),
                "bytes": len(idx_data),
                "sha256": sha256_bytes(idx_data),
                "content_type": idx_headers.get("content-type"),
            }
        except Exception as exc:
            idx_record = {
                "url": idx_url,
                "status": "fetch_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

        retrieved_at = utc_now_iso()
        record = FetchRecord(
            source=source,
            status="fetched",
            issued_at_utc=cycle.isoformat().replace("+00:00", "Z"),
            valid_at_utc=(cycle + timedelta(hours=lead_hour)).isoformat().replace("+00:00", "Z"),
            retrieved_at_utc=retrieved_at,
            available_at_method="issued_at_utc plus provider/publication buffer; use 6h for strict H24N backtests unless stronger release proof is documented",
            url=url,
            output_path=str(out_path.relative_to(EXPERIMENT_DIR)),
            bytes=len(data),
            sha256=sha256_bytes(data),
            content_type=headers.get("content-type"),
            notes=f"Latest accessible {source.upper()} filtered NOMADS GRIB2 subset for Hong Kong bounding box, lead f{lead_hour:03d}.",
        )
        return record, {"attempts": attempts, "index": idx_record}

    raise RuntimeError(f"No fetchable {source} cycle found in recent cycle window")


def s3_list(
    bucket: str, prefix: str, max_keys: int = 1000, delimiter: str | None = None
) -> dict[str, Any]:
    params = {"list-type": "2", "prefix": prefix, "max-keys": str(max_keys)}
    if delimiter:
        params["delimiter"] = delimiter
    url = f"https://{bucket}.s3.amazonaws.com/?" + urlencode(params)
    data, _headers = request_bytes(url, timeout=60)
    root = ET.fromstring(data)
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys = []
    for node in root.findall("s:Contents", ns):
        keys.append(
            {
                "key": node.findtext("s:Key", default="", namespaces=ns),
                "last_modified": node.findtext("s:LastModified", default="", namespaces=ns),
                "etag": node.findtext("s:ETag", default="", namespaces=ns),
                "size": int(node.findtext("s:Size", default="0", namespaces=ns)),
                "storage_class": node.findtext("s:StorageClass", default="", namespaces=ns),
            }
        )
    common_prefixes = [
        node.findtext("s:Prefix", default="", namespaces=ns)
        for node in root.findall("s:CommonPrefixes", ns)
    ]
    return {"url": url, "keys": keys, "common_prefixes": common_prefixes}


def parse_himawari_observed_at_from_key(key: str) -> str | None:
    match = re.search(r"HS_H09_(\d{8})_(\d{4})_", key)
    if not match:
        match = re.search(r"HS_H08_(\d{8})_(\d{4})_", key)
    if not match:
        return None
    dt = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M").replace(
        tzinfo=timezone.utc
    )
    return dt.isoformat().replace("+00:00", "Z")


def fetch_latest_himawari() -> tuple[FetchRecord, dict[str, Any]]:
    bucket = "noaa-himawari9"
    now = datetime.now(timezone.utc)
    attempts: list[dict[str, Any]] = []

    for hours_back in range(0, 96):
        probe_time = now - timedelta(hours=hours_back)
        hour_prefix = f"AHI-L1b-FLDK/{probe_time:%Y/%m/%d/%H}"
        listing = s3_list(bucket, hour_prefix, max_keys=1000, delimiter="/")
        attempts.append(
            {
                "bucket": bucket,
                "hour_prefix": hour_prefix,
                "scan_prefix_count": len(listing["common_prefixes"]),
                "url": listing["url"],
            }
        )
        if not listing["common_prefixes"]:
            continue

        scan_prefix = sorted(listing["common_prefixes"])[-1]
        scan_listing = s3_list(bucket, scan_prefix, max_keys=1000)
        safe_write_text(
            EXPERIMENT_DIR / "metadata" / "himawari_latest_scan_listing.json",
            json.dumps(scan_listing, indent=2, sort_keys=True),
        )

        preferred = [
            item
            for item in scan_listing["keys"]
            if "_B13_" in item["key"] and item["key"].endswith(".DAT.bz2")
        ]
        if not preferred:
            preferred = [item for item in scan_listing["keys"] if item["key"].endswith(".DAT.bz2")]
        if not preferred:
            attempts[-1]["scan_prefix"] = scan_prefix
            attempts[-1]["error"] = "scan prefix had no DAT.bz2 objects"
            continue

        selected = sorted(preferred, key=lambda item: item["key"])[0]
        url = f"https://{bucket}.s3.amazonaws.com/{selected['key']}"
        data, headers = request_bytes(url, timeout=120)
        out_path = EXPERIMENT_DIR / "raw" / "himawari" / Path(selected["key"]).name
        safe_write_bytes(out_path, data)

        decompressed_header_path: str | None = None
        try:
            decompressed = bz2.decompress(data[: min(len(data), 2_000_000)])
            header_path = (
                EXPERIMENT_DIR
                / "raw"
                / "himawari"
                / (Path(selected["key"]).name + ".first_decompressed_bytes.bin")
            )
            safe_write_bytes(header_path, decompressed[:4096])
            decompressed_header_path = str(header_path.relative_to(EXPERIMENT_DIR))
        except Exception:
            decompressed_header_path = None

        observed_at = parse_himawari_observed_at_from_key(selected["key"])
        record = FetchRecord(
            source="himawari9",
            status="fetched",
            issued_at_utc=observed_at,
            valid_at_utc=observed_at,
            retrieved_at_utc=utc_now_iso(),
            available_at_method="satellite observed_at from object key; S3 LastModified is used as observed public availability proxy for this smoke",
            url=url,
            output_path=str(out_path.relative_to(EXPERIMENT_DIR)),
            bytes=len(data),
            sha256=sha256_bytes(data),
            content_type=headers.get("content-type"),
            notes=(
                "Latest visible Himawari-9 full-disk AHI object from public AWS S3. "
                f"S3 LastModified={selected['last_modified']}; selected_key={selected['key']}; "
                f"first_decompressed_bytes={decompressed_header_path}."
            ),
        )
        return record, {
            "attempts": attempts,
            "selected_object": selected,
            "scan_listing_path": "metadata/himawari_latest_scan_listing.json",
        }

    raise RuntimeError("No fetchable Himawari-9 AHI object found in recent window")


def write_experiment_docs(records: list[FetchRecord], details: dict[str, Any]) -> None:
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": utc_now_iso(),
        "purpose": "Public-source latest issued data fetch smoke for GFS, GEFS, and Himawari-9 without GribStream.",
        "records": [asdict(record) for record in records],
        "details": details,
    }
    safe_write_text(
        EXPERIMENT_DIR / "artifacts" / "fetch_summary.json",
        json.dumps(manifest, indent=2, sort_keys=True),
    )

    rows = []
    for record in records:
        rows.append(
            "| {source} | {status} | {issued} | {valid} | {bytes} | `{path}` |".format(
                source=record.source,
                status=record.status,
                issued=record.issued_at_utc or "",
                valid=record.valid_at_utc or "",
                bytes=record.bytes,
                path=record.output_path or "",
            )
        )

    readme = f"""# Public GFS/GEFS/Himawari Fetch Smoke

Generated: `{manifest["generated_at_utc"]}`

This folder proves direct public-provider fetchability for the latest accessible GFS, GEFS, and Himawari-9 payloads without using GribStream.

| Source | Status | issuedAt / observedAt UTC | validAt UTC | Bytes | Saved payload |
|---|---|---:|---:|---:|---|
{chr(10).join(rows)}

Raw provider payloads live under `raw/`. Machine-readable metadata lives in `artifacts/fetch_summary.json`.
"""
    asof = """## As-of contract

This smoke does not score a model. It verifies provider access and timestamp fields.

For GFS/GEFS:

```text
issuedAt = model cycle / run initialization time
validAt = issuedAt + forecast lead
availableAt for strict H24N backtests = issuedAt + configured publication buffer
default buffer = 6 hours unless a provider-specific release audit proves a tighter value
eligible if availableAt <= target_date T-1 15:00 HKT
```

For Himawari:

```text
observedAt = image timestamp parsed from object key
availableAt proxy in this smoke = S3 LastModified
eligible if observedAt + latency buffer <= target_date T-1 15:00 HKT
```
"""
    results = f"""## Results

All three requested public sources returned provider-native payloads.

| Source | Result |
|---|---|
| GFS | Latest accessible filtered GRIB2 subset saved. |
| GEFS | Latest accessible control-member filtered GRIB2 subset saved. |
| Himawari-9 | Latest visible full-disk AHI `.DAT.bz2` object saved. |

See `artifacts/fetch_summary.json` for URLs, hashes, byte counts, issued/observed timestamps, and request details.
"""
    evidence = """## Evidence map

- `artifacts/fetch_summary.json`: URLs, hashes, byte counts, timestamps, and request details.
- `STATUS.yaml`: machine-readable completion gate.
- `normalized/`: compact normalized CSV/JSON evidence when normalization has been run.
"""
    write_bounded_readme_section(
        EXPERIMENT_DIR / "README.md",
        start_marker=README_RESULTS_START,
        end_marker=README_RESULTS_END,
        section=demote_markdown_headings(
            "\n\n".join(section.strip() for section in (readme, asof, results, evidence))
        ),
        default_title="0005 Public GFS, GEFS, and Himawari Fetch Smoke",
    )

    status = """state: COMPLETE
gate_result: FETCH_SMOKE_PASS
uses_gribstream: false
uses_confirmation_rows: false
"""
    safe_write_text(EXPERIMENT_DIR / "STATUS.yaml", status)


def main() -> int:
    (EXPERIMENT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_DIR / "metadata").mkdir(parents=True, exist_ok=True)
    records: list[FetchRecord] = []
    details: dict[str, Any] = {}

    start = utc_now_iso()
    safe_write_text(
        EXPERIMENT_DIR / "logs" / "run_log.jsonl",
        json.dumps({"event": "start", "at_utc": start}) + "\n",
    )

    for source, url_builder, index_builder, stem in [
        ("gfs", build_gfs_url, build_gfs_index_url, "gfs"),
        ("gefs", build_gefs_url, build_gefs_index_url, "gefs_control"),
    ]:
        record, extra = fetch_latest_grib(source, url_builder, index_builder, stem)
        records.append(record)
        details[source] = extra
        with (EXPERIMENT_DIR / "logs" / "run_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"event": f"{source}_fetched", "record": asdict(record)}) + "\n"
            )
        time.sleep(1)

    record, extra = fetch_latest_himawari()
    records.append(record)
    details["himawari9"] = extra
    with (EXPERIMENT_DIR / "logs" / "run_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "himawari9_fetched", "record": asdict(record)}) + "\n")

    write_experiment_docs(records, details)
    with (EXPERIMENT_DIR / "logs" / "run_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "complete", "at_utc": utc_now_iso()}) + "\n")
    print(EXPERIMENT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
