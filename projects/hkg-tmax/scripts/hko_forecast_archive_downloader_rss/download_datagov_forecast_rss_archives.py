from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import httpx

HKT = ZoneInfo("Asia/Hong_Kong")
LIST_URL = "https://app.data.gov.hk/v1/historical-archive/list-file-versions"
GET_URL = "https://app.data.gov.hk/v1/historical-archive/get-file"


@dataclass(frozen=True)
class Feed:
    suffix: str
    slug: str
    url: str
    start: str


FORECAST_FEEDS = (
    Feed("rss_local_forecast_en", "local_en", "https://rss.weather.gov.hk/rss/LocalWeatherForecast.xml", "20200601"),
    Feed("rss_local_forecast_tc", "local_tc", "https://rss.weather.gov.hk/rss/LocalWeatherForecast_uc.xml", "20200601"),
    Feed("rss_local_forecast_sc", "local_sc", "https://rss.weather.gov.hk/sc/rss/LocalWeatherForecast_uc.xml", "20200601"),
    Feed("rss_9day_forecast_en", "9day_en", "https://rss.weather.gov.hk/rss/SeveralDaysWeatherForecast_v2.xml", "20210401"),
    Feed("rss_9day_forecast_tc", "9day_tc", "https://rss.weather.gov.hk/rss/SeveralDaysWeatherForecast_v2_uc.xml", "20210401"),
    Feed("rss_9day_forecast_sc", "9day_sc", "https://rss.weather.gov.hk/sc/rss/SeveralDaysWeatherForecast_v2_uc.xml", "20210401"),
)


def default_end() -> str:
    return (datetime.now(UTC).astimezone(HKT) - timedelta(days=1)).strftime("%Y%m%d")


def request_json(client: httpx.Client, url: str) -> dict:
    response = client.get(url)
    response.raise_for_status()
    return response.json()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(content)
    temporary.replace(path)


def fetch_bytes(client: httpx.Client, url: str, retries: int) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.get(url)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(min(30.0, 2.0**attempt))
                continue
            response.raise_for_status()
            return response
        except Exception as exc:  # noqa: BLE001 - record final HTTP failure with context.
            last_error = exc
            if attempt < retries:
                time.sleep(min(30.0, 2.0**attempt))
                continue
            raise
    raise RuntimeError(f"unreachable retry state for {url}: {last_error}")


def listing_url(feed: Feed, end: str) -> str:
    return f"{LIST_URL}?{urlencode({'url': feed.url, 'start': feed.start, 'end': end})}"


def get_file_url(feed: Feed, timestamp: str) -> str:
    return f"{GET_URL}?{urlencode({'url': feed.url, 'time': timestamp})}"


def data_file_items(listing: dict) -> list[dict[str, object]]:
    items = listing.get("data-files")
    if isinstance(items, list) and items:
        return [item for item in items if isinstance(item, dict)]
    timestamps = listing.get("timestamps")
    if isinstance(timestamps, list):
        return [{"timestamp": item, "period": "", "filename": "", "size": ""} for item in timestamps]
    return []


def write_manifest(output_root: Path, rows: list[dict[str, object]]) -> None:
    manifest = output_root / "manifest.csv"
    fields = [
        "feed",
        "kind",
        "timestamp",
        "url",
        "status_code",
        "content_sha256",
        "content_length",
        "path",
        "retrieved_at_utc",
        "data_gov_filename",
        "data_gov_period",
        "data_gov_expected_size",
        "skipped_existing",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def download(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).expanduser().resolve()
    listing_root = output_root / "l"
    archive_root = output_root / "a"
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    headers = {"User-Agent": args.user_agent}

    with httpx.Client(timeout=args.timeout_seconds, follow_redirects=True, headers=headers) as client:
        for feed in FORECAST_FEEDS:
            url = listing_url(feed, args.end)
            listing = request_json(client, url)
            listing_content = json.dumps(listing, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
            listing_path = listing_root / f"{feed.slug}_{feed.start}_{args.end}.json"
            atomic_write(listing_path, listing_content)
            listing_digest = sha256_bytes(listing_content)
            rows.append(
                {
                    "feed": feed.suffix,
                    "kind": "listing",
                    "timestamp": "",
                    "url": url,
                    "status_code": 200,
                    "content_sha256": listing_digest,
                    "content_length": len(listing_content),
                    "path": str(listing_path),
                    "retrieved_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "data_gov_filename": "",
                    "data_gov_period": "",
                    "data_gov_expected_size": "",
                    "skipped_existing": False,
                }
            )

            items = data_file_items(listing)
            print(f"{feed.suffix}: {len(items)} archive files")
            for index, item in enumerate(items, 1):
                timestamp = str(item.get("timestamp", ""))
                if not timestamp:
                    continue
                file_url = get_file_url(feed, timestamp)
                extension = "zip" if str(item.get("filename", "")).lower().endswith(".zip") else "xml"
                archive_path = archive_root / f"{feed.slug}_{timestamp}.{extension}"
                metadata_path = archive_path.with_suffix(archive_path.suffix + ".metadata.json")
                skipped = archive_path.exists() and archive_path.stat().st_size > 0
                if skipped:
                    content = archive_path.read_bytes()
                    status_code = 200
                    final_url = file_url
                    headers_dict: dict[str, str] = {}
                else:
                    response = fetch_bytes(client, file_url, args.retries)
                    content = response.content
                    status_code = response.status_code
                    final_url = str(response.url)
                    headers_dict = dict(response.headers)
                    atomic_write(archive_path, content)
                    atomic_write(
                        metadata_path,
                        (
                            json.dumps(
                                {
                                    "feed": feed.suffix,
                                    "resource_url": feed.url,
                                    "timestamp": timestamp,
                                    "request_url": file_url,
                                    "final_url": final_url,
                                    "status_code": status_code,
                                    "headers": headers_dict,
                                    "content_sha256": sha256_bytes(content),
                                    "content_length": len(content),
                                    "retrieved_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                                    "data_gov_listing_item": item,
                                },
                                ensure_ascii=False,
                                indent=2,
                                sort_keys=True,
                            )
                            + "\n"
                        ).encode("utf-8"),
                    )
                    if args.delay_seconds > 0:
                        time.sleep(args.delay_seconds)

                rows.append(
                    {
                        "feed": feed.suffix,
                        "kind": "archive",
                        "timestamp": timestamp,
                        "url": file_url,
                        "status_code": status_code,
                        "content_sha256": sha256_bytes(content),
                        "content_length": len(content),
                        "path": str(archive_path),
                        "retrieved_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                        "data_gov_filename": str(item.get("filename", "")),
                        "data_gov_period": str(item.get("period", "")),
                        "data_gov_expected_size": str(item.get("size", "")),
                        "skipped_existing": skipped,
                    }
                )
                if index % 25 == 0:
                    write_manifest(output_root, rows)
                    print(f"{feed.suffix}: downloaded/listed {index}/{len(items)}")

    write_manifest(output_root, rows)
    print(f"wrote {len(rows)} manifest rows under {output_root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download DATA.GOV.HK historical HKO forecast RSS archives")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--end", default=default_end())
    parser.add_argument("--delay-seconds", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--user-agent", default="HKG-Tmax-Research/0.1")
    return parser


def main() -> int:
    return download(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
