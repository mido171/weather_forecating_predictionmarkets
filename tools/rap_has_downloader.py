#!/usr/bin/env python3
"""
Download a NOAA AIRS/HAS RAP130 order into a local cache.

Supports:
  - status check (best-effort) using the HAS Check Order Status endpoint
  - FTP or HTTP download of fileList.txt and all listed files
  - resume/idempotent behavior
  - manifest.json with file metadata and optional SHA256
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import ftplib
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


HAS_STATUS_URL = "https://www.ncei.noaa.gov/has/HAS.CheckOrderStatus"
DEFAULT_FTP_HOST = "ftp.ncdc.noaa.gov"
DEFAULT_HTTP_BASE = "https://www.ncei.noaa.gov/pub/has"


def _utc_now() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_has_status(email: str, has_id: str) -> str:
    """
    Best-effort status check. Returns: complete / processing / not_found / unknown
    """
    try:
        r = requests.get(HAS_STATUS_URL, params={"email": email, "request": has_id}, timeout=30)
        text = r.text.lower()
    except Exception:
        return "unknown"

    if "complete" in text:
        return "complete"
    if "processing" in text or "in progress" in text or "pending" in text:
        return "processing"
    if "not found" in text or "no matching" in text or "invalid request" in text:
        return "not_found"
    return "unknown"


def _parse_filelist(lines: List[str]) -> List[Tuple[str, Optional[int]]]:
    """
    Parse fileList.txt.
    Supports lines with either:
      <size> <path>
      <path>
    Returns list of (path, size or None).
    """
    out: List[Tuple[str, Optional[int]]] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) == 1:
            out.append((parts[0], None))
            continue
        # guess: first token is size
        size = None
        if re.match(r"^\d+$", parts[0]):
            size = int(parts[0])
            path = parts[-1]
            out.append((path, size))
        else:
            # fallback: last token is path
            out.append((parts[-1], None))
    return out


def _download_http(url: str, out_path: Path, timeout: int = 60) -> None:
    _ensure_dir(out_path.parent)
    tmp = out_path.with_suffix(out_path.suffix + ".partial")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(out_path)


def _download_ftp(ftp: ftplib.FTP, remote_path: str, out_path: Path) -> None:
    _ensure_dir(out_path.parent)
    tmp = out_path.with_suffix(out_path.suffix + ".partial")
    with tmp.open("wb") as f:
        ftp.retrbinary(f"RETR {remote_path}", f.write)
    tmp.replace(out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--has-id", required=True, help="HAS order ID, e.g., HAS123456789")
    ap.add_argument("--email", required=True, help="Email used for HAS order status check")
    ap.add_argument("--cache-dir", required=True, help="Cache root (e.g., data/rap130/has)")
    ap.add_argument("--transport", choices=["ftp", "http"], default="ftp")
    ap.add_argument("--verify", action="store_true", help="Verify nonzero size + counts vs fileList")
    ap.add_argument("--resume", action="store_true", help="Skip files already present with matching size")
    ap.add_argument("--hash", action="store_true", help="Compute SHA256 for downloaded files")
    ap.add_argument("--skip-status", action="store_true", help="Skip HAS status check")
    args = ap.parse_args()

    has_id = args.has_id.strip()
    cache_root = Path(args.cache_dir).expanduser().resolve()
    order_root = cache_root / has_id
    raw_root = order_root / "raw"
    _ensure_dir(raw_root)

    if not args.skip_status:
        status = _check_has_status(args.email, has_id)
        print(f"HAS status: {status}")
        if status not in {"complete", "unknown"}:
            print("Order not complete; aborting download.")
            return 2

    # Download fileList.txt
    filelist_path = order_root / "fileList.txt"
    if args.transport == "http":
        url = f"{DEFAULT_HTTP_BASE}/{has_id}/fileList.txt"
        print(f"Downloading fileList via HTTP: {url}")
        _download_http(url, filelist_path)
    else:
        print(f"Downloading fileList via FTP: {DEFAULT_FTP_HOST}")
        with ftplib.FTP(DEFAULT_FTP_HOST) as ftp:
            ftp.login()
            ftp.cwd(f"/pub/has/{has_id}")
            _download_ftp(ftp, "fileList.txt", filelist_path)

    lines = filelist_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    entries = _parse_filelist(lines)
    print(f"fileList entries: {len(entries)}")

    manifest: List[Dict[str, str]] = []
    downloaded = 0
    skipped = 0

    if args.transport == "ftp":
        ftp = ftplib.FTP(DEFAULT_FTP_HOST)
        ftp.login()
        ftp.cwd(f"/pub/has/{has_id}")
    else:
        ftp = None

    try:
        for rel_path, exp_size in entries:
            rel_path = rel_path.lstrip("/")
            out_path = raw_root / rel_path

            if args.resume and out_path.exists():
                if exp_size is None or out_path.stat().st_size == exp_size:
                    skipped += 1
                    continue

            if args.transport == "http":
                url = f"{DEFAULT_HTTP_BASE}/{has_id}/{rel_path}"
                _download_http(url, out_path)
            else:
                assert ftp is not None
                _download_ftp(ftp, rel_path, out_path)

            downloaded += 1
            size = out_path.stat().st_size
            sha = _sha256(out_path) if args.hash else ""

            manifest.append(
                {
                    "path": str(out_path),
                    "rel_path": rel_path,
                    "size_bytes": str(size),
                    "downloaded_at_utc": _utc_now(),
                    "transport": args.transport,
                    "sha256": sha,
                }
            )
    finally:
        if ftp is not None:
            ftp.quit()

    manifest_path = order_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Downloaded: {downloaded}, skipped: {skipped}")
    print(f"Manifest: {manifest_path}")

    if args.verify:
        missing = 0
        for rel_path, exp_size in entries:
            p = raw_root / rel_path
            if not p.exists():
                missing += 1
                continue
            if exp_size is not None and p.stat().st_size != exp_size:
                missing += 1
        if missing > 0:
            print(f"VERIFY FAILED: missing/mismatch files: {missing}")
            return 3
        print("VERIFY OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
