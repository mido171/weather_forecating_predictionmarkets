from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_hko_press_archive_offline import DEFAULT_ARCHIVE_DB  # noqa: E402
from scripts.run_hkg_t24_beastmode_signal_discovery import (  # noqa: E402
    RESEARCH_ROOT,
    markdown_table,
    write_csv,
    write_json,
    write_text,
)

FOLDER_NAME = "0045_missing_press_raw_detail_backfill_or_blocker"
DEFAULT_START_YEAR = 2007
DEFAULT_END_YEAR = 2026
DEFAULT_DATA_ROOT = Path(r"C:\hko_press_2000_2026")


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_candidate_status(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Missing HKO press archive database: {db_path}")
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        candidates = pd.read_sql(
            """
            SELECT source, index_date, title, product_type, url, discovered_at_utc
            FROM candidates
            WHERE source='info_gov'
            """,
            connection,
        )
        retrievals = pd.read_sql(
            """
            SELECT
                url,
                COUNT(*) AS retrieval_attempts,
                SUM(CASE WHEN status_code BETWEEN 200 AND 299 AND raw_path IS NOT NULL THEN 1 ELSE 0 END)
                    AS success_attempts,
                MAX(attempted_at_utc) AS last_attempted_at_utc,
                MAX(CASE WHEN status_code BETWEEN 200 AND 299 AND raw_path IS NOT NULL THEN attempted_at_utc END)
                    AS last_success_at_utc,
                MAX(status_code) AS max_status_code,
                MAX(error) AS sample_error
            FROM retrievals
            WHERE source='info_gov_bulletin'
            GROUP BY url
            """,
            connection,
        )
    out = candidates.merge(retrievals, on="url", how="left")
    out["index_date"] = pd.to_datetime(out["index_date"], errors="coerce").dt.normalize()
    out["index_year"] = out["index_date"].dt.year
    for column in ("retrieval_attempts", "success_attempts"):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)
    out["has_successful_raw_detail"] = out["success_attempts"] > 0
    out["has_any_attempt"] = out["retrieval_attempts"] > 0
    return out.sort_values(["index_date", "url"]).reset_index(drop=True)


def coverage_by_year_product(candidate_status: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        candidate_status.groupby(["index_year", "product_type"], observed=True)
        .agg(
            candidate_urls=("url", "nunique"),
            successful_raw_urls=("has_successful_raw_detail", "sum"),
            attempted_urls=("has_any_attempt", "sum"),
        )
        .reset_index()
    )
    grouped["missing_success_urls"] = grouped["candidate_urls"] - grouped["successful_raw_urls"]
    grouped["never_attempted_urls"] = grouped["candidate_urls"] - grouped["attempted_urls"]
    grouped["raw_success_coverage_pct"] = np.where(
        grouped["candidate_urls"] > 0,
        grouped["successful_raw_urls"] / grouped["candidate_urls"],
        math.nan,
    )
    grouped["status"] = np.where(
        grouped["missing_success_urls"].eq(0),
        "downloaded",
        np.where(grouped["successful_raw_urls"].eq(0), "no_successful_raw_detail", "partial"),
    )
    return grouped.sort_values(["index_year", "product_type"]).reset_index(drop=True)


def missing_candidates(
    candidate_status: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    product_types: set[str],
) -> pd.DataFrame:
    mask = candidate_status["index_year"].between(start_year, end_year, inclusive="both")
    mask &= ~candidate_status["has_successful_raw_detail"]
    if product_types:
        mask &= candidate_status["product_type"].isin(product_types)
    columns = [
        "index_date",
        "index_year",
        "product_type",
        "title",
        "url",
        "retrieval_attempts",
        "last_attempted_at_utc",
        "max_status_code",
        "sample_error",
    ]
    out = candidate_status.loc[mask, columns].copy()
    out["index_date"] = out["index_date"].dt.date.astype(str)
    return out.sort_values(["index_date", "url"]).reset_index(drop=True)


def priority_sample(missing: pd.DataFrame, *, per_year: int) -> pd.DataFrame:
    if missing.empty:
        return missing.copy()
    return (
        missing.groupby("index_year", observed=True, group_keys=False)
        .head(per_year)
        .reset_index(drop=True)
    )


def probe_missing_urls(
    missing: pd.DataFrame,
    *,
    probe_count: int,
    artifacts: Path,
    timeout_seconds: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if probe_count <= 0 or missing.empty:
        return pd.DataFrame(
            columns=[
                "url",
                "index_date",
                "product_type",
                "ok",
                "status_code",
                "error_type",
                "error",
                "content_length",
                "content_sha256",
                "raw_probe_path",
            ]
        )
    probe_root = artifacts / "probe_raw"
    for row in missing.head(probe_count).itertuples(index=False):
        base: dict[str, Any] = {
            "url": str(row.url),
            "index_date": str(row.index_date),
            "product_type": str(row.product_type),
            "ok": False,
            "status_code": None,
            "error_type": "",
            "error": "",
            "content_length": 0,
            "content_sha256": "",
            "raw_probe_path": "",
        }
        try:
            with httpx.Client(
                follow_redirects=True,
                timeout=timeout_seconds,
                headers={"User-Agent": "HKG-Tmax-Research-Probe/1.0"},
            ) as client:
                response = client.get(str(row.url))
            digest = sha256_bytes(response.content)
            base.update(
                {
                    "ok": 200 <= response.status_code < 300,
                    "status_code": int(response.status_code),
                    "content_length": len(response.content),
                    "content_sha256": digest,
                }
            )
            if 200 <= response.status_code < 300:
                year = str(row.index_year)
                probe_dir = probe_root / year
                probe_dir.mkdir(parents=True, exist_ok=True)
                raw_path = probe_dir / f"{digest}.html"
                if not raw_path.exists():
                    raw_path.write_bytes(response.content)
                sidecar = raw_path.with_suffix(raw_path.suffix + ".metadata.json")
                if not sidecar.exists():
                    sidecar.write_text(
                        json.dumps(
                            {
                                "url": str(row.url),
                                "index_date": str(row.index_date),
                                "product_type": str(row.product_type),
                                "status_code": int(response.status_code),
                                "headers": dict(response.headers),
                                "content_sha256": digest,
                                "content_length": len(response.content),
                                "retrieved_at_utc": now_utc(),
                                "note": "Probe only; not inserted into canonical C:\\hko_press_2000_2026 archive.",
                            },
                            indent=2,
                            sort_keys=True,
                        ),
                        encoding="utf-8",
                    )
                base["raw_probe_path"] = str(raw_path)
        except Exception as exc:  # noqa: BLE001 - probe failures are blocker evidence.
            base["error_type"] = type(exc).__name__
            base["error"] = str(exc)
        rows.append(base)
    return pd.DataFrame(rows)


def probe_blocker_status(probe_results: pd.DataFrame) -> str:
    if probe_results.empty:
        return "not_probed"
    if probe_results["ok"].any():
        return "probe_success_not_canonical_archive"
    errors = " ".join(probe_results["error"].fillna("").astype(str).to_list()).lower()
    if "forbidden" in errors or "10013" in errors or "permission" in errors:
        return "network_socket_blocked"
    return "probe_failed"


def command_table(*, start_year: int, end_year: int, data_root: Path) -> pd.DataFrame:
    archive_script = REPO_ROOT / "scripts" / "hko_forecast_archive_downloader_rss" / "hko_archive.py"
    rows = [
        {
            "purpose": "finish_2007_partial",
            "command": (
                f".\\.venv\\Scripts\\python.exe {archive_script} official-details "
                f"--data-root {data_root} --start 2007-01-01 --end 2007-12-31 "
                "--types local,7day --missing-success-only --delay-seconds 1.25 --timeout-seconds 45 --max-retries 5"
            ),
        },
        {
            "purpose": "start_2008_bounded_smoke",
            "command": (
                f".\\.venv\\Scripts\\python.exe {archive_script} official-details "
                f"--data-root {data_root} --start 2008-01-01 --end 2008-01-07 "
                "--types local,7day --missing-success-only --limit 100 --delay-seconds 1.25 --timeout-seconds 45 --max-retries 5"
            ),
        },
        {
            "purpose": "full_missing_period_after_smoke",
            "command": (
                f".\\.venv\\Scripts\\python.exe {archive_script} official-details "
                f"--data-root {data_root} --start {start_year}-01-01 --end {end_year}-12-31 "
                "--types local,7day,9day --missing-success-only --delay-seconds 1.25 --timeout-seconds 45 --max-retries 5"
            ),
        },
        {
            "purpose": "refresh_scored_export_after_backfill",
            "command": (
                ".\\.venv\\Scripts\\python.exe "
                f"{REPO_ROOT / 'scripts' / 'run_hkg_t24_forecast_archive_continuous_scored_export.py'}"
            ),
        },
    ]
    return pd.DataFrame(rows)


def write_outputs(
    *,
    candidate_status: pd.DataFrame,
    coverage: pd.DataFrame,
    missing: pd.DataFrame,
    sample: pd.DataFrame,
    probe_results: pd.DataFrame,
    commands: pd.DataFrame,
    start_year: int,
    end_year: int,
    generated_at_utc: str,
) -> dict[str, Any]:
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    write_csv(artifacts / "coverage_by_year_product.csv", coverage)
    write_csv(artifacts / "missing_candidates.csv", missing)
    write_csv(artifacts / "priority_missing_sample.csv", sample)
    write_csv(artifacts / "probe_results.csv", probe_results)
    write_csv(artifacts / "backfill_commands.csv", commands)

    target_coverage = coverage[coverage["index_year"].between(start_year, end_year, inclusive="both")]
    zero_success_years = sorted(
        int(year)
        for year in target_coverage.loc[target_coverage["successful_raw_urls"].eq(0), "index_year"]
        .dropna()
        .unique()
    )
    partial_years = sorted(
        int(year)
        for year in target_coverage.loc[target_coverage["status"].eq("partial"), "index_year"]
        .dropna()
        .unique()
    )
    blocker = probe_blocker_status(probe_results)
    manifest: dict[str, Any] = {
        "generated_at_utc": generated_at_utc,
        "folder": FOLDER_NAME,
        "candidate_rows": int(len(candidate_status)),
        "start_year": int(start_year),
        "end_year": int(end_year),
        "missing_candidate_urls": int(len(missing)),
        "zero_success_years": zero_success_years,
        "partial_years": partial_years,
        "probe_count": int(len(probe_results)),
        "probe_blocker_status": blocker,
        "probe_errors": probe_results[["url", "error_type", "error"]].to_dict("records")
        if not probe_results.empty
        else [],
        "next_required_action": "run_backfill_commands_outside_socket_blocked_sandbox_then_rerun_0044",
    }
    write_json(artifacts / "summary.json", manifest)
    write_json(RESEARCH_ROOT / "missing_press_raw_detail_backfill_or_blocker_manifest.json", manifest)

    readme = f"""# Missing Press Raw-Detail Backfill Or Blocker

Generated: `{generated_at_utc}`

## Purpose

`0044` promoted all currently available HKO press raw detail into the scored forecast export, but the official forecast history is still non-contiguous. This folder turns the remaining problem into an executable acquisition plan: exactly which candidate URLs still lack successful raw detail, what bounded commands should fetch them, and what happened when this sandbox attempted a provider probe.

## Current Result

| Metric | Value |
|---|---:|
| Candidate rows audited | {manifest['candidate_rows']} |
| Missing candidate URLs in scope | {manifest['missing_candidate_urls']} |
| Scope years | {start_year} to {end_year} |
| Partial years | {partial_years} |
| Zero-success years | {zero_success_years} |
| Probe status | {blocker} |

## Probe Evidence

{markdown_table(probe_results, max_rows=10)}

## Coverage By Year And Product

{markdown_table(target_coverage, max_rows=80)}

## Priority Missing URL Sample

{markdown_table(sample, max_rows=60)}

## Backfill Commands

{markdown_table(commands, max_rows=10)}

## Interpretation

The 2008+ gap is not a parser limitation. The candidate URLs are known, but the canonical raw-detail archive does not yet contain successful HTML snapshots for those URLs. A bounded HTTP probe from this Codex sandbox failed with the status shown above, so this turn cannot lawfully complete the download from inside the current environment.

The next acquisition run should execute the bounded smoke command first. After successful raw snapshots are written into `C:\\hko_press_2000_2026`, rerun `0044` to refresh the scored export. Only after the continuous scored forecast frame exists should the official-anchor correction and router experiments be rerun.
"""
    write_text(folder / "README.md", readme)
    update_master_index(manifest)
    update_milestones(manifest)
    return manifest


def update_master_index(manifest: dict[str, Any]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Missing Press Raw-Detail Backfill Or Blocker\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_missing_press_raw_detail_backfill_or_blocker.py`:

- `{FOLDER_NAME}`: missing candidate URL manifest, bounded backfill commands, and current sandbox/provider probe evidence.

| Metric | Value |
|---|---:|
| Missing candidate URLs | {manifest['missing_candidate_urls']} |
| Partial years | {manifest['partial_years']} |
| Zero-success years | {manifest['zero_success_years']} |
| Probe blocker status | {manifest['probe_blocker_status']} |

This does not start modelling. It is the acquisition gate needed before rerunning the official-forecast anchor stack.
"""
    write_text(index_path, text)


def update_milestones(manifest: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# HKG Tmax Milestones\n"
    section_marker = "\n## Missing Press Raw-Detail Backfill Or Blocker\n"
    blockers_marker = "\n## Current Blockers And Gaps\n"
    next_marker = "\n## Exact Next Recommended Codex Task\n"
    if section_marker in existing:
        base, rest = existing.split(section_marker, 1)
        suffix = f"{blockers_marker}{rest.split(blockers_marker, 1)[1]}" if blockers_marker in rest else ""
    elif blockers_marker in existing:
        base, rest = existing.split(blockers_marker, 1)
        suffix = f"{blockers_marker}{rest}"
    else:
        base = existing.rstrip()
        suffix = ""
    section = f"""{base.rstrip()}
{section_marker}
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_missing_press_raw_detail_backfill_or_blocker.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Missing raw-detail URLs | `{manifest['missing_candidate_urls']}` candidate URLs in years `{manifest['start_year']}`-`{manifest['end_year']}` still lack successful raw detail | Acquisition not complete |
| Partial years | `{manifest['partial_years']}` | Need bounded catch-up |
| Zero-success years | `{manifest['zero_success_years']}` | Need raw-detail backfill |
| Sandbox probe | `{manifest['probe_blocker_status']}` | Current Codex sandbox cannot fetch provider URLs directly |

Interpretation: `0045` converts the remaining forecast-archive gap into an executable acquisition manifest and records the current socket blocker. No modelling or router rerun should happen until raw detail snapshots are acquired into `C:\\hko_press_2000_2026` and `0044` is rerun successfully.
"""
    if suffix:
        if next_marker in suffix:
            before_next, _after_next = suffix.split(next_marker, 1)
            next_task = f"""{next_marker}

Run the `0045` bounded backfill smoke command outside the current socket-blocked sandbox, starting with 2008-01-01 through 2008-01-07 and `--limit 100`; then rerun `0044` to verify raw detail promotion before launching the full 2008-2026 backfill.
"""
            suffix = before_next.rstrip() + next_task
        section += suffix
    write_text(path, section)


def run(
    *,
    archive_db: Path = DEFAULT_ARCHIVE_DB,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    product_types: set[str] | None = None,
    sample_per_year: int = 5,
    probe_count: int = 1,
    probe_timeout_seconds: float = 8.0,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    generated_at = now_utc()
    product_types = product_types or {"local", "5day", "7day", "9day"}
    status = load_candidate_status(archive_db)
    coverage = coverage_by_year_product(status)
    missing = missing_candidates(
        status,
        start_year=start_year,
        end_year=end_year,
        product_types=product_types,
    )
    sample = priority_sample(missing, per_year=sample_per_year)
    artifacts = RESEARCH_ROOT / FOLDER_NAME / "artifacts"
    probe_results = probe_missing_urls(
        missing,
        probe_count=probe_count,
        artifacts=artifacts,
        timeout_seconds=probe_timeout_seconds,
    )
    commands = command_table(start_year=start_year, end_year=end_year, data_root=data_root)
    return write_outputs(
        candidate_status=status,
        coverage=coverage,
        missing=missing,
        sample=sample,
        probe_results=probe_results,
        commands=commands,
        start_year=start_year,
        end_year=end_year,
        generated_at_utc=generated_at,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HKO press raw-detail backfill plan/blocker report.")
    parser.add_argument("--archive-db", type=Path, default=DEFAULT_ARCHIVE_DB)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--types", default="local,5day,7day,9day")
    parser.add_argument("--sample-per-year", type=int, default=5)
    parser.add_argument("--probe-count", type=int, default=1)
    parser.add_argument("--probe-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    product_types = {part.strip() for part in args.types.split(",") if part.strip()}
    manifest = run(
        archive_db=args.archive_db,
        start_year=args.start_year,
        end_year=args.end_year,
        product_types=product_types,
        sample_per_year=args.sample_per_year,
        probe_count=args.probe_count,
        probe_timeout_seconds=args.probe_timeout_seconds,
        data_root=args.data_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
