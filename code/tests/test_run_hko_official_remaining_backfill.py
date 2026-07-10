from __future__ import annotations

import sqlite3
import sys
from datetime import date
from pathlib import Path

from scripts.run_hko_official_remaining_backfill import Config, detail_command, summarize_remaining


def make_config(tmp_path: Path, db_path: Path) -> Config:
    return Config(
        python=sys.executable,
        data_root=tmp_path,
        archive_db=db_path,
        start=date(2013, 11, 15),
        end=date(2013, 11, 16),
        product_types=("local", "5day", "7day", "9day"),
        batch_size=100,
        max_batches=0,
        max_stalled_batches=2,
        delay_seconds=0.35,
        timeout_seconds=60.0,
        max_retries=5,
        progress_interval_seconds=30.0,
        probe_url=None,
        probe_timeout_seconds=15.0,
        skip_network_check=True,
        check_only=False,
        finalize=False,
        monitor_each_batch=False,
        output_dir=tmp_path / "datasets",
        monitor_output_dir=tmp_path / "monitor",
        bundle_stem="test_bundle",
        force_lock=False,
    )


def create_archive_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE candidates (
                source TEXT NOT NULL,
                product_type TEXT NOT NULL,
                index_date TEXT NOT NULL,
                title TEXT,
                url TEXT NOT NULL
            )
            """,
        )
        conn.execute(
            """
            CREATE TABLE retrievals (
                source TEXT NOT NULL,
                url TEXT NOT NULL,
                attempted_at_utc TEXT,
                status_code INTEGER,
                raw_path TEXT,
                error TEXT
            )
            """,
        )


def insert_candidate(
    conn: sqlite3.Connection,
    *,
    product_type: str,
    index_date: str,
    url: str,
) -> None:
    conn.execute(
        """
        INSERT INTO candidates (source, product_type, index_date, title, url)
        VALUES ('info_gov', ?, ?, ?, ?)
        """,
        (product_type, index_date, f"{product_type} {index_date}", url),
    )


def test_summary_counts_only_candidates_without_successful_raw_html(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.sqlite3"
    create_archive_db(db_path)
    raw_path = tmp_path / "raw" / "success.htm"
    raw_path.parent.mkdir()
    raw_path.write_text("ok", encoding="utf-8")

    with sqlite3.connect(db_path) as conn:
        insert_candidate(conn, product_type="local", index_date="2013-11-15", url="https://ok")
        insert_candidate(conn, product_type="local", index_date="2013-11-15", url="https://missing")
        insert_candidate(conn, product_type="7day", index_date="2013-11-16", url="https://not-found")
        conn.execute(
            """
            INSERT INTO retrievals (source, url, attempted_at_utc, status_code, raw_path, error)
            VALUES ('info_gov_bulletin', 'https://ok', '2026-06-22T00:00:00Z', 200, ?, NULL)
            """,
            (str(raw_path),),
        )
        conn.execute(
            """
            INSERT INTO retrievals (source, url, attempted_at_utc, status_code, raw_path, error)
            VALUES ('info_gov_bulletin', 'https://not-found', '2026-06-22T00:00:00Z', 404, NULL, NULL)
            """,
        )

    summary = summarize_remaining(make_config(tmp_path, db_path))

    assert summary["candidate_urls"] == 3
    assert summary["successful_raw_urls"] == 1
    assert summary["remaining_urls"] == 2
    assert {row["url"] for row in summary["first_remaining"]} == {
        "https://missing",
        "https://not-found",
    }
    assert summary["remaining_by_year"] == [{"index_year": "2013", "remaining_urls": 2}]
    assert {row["status_code"]: row["unique_urls"] for row in summary["retrieval_status_counts"]} == {
        200: 1,
        404: 1,
    }


def test_summary_treats_success_with_missing_raw_file_as_remaining(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.sqlite3"
    create_archive_db(db_path)

    with sqlite3.connect(db_path) as conn:
        insert_candidate(conn, product_type="local", index_date="2013-11-15", url="https://lost")
        conn.execute(
            """
            INSERT INTO retrievals (source, url, attempted_at_utc, status_code, raw_path, error)
            VALUES ('info_gov_bulletin', 'https://lost', '2026-06-22T00:00:00Z', 200, ?, NULL)
            """,
            (str(tmp_path / "raw" / "does-not-exist.htm"),),
        )

    summary = summarize_remaining(make_config(tmp_path, db_path))

    assert summary["candidate_urls"] == 1
    assert summary["successful_raw_urls"] == 0
    assert summary["remaining_urls"] == 1
    assert summary["first_remaining"][0]["url"] == "https://lost"


def test_detail_command_uses_missing_success_only_and_batch_limit(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.sqlite3"
    config = make_config(tmp_path, db_path)

    command = detail_command(config)

    assert "--missing-success-only" in command
    assert "-u" in command
    assert command[command.index("--limit") + 1] == "100"
    assert command[command.index("--data-root") + 1] == str(tmp_path)
    assert command[command.index("--types") + 1] == "local,5day,7day,9day"
