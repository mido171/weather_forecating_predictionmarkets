from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from hkg_tmax.paths import find_project_root

ARCHIVE_MODULE_DIR = find_project_root(Path(__file__)) / "scripts" / "hko_forecast_archive_downloader_rss"
if str(ARCHIVE_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(ARCHIVE_MODULE_DIR))

from hko_archive import build_parser, candidate_date_in_range  # noqa: E402


def test_candidate_date_in_range_handles_open_and_closed_bounds() -> None:
    assert candidate_date_in_range("2008-01-02", date(2008, 1, 1), date(2008, 1, 31))
    assert not candidate_date_in_range("2007-12-31", date(2008, 1, 1), None)
    assert not candidate_date_in_range("2008-02-01", None, date(2008, 1, 31))
    assert candidate_date_in_range("2008-02-01", None, None)
    assert not candidate_date_in_range(None, date(2008, 1, 1), None)


def test_official_details_parser_accepts_bounded_missing_only_args() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "official-details",
            "--data-root",
            r"C:\hko_press_2000_2026",
            "--start",
            "2008-01-01",
            "--end",
            "2008-01-07",
            "--limit",
            "100",
            "--missing-success-only",
        ]
    )

    assert args.command == "official-details"
    assert args.start == "2008-01-01"
    assert args.end == "2008-01-07"
    assert args.limit == 100
    assert args.missing_success_only is True
