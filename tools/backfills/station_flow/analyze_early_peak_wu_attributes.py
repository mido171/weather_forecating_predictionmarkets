from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import sqlite3
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    cur = Path(__file__).resolve()
    for p in cur.parents:
        if (p / "pom.xml").is_file() and (p / "apps" / "ingestion-service").is_dir():
            return p
    raise FileNotFoundError("repo root not found")


ROOT = repo_root()

WU_TABLE = "wu_observations_30m"
TEXT_COLUMNS = {
    "station_id",
    "request_location_id",
    "valid_time_utc",
    "valid_time_local",
    "target_date_local",
    "clds",
    "wx_phrase",
    "uv_desc",
    "wdir_cardinal",
}
TIME_HEAVY_COLUMNS = {"valid_time_utc", "valid_time_local"}
NUMERIC_COLUMNS = {
    "cutoff_minutes_local",
    "temp",
    "dew_pt",
    "rh",
    "pressure",
    "vis",
    "wspd",
    "wdir",
    "gust",
    "precip_hrly",
    "uv_index",
}
INLINE_DISTINCT_LIMIT = 2500


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze WU attributes across EarlyPeak SQLite stores.")
    p.add_argument("--data-root", default=r"D:\Ahmed\data\sqlite\EarlyPeak")
    p.add_argument("--out-root", default="")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def init_logger(level_name: str) -> logging.Logger:
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("wu_attribute_audit")


def discover_station_dbs(data_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for station_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        dbs = sorted(station_dir.glob("*.sqlite"))
        if not dbs:
            continue
        out[station_dir.name.upper()] = dbs[0]
    if not out:
        raise FileNotFoundError(f"No station sqlite files found under {data_root}")
    return out


def table_columns(conn: sqlite3.Connection, table_name: str) -> list[dict[str, Any]]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [
        {
            "cid": int(row[0]),
            "name": str(row[1]),
            "sqlite_type": str(row[2]),
            "notnull": bool(row[3]),
            "default_value": row[4],
            "pk": int(row[5]),
        }
        for row in rows
    ]


def col_kind(column_name: str) -> str:
    if column_name in TIME_HEAVY_COLUMNS:
        return "timestamp"
    if column_name == "target_date_local":
        return "date"
    if column_name in {"station_id", "request_location_id"}:
        return "identifier"
    if column_name in {"clds", "wx_phrase", "uv_desc", "wdir_cardinal"}:
        return "categorical_text"
    if column_name == "cutoff_minutes_local":
        return "derived_discrete_numeric"
    if column_name in NUMERIC_COLUMNS:
        return "numeric"
    return "text"


def sql_non_missing_expr(column_name: str) -> str:
    if column_name in TEXT_COLUMNS:
        return f"{column_name} IS NOT NULL AND TRIM(CAST({column_name} AS TEXT)) <> ''"
    return f"{column_name} IS NOT NULL"


def sql_blank_expr(column_name: str) -> str:
    if column_name in TEXT_COLUMNS:
        return f"{column_name} IS NOT NULL AND TRIM(CAST({column_name} AS TEXT)) = ''"
    return "0"


def normalize_value(column_name: str, value: Any) -> str:
    if value is None:
        return ""
    if column_name in NUMERIC_COLUMNS:
        if isinstance(value, float):
            text = f"{value:.15g}"
        else:
            text = str(value)
        return text
    return str(value)


def sort_value_key(column_name: str, value_text: str) -> tuple[Any, str]:
    if column_name in NUMERIC_COLUMNS:
        try:
            return (0, float(value_text))
        except Exception:
            return (1, value_text)
    return (0, value_text)


def compute_station_overview(conn: sqlite3.Connection, station_id: str) -> dict[str, Any]:
    total_rows = int(conn.execute(f"SELECT COUNT(*) FROM {WU_TABLE}").fetchone()[0] or 0)
    distinct_days = int(conn.execute(f"SELECT COUNT(DISTINCT target_date_local) FROM {WU_TABLE}").fetchone()[0] or 0)
    daily_counts = [
        int(row[1])
        for row in conn.execute(
            f"SELECT target_date_local, COUNT(*) FROM {WU_TABLE} GROUP BY target_date_local ORDER BY target_date_local"
        ).fetchall()
    ]
    avg_rows_per_day = float(sum(daily_counts) / len(daily_counts)) if daily_counts else 0.0
    median_rows_per_day = float(statistics.median(daily_counts)) if daily_counts else 0.0
    min_rows_per_day = int(min(daily_counts)) if daily_counts else 0
    max_rows_per_day = int(max(daily_counts)) if daily_counts else 0
    pct_days_ge_24 = (100.0 * sum(1 for x in daily_counts if x >= 24) / len(daily_counts)) if daily_counts else 0.0
    pct_days_ge_48 = (100.0 * sum(1 for x in daily_counts if x >= 48) / len(daily_counts)) if daily_counts else 0.0
    min_local_date, max_local_date = conn.execute(
        f"SELECT MIN(target_date_local), MAX(target_date_local) FROM {WU_TABLE}"
    ).fetchone()
    min_valid_utc, max_valid_utc = conn.execute(
        f"SELECT MIN(valid_time_utc), MAX(valid_time_utc) FROM {WU_TABLE}"
    ).fetchone()
    return {
        "station_id": station_id,
        "rows_total": total_rows,
        "distinct_target_dates": distinct_days,
        "min_target_date_local": str(min_local_date or ""),
        "max_target_date_local": str(max_local_date or ""),
        "min_valid_time_utc": str(min_valid_utc or ""),
        "max_valid_time_utc": str(max_valid_utc or ""),
        "avg_rows_per_day": avg_rows_per_day,
        "median_rows_per_day": median_rows_per_day,
        "min_rows_per_day": min_rows_per_day,
        "max_rows_per_day": max_rows_per_day,
        "pct_days_ge_24_rows": pct_days_ge_24,
        "pct_days_ge_48_rows": pct_days_ge_48,
    }


def compute_station_attribute_summary(
    conn: sqlite3.Connection,
    station_id: str,
    column_name: str,
    total_rows: int,
    distinct_days: int,
) -> dict[str, Any]:
    non_missing_expr = sql_non_missing_expr(column_name)
    blank_expr = sql_blank_expr(column_name)
    is_numeric = column_name in NUMERIC_COLUMNS
    avg_expr = f"AVG(CASE WHEN {non_missing_expr} THEN {column_name} END)" if is_numeric else "NULL"
    zero_expr = (
        f"SUM(CASE WHEN {non_missing_expr} AND CAST({column_name} AS REAL) = 0 THEN 1 ELSE 0 END)"
        if is_numeric
        else "NULL"
    )
    sql = f"""
        SELECT
            SUM(CASE WHEN {non_missing_expr} THEN 1 ELSE 0 END) AS non_missing_rows,
            SUM(CASE WHEN {column_name} IS NULL THEN 1 ELSE 0 END) AS null_rows,
            SUM(CASE WHEN {blank_expr} THEN 1 ELSE 0 END) AS blank_rows,
            COUNT(DISTINCT CASE WHEN {non_missing_expr} THEN {column_name} END) AS distinct_non_missing_values,
            COUNT(DISTINCT CASE WHEN {non_missing_expr} THEN target_date_local END) AS days_with_any_value,
            MIN(CASE WHEN {non_missing_expr} THEN {column_name} END) AS min_value,
            MAX(CASE WHEN {non_missing_expr} THEN {column_name} END) AS max_value,
            {avg_expr} AS avg_value,
            {zero_expr} AS zero_count
        FROM {WU_TABLE}
    """
    row = conn.execute(sql).fetchone()
    non_missing_rows = int(row[0] or 0)
    null_rows = int(row[1] or 0)
    blank_rows = int(row[2] or 0)
    distinct_non_missing_values = int(row[3] or 0)
    days_with_any_value = int(row[4] or 0)
    min_value = row[5]
    max_value = row[6]
    avg_value = row[7]
    zero_count = int(row[8] or 0) if row[8] is not None else None
    coverage_pct = (100.0 * non_missing_rows / total_rows) if total_rows else 0.0
    day_coverage_pct = (100.0 * days_with_any_value / distinct_days) if distinct_days else 0.0
    return {
        "station_id": station_id,
        "attribute": column_name,
        "logical_type": col_kind(column_name),
        "rows_total": int(total_rows),
        "rows_non_missing": non_missing_rows,
        "rows_null": null_rows,
        "rows_blank": blank_rows,
        "coverage_pct_rows": coverage_pct,
        "distinct_non_missing_values": distinct_non_missing_values,
        "days_total": int(distinct_days),
        "days_with_any_value": days_with_any_value,
        "coverage_pct_days": day_coverage_pct,
        "min_value": normalize_value(column_name, min_value),
        "max_value": normalize_value(column_name, max_value),
        "avg_value": float(avg_value) if avg_value is not None else None,
        "zero_count": zero_count,
        "zero_pct_of_rows": (100.0 * zero_count / total_rows) if (zero_count is not None and total_rows) else None,
        "zero_pct_of_non_missing": (100.0 * zero_count / non_missing_rows)
        if (zero_count is not None and non_missing_rows)
        else None,
    }


def compute_station_year_summary(conn: sqlite3.Connection, station_id: str, column_name: str) -> list[dict[str, Any]]:
    non_missing_expr = sql_non_missing_expr(column_name)
    blank_expr = sql_blank_expr(column_name)
    sql = f"""
        SELECT
            SUBSTR(target_date_local, 1, 4) AS year_local,
            COUNT(*) AS rows_total,
            SUM(CASE WHEN {non_missing_expr} THEN 1 ELSE 0 END) AS rows_non_missing,
            SUM(CASE WHEN {column_name} IS NULL THEN 1 ELSE 0 END) AS rows_null,
            SUM(CASE WHEN {blank_expr} THEN 1 ELSE 0 END) AS rows_blank,
            COUNT(DISTINCT target_date_local) AS days_total,
            COUNT(DISTINCT CASE WHEN {non_missing_expr} THEN target_date_local END) AS days_with_any_value
        FROM {WU_TABLE}
        GROUP BY SUBSTR(target_date_local, 1, 4)
        ORDER BY year_local
    """
    out: list[dict[str, Any]] = []
    for year_local, rows_total, rows_non_missing, rows_null, rows_blank, days_total, days_with_any in conn.execute(sql):
        rows_total = int(rows_total or 0)
        rows_non_missing = int(rows_non_missing or 0)
        rows_null = int(rows_null or 0)
        rows_blank = int(rows_blank or 0)
        days_total = int(days_total or 0)
        days_with_any = int(days_with_any or 0)
        out.append(
            {
                "station_id": station_id,
                "attribute": column_name,
                "year_local": str(year_local or ""),
                "rows_total": rows_total,
                "rows_non_missing": rows_non_missing,
                "rows_null": rows_null,
                "rows_blank": rows_blank,
                "coverage_pct_rows": (100.0 * rows_non_missing / rows_total) if rows_total else 0.0,
                "days_total": days_total,
                "days_with_any_value": days_with_any,
                "coverage_pct_days": (100.0 * days_with_any / days_total) if days_total else 0.0,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_csv_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_exhaustive_catalog_json_gz(
    path: Path,
    *,
    generated_at_utc: str,
    data_root: Path,
    out_root: Path,
    stations: list[str],
    attributes: list[str],
    row_count_total: int,
    distinct_union_local_dates: int,
    overall_attribute_rows: list[dict[str, Any]],
    station_attribute_rows: list[dict[str, Any]],
    station_year_rows: list[dict[str, Any]],
    distinct_rows: list[dict[str, Any]],
) -> None:
    overall_by_attr = {row["attribute"]: row for row in overall_attribute_rows}
    station_rows_by_attr: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    year_rows_by_attr: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    distinct_rows_by_attr: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in station_attribute_rows:
        station_rows_by_attr[row["attribute"]].append(row)
    for row in station_year_rows:
        year_rows_by_attr[row["attribute"]].append(row)
    for row in distinct_rows:
        distinct_rows_by_attr[row["attribute"]].append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        f.write("{")
        f.write(
            json.dumps(
                {
                    "generated_at_utc": generated_at_utc,
                    "data_root": str(data_root),
                    "out_root": str(out_root),
                    "station_count": len(stations),
                    "stations": stations,
                    "attribute_count": len(attributes),
                    "row_count_total": row_count_total,
                    "distinct_union_local_dates": distinct_union_local_dates,
                },
                ensure_ascii=True,
                separators=(",", ":"),
            )[1:-1]
        )
        f.write(',"attributes":[')
        first_attr = True
        for attr in attributes:
            if not first_attr:
                f.write(",")
            first_attr = False
            payload = {
                "attribute": attr,
                "logical_type": col_kind(attr),
                "overall_coverage": overall_by_attr.get(attr, {}),
                "per_station_coverage": sorted(
                    station_rows_by_attr.get(attr, []),
                    key=lambda row: row["station_id"],
                ),
                "per_station_year_coverage": sorted(
                    year_rows_by_attr.get(attr, []),
                    key=lambda row: (row["station_id"], row["year_local"]),
                ),
                "exhaustive_possible_values": distinct_rows_by_attr.get(attr, []),
            }
            f.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
        f.write("]}")


def render_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep] + body)


def main() -> int:
    args = parse_args()
    logger = init_logger(args.log_level)
    data_root = Path(args.data_root).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(args.out_root).resolve() if args.out_root else (data_root / "reports" / f"wu_attribute_audit_{timestamp}")
    out_root.mkdir(parents=True, exist_ok=True)

    station_dbs = discover_station_dbs(data_root)
    first_conn = sqlite3.connect(str(next(iter(station_dbs.values()))))
    columns_meta = table_columns(first_conn, WU_TABLE)
    first_conn.close()
    attributes = [col["name"] for col in columns_meta]

    logger.info("WU_ATTRIBUTE_AUDIT_START stations=%s out_root=%s", ",".join(sorted(station_dbs)), out_root)

    station_overview_rows: list[dict[str, Any]] = []
    station_attribute_rows: list[dict[str, Any]] = []
    station_year_rows: list[dict[str, Any]] = []
    distinct_value_counts: dict[str, Counter[str]] = {attr: Counter() for attr in attributes}
    distinct_value_stations: dict[str, defaultdict[str, set[str]]] = {
        attr: defaultdict(set) for attr in attributes
    }
    all_target_dates: set[str] = set()

    for station_id, db_path in sorted(station_dbs.items()):
        logger.info("STATION_ANALYZE station=%s db=%s", station_id, db_path)
        conn = sqlite3.connect(str(db_path))
        station_overview = compute_station_overview(conn, station_id)
        station_overview_rows.append(station_overview)
        all_target_dates.update(
            row[0]
            for row in conn.execute(f"SELECT DISTINCT target_date_local FROM {WU_TABLE} WHERE target_date_local IS NOT NULL")
            if row[0] is not None
        )

        for attr in attributes:
            summary = compute_station_attribute_summary(
                conn=conn,
                station_id=station_id,
                column_name=attr,
                total_rows=station_overview["rows_total"],
                distinct_days=station_overview["distinct_target_dates"],
            )
            station_attribute_rows.append(summary)

            for year_row in compute_station_year_summary(conn, station_id, attr):
                station_year_rows.append(year_row)

            non_missing_expr = sql_non_missing_expr(attr)
            for raw_value, value_count in conn.execute(
                f"SELECT {attr}, COUNT(*) FROM {WU_TABLE} WHERE {non_missing_expr} GROUP BY {attr}"
            ):
                value_text = normalize_value(attr, raw_value)
                distinct_value_counts[attr][value_text] += int(value_count or 0)
                distinct_value_stations[attr][value_text].add(station_id)
        conn.close()

    total_rows_all = sum(int(row["rows_total"]) for row in station_overview_rows)
    total_station_days = sum(int(row["distinct_target_dates"]) for row in station_overview_rows)
    overall_attribute_rows: list[dict[str, Any]] = []
    for attr in attributes:
        attr_rows = [row for row in station_attribute_rows if row["attribute"] == attr]
        rows_total = sum(int(row["rows_total"]) for row in attr_rows)
        rows_non_missing = sum(int(row["rows_non_missing"]) for row in attr_rows)
        rows_null = sum(int(row["rows_null"]) for row in attr_rows)
        rows_blank = sum(int(row["rows_blank"]) for row in attr_rows)
        days_total = sum(int(row["days_total"]) for row in attr_rows)
        days_with_any_value = sum(int(row["days_with_any_value"]) for row in attr_rows)
        distinct_non_missing_values = len(distinct_value_counts[attr])
        coverage_values = [float(row["coverage_pct_rows"]) for row in attr_rows]
        day_coverage_values = [float(row["coverage_pct_days"]) for row in attr_rows]
        stations_with_any = [row["station_id"] for row in attr_rows if int(row["rows_non_missing"]) > 0]
        sorted_distinct_values = sorted(distinct_value_counts[attr], key=lambda x: sort_value_key(attr, x))
        avg_candidates = [row["avg_value"] for row in attr_rows if row["avg_value"] is not None]
        overall_attribute_rows.append(
            {
                "attribute": attr,
                "logical_type": col_kind(attr),
                "rows_total": rows_total,
                "rows_non_missing": rows_non_missing,
                "rows_null": rows_null,
                "rows_blank": rows_blank,
                "coverage_pct_rows": (100.0 * rows_non_missing / rows_total) if rows_total else 0.0,
                "distinct_non_missing_values": distinct_non_missing_values,
                "station_count_with_any": len(stations_with_any),
                "stations_with_any": ",".join(sorted(stations_with_any)),
                "station_min_coverage_pct_rows": min(coverage_values) if coverage_values else 0.0,
                "station_max_coverage_pct_rows": max(coverage_values) if coverage_values else 0.0,
                "station_avg_coverage_pct_rows": (sum(coverage_values) / len(coverage_values)) if coverage_values else 0.0,
                "station_day_coverage_pct_avg": (sum(day_coverage_values) / len(day_coverage_values))
                if day_coverage_values
                else 0.0,
                "station_day_coverage_pct_min": min(day_coverage_values) if day_coverage_values else 0.0,
                "station_day_coverage_pct_max": max(day_coverage_values) if day_coverage_values else 0.0,
                "station_days_total": days_total,
                "station_days_with_any_value": days_with_any_value,
                "coverage_pct_station_days": (100.0 * days_with_any_value / days_total) if days_total else 0.0,
                "global_min_value": sorted_distinct_values[0] if sorted_distinct_values else "",
                "global_max_value": sorted_distinct_values[-1] if sorted_distinct_values else "",
                "avg_of_station_means": (sum(avg_candidates) / len(avg_candidates)) if avg_candidates else None,
            }
        )

    distinct_rows: list[dict[str, Any]] = []
    for attr in attributes:
        for value_text in sorted(distinct_value_counts[attr], key=lambda x: sort_value_key(attr, x)):
            stations_present = sorted(distinct_value_stations[attr][value_text])
            distinct_rows.append(
                {
                    "attribute": attr,
                    "logical_type": col_kind(attr),
                    "distinct_value": value_text,
                    "overall_value_rows": int(distinct_value_counts[attr][value_text]),
                    "station_count": int(len(stations_present)),
                    "stations_present": ",".join(stations_present),
                }
            )

    station_overview_csv = out_root / "wu_station_overview.csv"
    attribute_overall_csv = out_root / "wu_attribute_overall_summary.csv"
    attribute_station_csv = out_root / "wu_attribute_station_summary.csv"
    attribute_year_csv_gz = out_root / "wu_attribute_year_summary.csv.gz"
    distinct_values_csv_gz = out_root / "wu_attribute_distinct_values.csv.gz"
    exhaustive_catalog_json_gz = out_root / "wu_attribute_exhaustive_catalog.json.gz"
    summary_json = out_root / "wu_attribute_audit_summary.json"
    report_md = out_root / "wu_attribute_audit_report.md"

    write_csv(station_overview_csv, station_overview_rows)
    write_csv(attribute_overall_csv, overall_attribute_rows)
    write_csv(attribute_station_csv, station_attribute_rows)
    write_csv_gz(attribute_year_csv_gz, station_year_rows)
    write_csv_gz(distinct_values_csv_gz, distinct_rows)
    generated_at_utc = now_utc()
    write_exhaustive_catalog_json_gz(
        exhaustive_catalog_json_gz,
        generated_at_utc=generated_at_utc,
        data_root=data_root,
        out_root=out_root,
        stations=sorted(station_dbs),
        attributes=attributes,
        row_count_total=total_rows_all,
        distinct_union_local_dates=len(all_target_dates),
        overall_attribute_rows=overall_attribute_rows,
        station_attribute_rows=station_attribute_rows,
        station_year_rows=station_year_rows,
        distinct_rows=distinct_rows,
    )

    overall_sorted = sorted(overall_attribute_rows, key=lambda row: row["attribute"])
    sparse_sorted = sorted(overall_attribute_rows, key=lambda row: row["coverage_pct_rows"])
    markdown_lines: list[str] = [
        "# WU Attribute Audit",
        "",
        f"- Generated at UTC: `{generated_at_utc}`",
        f"- Data root: `{data_root}`",
        f"- Stations: `{', '.join(sorted(station_dbs))}`",
        f"- Total WU rows across all stations: `{total_rows_all}`",
        f"- Total station-days across all stations: `{total_station_days}`",
        f"- Distinct local dates across the union: `{len(all_target_dates)}`",
        "",
        "## Station Overview",
        "",
        render_table(
            station_overview_rows,
            [
                "station_id",
                "rows_total",
                "distinct_target_dates",
                "min_target_date_local",
                "max_target_date_local",
                "avg_rows_per_day",
                "median_rows_per_day",
                "pct_days_ge_24_rows",
                "pct_days_ge_48_rows",
            ],
        ),
        "",
        "## Overall Attribute Coverage",
        "",
        render_table(
            overall_sorted,
            [
                "attribute",
                "logical_type",
                "coverage_pct_rows",
                "rows_non_missing",
                "rows_null",
                "rows_blank",
                "distinct_non_missing_values",
                "station_min_coverage_pct_rows",
                "station_max_coverage_pct_rows",
            ],
        ),
        "",
        "## Sparsest Attributes",
        "",
        render_table(
            sparse_sorted[:8],
            [
                "attribute",
                "logical_type",
                "coverage_pct_rows",
                "rows_non_missing",
                "rows_null",
                "rows_blank",
                "distinct_non_missing_values",
            ],
        ),
        "",
        "## Attribute Value Domains",
        "",
        "The exhaustive distinct-value inventory for every attribute is in the compressed CSV listed below. ",
        "For the report body, all non-time attributes are expanded inline. ",
        "The two timestamp-heavy fields are summarized by domain because literal inline enumeration would dominate the document.",
        "",
    ]

    distinct_lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in distinct_rows:
        distinct_lookup[row["attribute"]].append(row)

    for attr_row in overall_sorted:
        attr = attr_row["attribute"]
        markdown_lines.append(f"### `{attr}`")
        markdown_lines.append("")
        markdown_lines.append(f"- Logical type: `{attr_row['logical_type']}`")
        markdown_lines.append(
            f"- Row coverage: `{attr_row['coverage_pct_rows']:.4f}%` "
            f"(`{attr_row['rows_non_missing']}` / `{attr_row['rows_total']}`)"
        )
        markdown_lines.append(
            f"- Null rows: `{attr_row['rows_null']}` | blank rows: `{attr_row['rows_blank']}` | "
            f"distinct non-missing values: `{attr_row['distinct_non_missing_values']}`"
        )
        markdown_lines.append(
            f"- Station coverage range: `{attr_row['station_min_coverage_pct_rows']:.4f}%` .. "
            f"`{attr_row['station_max_coverage_pct_rows']:.4f}%`"
        )
        markdown_lines.append(
            f"- Station-day coverage range: `{attr_row['station_day_coverage_pct_min']:.4f}%` .. "
            f"`{attr_row['station_day_coverage_pct_max']:.4f}%`"
        )
        if attr_row["global_min_value"] != "" or attr_row["global_max_value"] != "":
            markdown_lines.append(
                f"- Global min/max non-missing value: `{attr_row['global_min_value']}` .. `{attr_row['global_max_value']}`"
            )
        if attr_row["avg_of_station_means"] is not None:
            markdown_lines.append(f"- Average of station means: `{attr_row['avg_of_station_means']:.6f}`")
        markdown_lines.append("- Possible values:")
        if attr in TIME_HEAVY_COLUMNS:
            markdown_lines.append(
                f"  - Timestamp-heavy domain with `{attr_row['distinct_non_missing_values']}` distinct values. "
                f"See `wu_attribute_distinct_values.csv.gz` for the exact exhaustive list."
            )
        else:
            values = distinct_lookup[attr]
            if len(values) <= INLINE_DISTINCT_LIMIT:
                for value_row in values:
                    value_text = value_row["distinct_value"]
                    shown = value_text if value_text != "" else "<EMPTY>"
                    markdown_lines.append(
                        f"  - `{shown}` | rows=`{value_row['overall_value_rows']}` | "
                        f"stations=`{value_row['station_count']}` | present=`{value_row['stations_present']}`"
                    )
            else:
                markdown_lines.append(
                    f"  - Domain too large for inline expansion (`{len(values)}` values). "
                    f"See `wu_attribute_distinct_values.csv.gz`."
                )
        markdown_lines.append("")

    summary_payload = {
        "generated_at_utc": generated_at_utc,
        "data_root": str(data_root),
        "out_root": str(out_root),
        "station_count": len(station_dbs),
        "stations": sorted(station_dbs),
        "attributes": attributes,
        "station_overview_csv": str(station_overview_csv),
        "attribute_overall_csv": str(attribute_overall_csv),
        "attribute_station_csv": str(attribute_station_csv),
        "attribute_year_csv_gz": str(attribute_year_csv_gz),
        "distinct_values_csv_gz": str(distinct_values_csv_gz),
        "exhaustive_catalog_json_gz": str(exhaustive_catalog_json_gz),
        "report_md": str(report_md),
        "row_count_total": total_rows_all,
        "distinct_union_local_dates": len(all_target_dates),
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    report_md.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    logger.info("WU_ATTRIBUTE_AUDIT_DONE summary=%s", summary_json)
    print(json.dumps(summary_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
