from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.load_datasets_to_mysql import (
    ColumnPlan,
    DatasetFile,
    LoaderConfig,
    TablePlan,
    build_create_table_sql,
    clean_column_names,
    config_from_args,
    discover_dataset_files,
    infer_csv_mysql_type,
    mysql_type_for_arrow_type,
    plan_table_names,
    run_loader,
    sanitize_identifier,
)


def dataset_file(path: Path, root: Path, file_type: str) -> DatasetFile:
    return DatasetFile(
        path=path,
        relative_path=path.relative_to(root).as_posix(),
        file_type=file_type,
        bytes=path.stat().st_size,
        sha256="a" * 64,
    )


def test_sanitize_identifier_handles_mysql_shape_and_length() -> None:
    assert sanitize_identifier("Forecast Max (C)", fallback="dataset") == "forecast_max_c"
    assert sanitize_identifier("2000 file", fallback="dataset") == "dataset_2000_file"
    assert sanitize_identifier("select", fallback="col") == "select_col"
    long_name = sanitize_identifier("x" * 100, fallback="dataset")
    assert len(long_name) <= 64
    assert long_name.startswith("x")


def test_table_name_collision_gets_hash_suffix(tmp_path: Path) -> None:
    first = tmp_path / "one" / "same.parquet"
    second = tmp_path / "two" / "same.parquet"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    table_names = plan_table_names(
        [
            dataset_file(first, tmp_path, "parquet"),
            dataset_file(second, tmp_path, "parquet"),
        ],
    )

    assert table_names[first.relative_to(tmp_path).as_posix()] == "same"
    assert table_names[second.relative_to(tmp_path).as_posix()].startswith("same_")
    assert table_names[first.relative_to(tmp_path).as_posix()] != table_names[
        second.relative_to(tmp_path).as_posix()
    ]


def test_column_names_avoid_ingest_metadata_and_duplicates() -> None:
    cleaned = clean_column_names(
        [
            "Issue At HKT",
            "issue-at-hkt",
            "ingest_source_sha256",
            "where",
        ],
    )

    assert cleaned[0] == "issue_at_hkt"
    assert cleaned[1].startswith("issue_at_hkt_")
    assert cleaned[2].startswith("ingest_source_sha256_")
    assert cleaned[3].startswith("where_col")


def test_arrow_type_mapping() -> None:
    pa = pytest.importorskip("pyarrow")

    assert mysql_type_for_arrow_type(pa.int64()) == "BIGINT"
    assert mysql_type_for_arrow_type(pa.float64()) == "DOUBLE"
    assert mysql_type_for_arrow_type(pa.bool_()) == "BOOLEAN"
    assert mysql_type_for_arrow_type(pa.date32()) == "DATE"
    assert mysql_type_for_arrow_type(pa.timestamp("us")) == "DATETIME(6)"
    assert mysql_type_for_arrow_type(pa.list_(pa.int64())) == "JSON"
    assert mysql_type_for_arrow_type(pa.struct([("x", pa.int64())])) == "JSON"
    assert mysql_type_for_arrow_type(pa.string()) == "TEXT"


def test_csv_type_inference() -> None:
    assert infer_csv_mysql_type("row_count", ["1", "2", "3"]) == "BIGINT"
    assert infer_csv_mysql_type("score", ["1.2", "3.4"]) == "DOUBLE"
    assert infer_csv_mysql_type("ok", ["true", "false"]) == "BOOLEAN"
    assert infer_csv_mysql_type("target_date", ["2026-06-22", "2026-06-23"]) == "DATE"
    assert infer_csv_mysql_type("issued_at_utc", ["2026-06-22T01:02:03Z"]) == "DATETIME(6)"
    assert infer_csv_mysql_type("payload", ['{"a": 1}', "[1, 2]"]) == "JSON"
    assert infer_csv_mysql_type("title", ["Weather Report"]) == "TEXT"


def test_create_table_sql_contains_ingest_metadata_and_source_columns(tmp_path: Path) -> None:
    source = tmp_path / "items.parquet"
    source.write_text("placeholder", encoding="utf-8")
    table_plan = TablePlan(
        source_file=dataset_file(source, tmp_path, "parquet"),
        table_name="items",
        row_count=1,
        columns=(
            ColumnPlan("issue_at_hkt", "issue_at_hkt", "DATETIME(6)", "datetime", True, 1),
            ColumnPlan("forecast_max_c", "forecast_max_c", "DOUBLE", "float", True, 2),
        ),
    )

    sql = build_create_table_sql(table_plan)

    assert "CREATE TABLE `items`" in sql
    assert "`ingest_row_id` BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY" in sql
    assert "`ingest_source_path` TEXT NOT NULL" in sql
    assert "`issue_at_hkt` DATETIME(6) NULL" in sql
    assert "`forecast_max_c` DOUBLE NULL" in sql
    assert "ENGINE=InnoDB DEFAULT CHARSET=utf8mb4" in sql


def test_discovery_includes_parquet_csv_and_zip_only(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    parquet_path = tmp_path / "a" / "table.parquet"
    csv_path = tmp_path / "manifest.csv"
    zip_path = tmp_path / "payload.zip"
    ignored_path = tmp_path / "README.md"
    parquet_path.write_text("pq", encoding="utf-8")
    csv_path.write_text("a\n1\n", encoding="utf-8")
    zip_path.write_bytes(b"zip")
    ignored_path.write_text("ignore", encoding="utf-8")

    files = discover_dataset_files(tmp_path)

    assert [file.relative_path for file in files] == [
        "a/table.parquet",
        "manifest.csv",
        "payload.zip",
    ]
    assert {file.file_type for file in files} == {"parquet", "csv", "zip"}


def test_dry_run_loads_tiny_fixtures_without_mysql(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    datasets_root = tmp_path / "datasets"
    datasets_root.mkdir()
    pd.DataFrame(
        [
            {"target_date": "2026-06-22", "forecast_max_c": 31.5},
            {"target_date": "2026-06-23", "forecast_max_c": 32.0},
        ],
    ).to_parquet(datasets_root / "daily_forecasts.parquet", index=False)
    (datasets_root / "MANIFEST.csv").write_text(
        "dataset_folder,file_name,bytes\nx,daily_forecasts.parquet,123\n",
        encoding="utf-8",
    )
    (datasets_root / "payload.zip").write_bytes(b"zip")

    config = LoaderConfig(
        datasets_root=datasets_root,
        host="127.0.0.1",
        port=3306,
        user="root",
        password="root",
        database="hkg_tmax_research",
        mode="replace",
        batch_size=10,
        connect_retries=1,
        connect_retry_delay_seconds=0.01,
        connect_timeout_seconds=1.0,
        include_csv=True,
        dry_run=True,
    )

    result = run_loader(config)

    assert result["status"] == "dry_run"
    assert result["tables"] == 2
    assert result["source_rows"] == 3
    assert result["zip_payloads_registered_only"] == 1


def test_config_rejects_unsafe_database_name() -> None:
    parser = pytest.importorskip("scripts.load_datasets_to_mysql").build_parser()
    args = parser.parse_args(["--database", "bad-name"])

    with pytest.raises(Exception, match="database name"):
        config_from_args(args)


@pytest.mark.skipif(os.environ.get("RUN_MYSQL_TESTS") != "1", reason="requires live MySQL")
def test_mysql_integration_loads_two_fixture_tables(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    datasets_root = tmp_path / "datasets"
    datasets_root.mkdir()
    pd.DataFrame([{"x": 1}, {"x": 2}]).to_parquet(datasets_root / "one.parquet", index=False)
    (datasets_root / "two.csv").write_text("name,value\na,1\nb,2\n", encoding="utf-8")

    config = LoaderConfig(
        datasets_root=datasets_root,
        host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        user=os.environ.get("MYSQL_USER", "root"),
        password=os.environ.get("MYSQL_PASSWORD", "root"),
        database=f"hkg_tmax_test_{os.getpid()}",
        mode="replace",
        batch_size=2,
        connect_retries=5,
        connect_retry_delay_seconds=0.5,
        connect_timeout_seconds=5.0,
        include_csv=True,
        dry_run=False,
    )

    result = run_loader(config)

    assert result["status"] == "succeeded"
    assert result["tables_loaded"] == 2
    assert result["rows_loaded"] == 4
