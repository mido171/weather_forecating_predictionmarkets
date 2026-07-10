from datetime import UTC, datetime

from hkg_tmax.acquisition import store_content_addressed_retrieval
from hkg_tmax.bronze import build_bronze_latest


def test_build_bronze_clmmaxt_from_ledger(tmp_path) -> None:
    content = (
        b'"Daily Maximum Temperature (deg C) at the Hong Kong Observatory"\n'
        b'Year,Month,Day,Value,"Data Completeness"\n'
        b"2026,6,18,31.2,C\n"
    )
    record = store_content_addressed_retrieval(
        tmp_path,
        source_id="hko_clmmaxt_hko",
        provider="HKO",
        content=content,
        retrieved_at=datetime(2026, 6, 19, 1, tzinfo=UTC),
        extension="csv",
        metadata={"requested_url": "https://example.test/clm.csv", "http_status": 200},
    )

    dataset = build_bronze_latest(tmp_path, source_id="hko_clmmaxt_hko")

    assert dataset.content_sha256 == record.content_sha256
    assert dataset.row_count == 1
    assert dataset.parquet_path.exists()
    assert dataset.metadata_path.exists()


def test_build_bronze_live_temperature_preserves_hkt_timestamp(tmp_path) -> None:
    content = (
        b"Date time,Automatic Weather Station,Air Temperature(degree Celsius)\n"
        b"202606190620,HK Observatory,28.2\n"
        b"202606190620,King's Park,27.8\n"
    )
    store_content_addressed_retrieval(
        tmp_path,
        source_id="hko_latest_1min_temperature",
        provider="HKO",
        content=content,
        retrieved_at=datetime(2026, 6, 18, 22, 30, tzinfo=UTC),
        extension="csv",
        metadata={"requested_url": "https://example.test/latest.csv", "http_status": 200},
    )

    dataset = build_bronze_latest(tmp_path, source_id="hko_latest_1min_temperature")

    assert dataset.row_count == 2
    assert dataset.parquet_path.exists()
