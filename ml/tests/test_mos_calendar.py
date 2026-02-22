from datetime import date, datetime, timezone

from weather_ml.mos_calendar import expected_asof_utc
from weather_ml.mos_config import MosDatasetConfig


def test_expected_asof_utc_t_minus_one_12z():
    cfg = MosDatasetConfig(
        station_id="KMIA",
        station_zoneid="America/New_York",
        feature_version="test",
        build_start_asof=date(2007, 1, 1),
        output_start_asof=date(2010, 1, 1),
        end_asof=date(2010, 1, 10),
        models=["GFS", "NAM"],
        variables=[],
    ).normalized()
    target_date = date(2024, 1, 2)
    expected = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert expected_asof_utc(target_date, cfg) == expected
