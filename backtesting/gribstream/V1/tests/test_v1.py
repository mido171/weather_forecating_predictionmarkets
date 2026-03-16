from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from backtesting.gribstream.V1 import db
from backtesting.gribstream.V1.config import kelvin_to_f, local_day_window_utc
from backtesting.gribstream.V1.derive_daily_tmax import _latest_revision_rows, select_raw_tmax
from backtesting.gribstream.V1.gribstream_client import _parse_csv_body
from backtesting.gribstream.V1.weights import ErrorRecord, _apply_family_cap, _apply_model_cap, _training_records


class V1PipelineTests(unittest.TestCase):
    def test_kelvin_to_fahrenheit(self) -> None:
        self.assertAlmostEqual(kelvin_to_f(273.15), 32.0, places=6)
        self.assertAlmostEqual(kelvin_to_f(300.0), 80.33, places=2)

    def test_local_day_window_standard_time(self) -> None:
        start_utc, end_utc = local_day_window_utc(date(2024, 1, 15), "America/New_York")
        self.assertEqual(start_utc.isoformat(), "2024-01-15T05:00:00+00:00")
        self.assertEqual(end_utc.isoformat(), "2024-01-16T05:00:00+00:00")

    def test_dst_boundaries(self) -> None:
        march_start, march_end = local_day_window_utc(date(2024, 3, 10), "America/New_York")
        november_start, november_end = local_day_window_utc(date(2024, 11, 3), "America/New_York")
        self.assertEqual((march_end - march_start).total_seconds() / 3600.0, 23.0)
        self.assertEqual((november_end - november_start).total_seconds() / 3600.0, 25.0)

    def test_parse_csv_sorts_unsorted_rows(self) -> None:
        csv_text = "\n".join(
            [
                "forecasted_at,forecasted_time,lat,lon,name,member,TMP|2 m above ground|",
                "2024-01-01T10:00:00Z,2024-01-01T16:00:00Z,40.78333,-73.96667,KNYC,,275",
                "2024-01-01T09:00:00Z,2024-01-01T14:00:00Z,40.78333,-73.96667,KNYC,,274",
                "2024-01-01T11:00:00Z,2024-01-01T14:00:00Z,40.78333,-73.96667,KNYC,,276",
            ]
        )
        rows = _parse_csv_body(csv_text)
        self.assertEqual(rows[0].forecasted_time_utc.isoformat(), "2024-01-01T14:00:00+00:00")
        self.assertEqual(rows[0].forecasted_at_utc.isoformat(), "2024-01-01T09:00:00+00:00")
        self.assertEqual(rows[1].forecasted_at_utc.isoformat(), "2024-01-01T11:00:00+00:00")

    def test_latest_revision_rows_keep_latest_forecasted_at(self) -> None:
        rows = [
            {
                "id": 1,
                "forecasted_time_utc": "2024-01-01T14:00:00Z",
                "forecasted_at_utc": "2024-01-01T09:00:00Z",
                "variable_name": "TMP",
                "variable_level": "2 m above ground",
                "variable_info": "",
                "member": None,
            },
            {
                "id": 2,
                "forecasted_time_utc": "2024-01-01T14:00:00Z",
                "forecasted_at_utc": "2024-01-01T11:00:00Z",
                "variable_name": "TMP",
                "variable_level": "2 m above ground",
                "variable_info": "",
                "member": None,
            },
        ]
        latest = _latest_revision_rows(rows)
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0]["id"], 2)

    def test_selected_raw_tmax_priority(self) -> None:
        self.assertEqual(
            select_raw_tmax(native_tmax_f=80.0, interpolated_tmax_f=79.0, snapshot_tmax_f=81.0),
            (80.0, "native_tmax"),
        )
        self.assertEqual(
            select_raw_tmax(native_tmax_f=None, interpolated_tmax_f=79.0, snapshot_tmax_f=81.0),
            (79.0, "interpolated_snapshot_tmax"),
        )
        self.assertEqual(
            select_raw_tmax(native_tmax_f=None, interpolated_tmax_f=None, snapshot_tmax_f=81.0),
            (81.0, "snapshot_tmax"),
        )

    def test_rolling_weights_use_only_prior_dates(self) -> None:
        history = [
            ErrorRecord(settlement_date_local=date(2024, 1, 1), error_f=1.0),
            ErrorRecord(settlement_date_local=date(2024, 1, 2), error_f=2.0),
            ErrorRecord(settlement_date_local=date(2024, 1, 4), error_f=99.0),
        ]
        training = _training_records(history, date(2024, 1, 4))
        self.assertEqual([record.settlement_date_local for record in training], [date(2024, 1, 1), date(2024, 1, 2)])

    def test_insufficient_history_exclusion_logic(self) -> None:
        history = [ErrorRecord(settlement_date_local=date(2024, 1, 1), error_f=1.0)]
        training = _training_records(history, date(2024, 1, 3))
        self.assertLess(len(training), 45)

    def test_model_cap_and_family_cap_logic(self) -> None:
        model_weights = {"a": 10.0, "b": 3.0, "c": 1.0}
        capped_model_weights, capped_models = _apply_model_cap(model_weights, cap=0.35)
        self.assertIn("a", capped_models)
        self.assertAlmostEqual(sum(capped_model_weights.values()), 1.0, places=6)
        families = {"a": "fam1", "b": "fam1", "c": "fam2"}
        family_weights, capped_families = _apply_family_cap(capped_model_weights, families, cap=0.50)
        self.assertIn("fam1", capped_families)
        self.assertAlmostEqual(sum(family_weights.values()), 1.0, places=6)
        self.assertLessEqual(family_weights["a"] + family_weights["b"], 0.500001)

    def test_idempotent_reruns_do_not_duplicate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite3"
            connection = db.connect(db_path)
            try:
                db.initialize_database(connection)
                row = {
                    "request_id": "req1",
                    "model_code": "hrrr",
                    "station_id": "KNYC",
                    "settlement_date_local": "2024-01-01",
                    "as_of_utc": "2024-01-01T13:00:00Z",
                    "forecasted_at_utc": "2024-01-01T12:00:00Z",
                    "forecasted_time_utc": "2024-01-01T18:00:00Z",
                    "forecasted_time_local": "2024-01-01T13:00:00-05:00",
                    "forecasted_date_local": "2024-01-01",
                    "lat": 40.78333,
                    "lon": -73.96667,
                    "coord_name": "KNYC",
                    "variable_name": "TMP",
                    "variable_level": "2 m above ground",
                    "variable_info": "",
                    "member": None,
                    "value_native": 280.0,
                    "unit_native": "K",
                    "value_f": 44.33,
                    "lead_minutes": 360,
                    "inserted_at_utc": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                }
                db.insert_gribstream_raw_forecasts(connection, [row, row])
                connection.commit()
                self.assertEqual(db.table_row_count(connection, "gribstream_raw_forecasts"), 1)
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
