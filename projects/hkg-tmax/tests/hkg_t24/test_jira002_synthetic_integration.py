from __future__ import annotations

from datetime import date

from fixtures.synthetic_h24n.create_synthetic_fixture import synthetic_feature_families
from hkg_t24.features.matrix_builder import build_strict_matrix_rows
from hkg_t24.models.experts import generate_expert_oof_predictions
from hkg_t24.models.folds import FoldSpec
from hkg_t24.models.oof import check_oof_integrity


def test_synthetic_strict_feature_matrix_and_oof_end_to_end() -> None:
    dates, labels, target_memory, official, nwp, online = synthetic_feature_families()
    rows = build_strict_matrix_rows(
        target_dates=dates,
        target_memory_by_date=target_memory,
        official_by_date=official,
        nwp_by_date=nwp,
        online_by_date=online,
        labels_by_date=labels,
    )
    assert rows
    assert all(
        name.startswith(("calendar__", "official__", "target__", "online__", "gfs__", "gefsmean__", "gefsens__"))
        for row in rows
        for name in row.features
    )
    assert all("target__lag1_" not in name for row in rows for name in row.features)

    predictions = generate_expert_oof_predictions(
        rows,
        [
            FoldSpec(
                "synthetic_integration",
                date(2021, 2, 10),
                date(2021, 2, 28),
                date(2021, 3, 1),
                date(2021, 3, 11),
            )
        ],
    )
    assert predictions
    assert check_oof_integrity(predictions).passed
    assert {prediction.expert_id for prediction in predictions} >= {
        "E0_OFFICIAL_RAW_ANCHOR",
        "E2_TARGET_MEMORY",
        "E4_GFS_MOS",
        "E5_GEFS_ENSEMBLE",
        "E10_DIAGNOSTIC_PROXY",
    }
