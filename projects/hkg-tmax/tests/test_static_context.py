from __future__ import annotations

import csv

from hkg_tmax.static_context import build_static_context


def test_static_context_builds_station_distance_and_solar_outputs(
    repo_root, monkeypatch, tmp_path
) -> None:
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("HKG_TMAX_DATA_ROOT", str(data_root))
    raw = data_root / "raw" / "isd-history.csv"
    raw.parent.mkdir(parents=True)
    raw.write_text(
        "\n".join(
            [
                "USAF,WBAN,STATION NAME,CTRY,STATE,ICAO,LAT,LON,ELEV(M),BEGIN,END",
                "450050,99999,HONG KONG OBSERVATORY,CH,,,+22.300,+114.167,+0062.0,19460901,20180930",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifests = data_root / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "retrieval_ledger.csv").write_text(
        "\n".join(
            [
                "retrieval_id,source_id,provider,retrieved_at,status,http_status,request_url,final_url,etag,last_modified,content_sha256,content_length,content_path,sidecar_path,deduplicated,error",
                f"ok,noaa_isd_history,NOAA,2026-06-19T00:00:00Z,success,200,https://example.test,https://example.test,,,,0,{raw},,false,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    outputs = build_static_context(repo_root, solar_year=2026)

    assert outputs.station_registry_csv.exists()
    assert outputs.station_distance_csv.exists()
    assert outputs.solar_geometry_csv.exists()
    station_rows = list(csv.DictReader(outputs.station_registry_csv.open(newline="", encoding="utf-8")))
    distance_rows = list(csv.DictReader(outputs.station_distance_csv.open(newline="", encoding="utf-8")))
    solar_rows = list(csv.DictReader(outputs.solar_geometry_csv.open(newline="", encoding="utf-8")))
    assert {row["station_id"] for row in station_rows} == {"HKO", "450050-99999"}
    assert any(row["origin_station_id"] == "HKO" for row in distance_rows)
    assert len(solar_rows) == 365 * 5 * 2
