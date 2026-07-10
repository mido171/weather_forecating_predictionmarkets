from __future__ import annotations

from hkg_tmax.acquisition_contracts import (
    PAIR_REQUIRED_FIELDS,
    validate_gridded_policy,
    validate_historical_live_pairs,
    write_historical_live_pair_artifacts,
)
from hkg_tmax.config import SourceCatalog, load_yaml


def test_historical_live_pairs_cover_required_reset_fields(repo_root) -> None:
    checks = validate_historical_live_pairs(repo_root)
    data = load_yaml(repo_root / "config" / "historical_live_pairs.yaml")
    pairs = data["pairs"]

    assert checks
    assert len(pairs) >= 15
    assert {field for pair in pairs for field in PAIR_REQUIRED_FIELDS if field in pair} == set(
        PAIR_REQUIRED_FIELDS
    )
    assert not any("polymarket" in str(pair).lower() for pair in pairs)
    assert {
        "gfs_operational",
        "gefs_operational_and_reforecast",
        "era5_era5_land_reanalysis",
        "gpm_imerg_precipitation",
        "static_geospatial_station_context",
    }.issubset({pair["family_id"] for pair in pairs})


def test_gridded_policy_has_required_domains_and_gated_bulk_downloads(repo_root) -> None:
    checks = validate_gridded_policy(repo_root)
    data = load_yaml(repo_root / "config" / "gridded_acquisition_policy.yaml")

    assert checks
    assert {"local_hk", "regional_schina", "synoptic_asia"}.issubset(data["domains"])
    families = {family["family_id"]: family for family in data["families"]}
    assert {"gfs_operational", "gefs_operational_and_reforecast", "era5", "oisst_v21"}.issubset(
        families
    )
    assert families["hko_current_satellite"]["bulk_download_allowed"] is True
    assert all(
        not family["bulk_download_allowed"]
        for family_id, family in families.items()
        if family_id != "hko_current_satellite"
    )


def test_historical_live_pair_artifacts_are_rendered(repo_root, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HKG_TMAX_DATA_ROOT", str(tmp_path / "data_root"))
    paths = write_historical_live_pair_artifacts(repo_root)

    assert repo_root / "metadata" / "historical_live_pairs.csv" in paths
    assert repo_root / "metadata" / "historical_live_pairs.parquet" in paths
    report = repo_root / "reports" / "historical_live_pairing.md"
    assert report in paths
    assert "Historical / Live Pairing" in report.read_text(encoding="utf-8")


def test_new_required_source_contracts_are_in_data_sources(repo_root) -> None:
    catalog = SourceCatalog.from_path(repo_root / "config" / "data_sources.yaml")
    source_ids = {source.id for source in catalog.sources}

    assert {
        "noaa_oisst_v21",
        "himawari_ahi_noaa_aws",
        "gpm_imerg_final",
        "gpm_imerg_early_late",
        "cams_eac4_reanalysis",
        "cams_global_composition_forecasts",
        "hong_kong_epd_air_quality_observations",
    }.issubset(source_ids)
