from hkg_tmax.config import SourceCatalog
from hkg_tmax.fetch import infer_extension
from hkg_tmax.sources import unresolved_template, write_source_inventory


def test_source_catalog_is_unique_and_large(repo_root) -> None:
    catalog = SourceCatalog.from_path(repo_root / "config" / "sources" / "data_sources.yaml")
    assert len(catalog.sources) >= 30
    assert catalog.get("hko_clmmaxt_hko").point_in_time_status == "PROXY_WITH_LIMITATIONS"
    assert catalog.get("copernicus_era5").point_in_time_status == "RETROSPECTIVE_ONLY"


def test_source_inventory_report(repo_root, tmp_path) -> None:
    catalog = SourceCatalog.from_path(repo_root / "config" / "sources" / "data_sources.yaml")
    root = tmp_path
    path = write_source_inventory(root, catalog)
    text = path.read_text()
    assert "hko_daily_extract" in text
    assert "RETROSPECTIVE_ONLY" in text


def test_extension_prefers_content_type_over_php() -> None:
    assert infer_extension("https://example.test/data.php", "text/csv") == "csv"
    assert infer_extension("https://example.test/file.grib2", None) == "grib2"


def test_template_detection() -> None:
    assert unresolved_template("https://example.test/{slug}")
    assert not unresolved_template("https://example.test/data")
