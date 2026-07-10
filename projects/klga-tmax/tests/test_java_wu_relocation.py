from __future__ import annotations

from klga_tmax.providers.wunderground.java_truth import _find_extraction_root


def test_java_wu_resolves_normalized_monorepo_app() -> None:
    root = _find_extraction_root()

    assert (root / "apps" / "ingestion-service" / "pom.xml").is_file()
    assert not (root / "ingestion-service" / "pom.xml").exists()
