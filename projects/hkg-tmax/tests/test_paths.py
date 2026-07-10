from __future__ import annotations

from pathlib import Path

import pytest

from hkg_tmax.paths import (
    ArchivePathResolutionError,
    ProjectPaths,
    archive_reference_fields,
    configured_input_path,
    find_project_root,
    resolve_archive_content_path,
)


def test_project_paths_discover_moved_src_layout() -> None:
    root = find_project_root(Path(__file__))
    paths = ProjectPaths.discover(Path(__file__), environ={})

    assert paths.project_root == root
    assert paths.data_root == root / "data"
    assert paths.run_root == root / "data" / "runs"
    assert paths.config_root == root / "config"
    assert paths.db_root == (root / "db" if (root / "db").is_dir() else root)


def test_project_paths_resolve_relative_overrides_from_project_root(tmp_path) -> None:
    paths = ProjectPaths.from_project_root(
        tmp_path,
        environ={
            "HKG_TMAX_DATA_ROOT": "runtime/data",
            "HKG_TMAX_RUN_ROOT": "runtime/runs",
            "HKG_TMAX_CONFIG_ROOT": "settings",
            "HKG_TMAX_DB_ROOT": "database",
            "HKG_TMAX_STORAGE_ROOT_ID": "test-root",
        },
    )

    assert paths.data_root == tmp_path / "runtime" / "data"
    assert paths.run_root == tmp_path / "runtime" / "runs"
    assert paths.config_root == tmp_path / "settings"
    assert paths.db_root == tmp_path / "database"
    assert paths.storage_root_id == "test-root"


def test_project_paths_load_only_path_settings_from_local_dotenv(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HKG_TMAX_DATA_ROOT", raising=False)
    monkeypatch.delenv("HKG_TMAX_RUN_ROOT", raising=False)
    (tmp_path / ".env").write_text(
        "HKG_TMAX_DATA_ROOT=external/data\n"
        "HKG_TMAX_RUN_ROOT=external/runs\n"
        "GRIBSTREAM_TOKEN=must-not-be-loaded-here\n",
        encoding="utf-8",
    )

    paths = ProjectPaths.from_project_root(tmp_path)

    assert paths.data_root == (tmp_path / "external" / "data").resolve()
    assert paths.run_root == (tmp_path / "external" / "runs").resolve()


def test_explicit_environment_keeps_project_paths_hermetic(tmp_path) -> None:
    (tmp_path / ".env").write_text(
        "HKG_TMAX_DATA_ROOT=external/data\n",
        encoding="utf-8",
    )

    paths = ProjectPaths.from_project_root(tmp_path, environ={})

    assert paths.data_root == (tmp_path / "data").resolve()


def test_project_discovery_honors_configured_root(tmp_path) -> None:
    expected = find_project_root(Path(__file__))

    assert find_project_root(
        tmp_path,
        environ={"HKG_TMAX_ROOT": str(expected)},
    ) == expected


def test_configured_input_path_prefers_file_specific_environment(tmp_path) -> None:
    paths = ProjectPaths.from_project_root(tmp_path, environ={})

    resolved = configured_input_path(
        paths,
        "HKG_TEST_SPEC_PATH",
        "spec.md",
        environ={"HKG_TEST_SPEC_PATH": "operator/spec.md"},
    )

    assert resolved == (tmp_path / "operator" / "spec.md").resolve()


def test_configured_input_path_uses_external_input_root(tmp_path) -> None:
    paths = ProjectPaths.from_project_root(tmp_path, environ={})

    resolved = configured_input_path(
        paths,
        "HKG_TEST_SPEC_PATH",
        "spec.md",
        environ={"HKG_TMAX_INPUT_ROOT": "external/inputs"},
    )

    assert resolved == (tmp_path / "external" / "inputs" / "spec.md").resolve()


def test_configured_input_path_uses_existing_home_relative_transition_fallback(tmp_path) -> None:
    paths = ProjectPaths.from_project_root(tmp_path, environ={})
    legacy = tmp_path / "home" / "Downloads" / "spec.md"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("historical input", encoding="utf-8")

    resolved = configured_input_path(
        paths,
        "HKG_TEST_SPEC_PATH",
        "spec.md",
        environ={},
        legacy_home_relative=Path("Downloads") / "spec.md",
        home=tmp_path / "home",
    )

    assert resolved == legacy.resolve()


def test_archive_reference_fields_preserve_absolute_provenance(tmp_path) -> None:
    content = tmp_path / "raw" / "source" / "payload.csv"
    sidecar = tmp_path / "raw" / "source" / "payload.metadata.json"

    fields = archive_reference_fields(
        data_root=tmp_path,
        content_path=content,
        sidecar_path=sidecar,
        storage_root_id="fixture-root",
    )

    assert fields["storage_schema_version"] == 2
    assert fields["storage_root_id"] == "fixture-root"
    assert fields["content_relpath"] == "raw/source/payload.csv"
    assert fields["sidecar_relpath"] == "raw/source/payload.metadata.json"
    assert fields["content_path"] == str(content.resolve())
    assert fields["legacy_content_path"] == str(content.resolve())


def test_archive_resolution_prefers_existing_legacy_absolute_path(tmp_path) -> None:
    data_root = tmp_path / "data"
    relative = data_root / "raw" / "source" / "payload.csv"
    relative.parent.mkdir(parents=True)
    relative.write_bytes(b"relative")
    legacy = tmp_path / "legacy.csv"
    legacy.write_bytes(b"legacy")

    resolved = resolve_archive_content_path(
        {
            "content_path": str(legacy.resolve()),
            "content_relpath": "raw/source/payload.csv",
        },
        data_root=data_root,
    )

    assert resolved == legacy.resolve()


def test_archive_resolution_checks_legacy_absolute_when_content_path_is_relative(tmp_path) -> None:
    legacy = tmp_path / "legacy.csv"
    legacy.write_bytes(b"legacy")

    resolved = resolve_archive_content_path(
        {
            "content_path": "raw/source/missing.csv",
            "legacy_content_path": str(legacy.resolve()),
        },
        data_root=tmp_path / "data",
    )

    assert resolved == legacy.resolve()


def test_archive_resolution_rebases_legacy_data_suffix(tmp_path) -> None:
    data_root = tmp_path / "relocated"
    relocated = data_root / "raw" / "source" / "payload.csv"
    relocated.parent.mkdir(parents=True)
    relocated.write_bytes(b"relocated")

    resolved = resolve_archive_content_path(
        {"content_path": r"C:\old-machine\hkg-data\raw\source\payload.csv"},
        data_root=data_root,
    )

    assert resolved == relocated.resolve()


def test_archive_resolution_uses_content_addressed_candidate(tmp_path) -> None:
    digest = "a" * 64
    candidate = tmp_path / "raw" / "objects" / "aa" / f"{digest}.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"content addressed")

    resolved = resolve_archive_content_path(
        {
            "content_path": r"C:\missing\objects\payload.json",
            "content_sha256": digest,
        },
        data_root=tmp_path,
    )

    assert resolved == candidate.resolve()


def test_archive_resolution_uses_relocated_sidecar_sibling(tmp_path) -> None:
    sidecar = tmp_path / "relocated" / "payload.metadata.json"
    content = sidecar.with_name("payload.csv")
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("{}", encoding="utf-8")
    content.write_bytes(b"sibling")

    resolved = resolve_archive_content_path(
        {
            "content_path": r"C:\old-machine\archive\payload.csv",
            "metadata": {"extension_inferred": "csv"},
        },
        data_root=tmp_path,
        sidecar_path=sidecar,
    )

    assert resolved == content.resolve()


def test_archive_resolution_rejects_traversal_and_missing_content(tmp_path) -> None:
    with pytest.raises(ArchivePathResolutionError, match="Unsafe archive relative path"):
        resolve_archive_content_path(
            {"content_relpath": "../outside.csv"},
            data_root=tmp_path,
        )

    with pytest.raises(ArchivePathResolutionError, match="Unable to resolve"):
        resolve_archive_content_path({}, data_root=tmp_path)


def test_production_modules_share_central_project_discovery() -> None:
    from hkg_t24 import cli as t24_cli
    from hkg_tmax_db import cli as db_cli
    from hkg_tmax_demo_trading import api as demo_api
    from hkg_tmax_demo_trading import store as demo_store

    paths = ProjectPaths.discover(Path(__file__), environ={})

    assert paths.project_root == t24_cli.REPO_ROOT
    assert paths.project_root == db_cli.REPO_ROOT
    assert demo_api.default_repo_root() == paths.project_root
    assert demo_api.default_static_dir(paths.project_root) == (
        paths.project_root / "apps" / "polymarket-backtester" / "dist"
    )
    assert (
        paths.db_root / "migrations" / "postgres" / "20260706_0009_demo_trading_backtester.sql"
    ) == demo_store.MIGRATION_PATH
