from hkg_tmax.manifest import build_manifest


def test_manifest_excludes_raw_data(repo_root, tmp_path) -> None:
    root = tmp_path
    (root / "file.txt").write_text("x")
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "secret.bin").write_bytes(b"y")
    manifest = build_manifest(root)
    paths = {item["path"] for item in manifest["files"]}
    assert "file.txt" in paths
    assert "data/raw/secret.bin" not in paths
