from __future__ import annotations

import csv
import hashlib
import zipfile
from pathlib import Path

import pytest

from scripts import manage_campaign_documentation as docs


def _write(root: Path, relative: str, content: str = "content\n") -> Path:
    path = root.joinpath(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _canonical_tree(root: Path) -> None:
    _write(root, "README.md")
    _write(root, "hkg-tmax/README.md")
    _write(root, "hkg-tmax/exp-0001/README.md")


def _snapshot(root: Path, output: Path) -> Path:
    return docs.write_snapshot(docs.inventory_documents(root), output, "a" * 40)


def _archive(path: Path, entries: list[tuple[str, bytes]]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return path


def test_default_campaigns_root_is_the_project_campaigns_directory() -> None:
    expected = (docs.PROJECT_ROOT / "experiments" / "campaigns").resolve()

    assert expected == docs.DEFAULT_CAMPAIGNS_ROOT
    assert docs.DEFAULT_CAMPAIGNS_ROOT.is_relative_to(docs.PROJECT_ROOT)


def test_inventory_hashes_and_classifies_markdown_and_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaigns"
    root.mkdir()
    root_readme = _write(root, "README.md", "root\n")
    _write(root, "notes.md")
    _write(root, "hkg-tmax/README.md")
    _write(root, "hkg-tmax/source.txt")
    _write(root, "hkg-tmax/exp-0001/README.md")
    _write(root, "hkg-tmax/exp-0001/results/RESULTS.md")
    _write(root, "hkg-tmax/exp-0001/evidence/input.txt")
    machine_file = _write(root, "hkg-tmax/exp-0001/results/large-machine-output.csv")
    original_is_file = Path.is_file

    def reject_machine_file_probe(path: Path) -> bool:
        if path == machine_file:
            raise AssertionError("inventory must not inspect unrelated machine files")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", reject_machine_file_probe)

    records = {record.original_path: record for record in docs.inventory_documents(root)}

    assert set(records) == {
        "README.md",
        "notes.md",
        "hkg-tmax/README.md",
        "hkg-tmax/source.txt",
        "hkg-tmax/exp-0001/README.md",
        "hkg-tmax/exp-0001/results/RESULTS.md",
        "hkg-tmax/exp-0001/evidence/input.txt",
    }
    assert records["README.md"].sha256 == hashlib.sha256(root_readme.read_bytes()).hexdigest()
    assert records["README.md"].bytes == root_readme.stat().st_size
    assert records["notes.md"].target_readme == "README.md"
    assert records["notes.md"].disposition == docs.DISPOSITION_MERGE_PRUNE
    assert records["hkg-tmax/source.txt"].target_readme == "hkg-tmax/README.md"
    assert records["hkg-tmax/source.txt"].disposition == docs.DISPOSITION_RETAIN_TEXT
    nested = records["hkg-tmax/exp-0001/results/RESULTS.md"]
    assert nested.target_readme == "hkg-tmax/exp-0001/README.md"


def test_snapshot_has_required_provenance_columns(tmp_path: Path) -> None:
    root = tmp_path / "campaigns"
    root.mkdir()
    _write(root, "README.md", "root\n")
    output = tmp_path / "snapshot.csv"

    docs.write_snapshot(docs.inventory_documents(root), output, "deadbeef")

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    root_bytes = (root / "README.md").read_bytes()
    assert tuple(rows[0]) == docs.SNAPSHOT_FIELDS
    assert rows == [
        {
            "source_commit": "deadbeef",
            "original_path": "README.md",
            "sha256": hashlib.sha256(root_bytes).hexdigest(),
            "bytes": str(len(root_bytes)),
            "target_readme": "README.md",
            "disposition": docs.DISPOSITION_RETAIN_README,
        }
    ]


def test_layout_requires_only_root_campaign_and_experiment_readmes(tmp_path: Path) -> None:
    root = tmp_path / "campaigns"
    root.mkdir()
    _canonical_tree(root)
    _write(root, "hkg-tmax/exp-0001/evidence.json")

    assert docs.check_layout(root).is_valid

    _write(root, "hkg-tmax/exp-0001/results/README.md")
    (root / "hkg-tmax" / "README.md").unlink()
    report = docs.check_layout(root)

    assert report.missing == ("hkg-tmax/README.md",)
    assert report.unexpected == ("hkg-tmax/exp-0001/results/README.md",)


def test_prune_dry_run_then_execute_removes_only_snapshotted_markdown_and_empty_dirs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaigns"
    root.mkdir()
    _canonical_tree(root)
    legacy = _write(root, "hkg-tmax/exp-0001/legacy/deep/RESULTS.md")
    retained_text = _write(root, "hkg-tmax/exp-0001/legacy_input.txt")
    retained_json = _write(root, "hkg-tmax/exp-0001/evidence/metrics.json")
    untouched_empty = root / "hkg-tmax" / "exp-0001" / "untouched-empty"
    untouched_empty.mkdir()
    snapshot = _snapshot(root, tmp_path / "snapshot.csv")

    dry_run = docs.prune_markdown(root, snapshot)

    assert dry_run == docs.PrunePlan(
        files=("hkg-tmax/exp-0001/legacy/deep/RESULTS.md",),
        directories=(
            "hkg-tmax/exp-0001/legacy/deep",
            "hkg-tmax/exp-0001/legacy",
        ),
        executed=False,
    )
    assert legacy.is_file()
    assert legacy.parent.is_dir()

    executed = docs.prune_markdown(root, snapshot, execute=True)

    assert executed.executed
    assert not legacy.exists()
    assert not (root / "hkg-tmax" / "exp-0001" / "legacy").exists()
    assert retained_text.is_file()
    assert retained_json.is_file()
    assert retained_json.parent.is_dir()
    assert untouched_empty.is_dir()
    assert docs.check_layout(root).is_valid


def test_prune_preflights_every_hash_before_deleting_any_file(tmp_path: Path) -> None:
    root = tmp_path / "campaigns"
    root.mkdir()
    _canonical_tree(root)
    first = _write(root, "hkg-tmax/exp-0001/FIRST.md", "first\n")
    changed = _write(root, "hkg-tmax/exp-0001/SECOND.md", "second\n")
    snapshot = _snapshot(root, tmp_path / "snapshot.csv")
    changed.write_text("changed\n", encoding="utf-8")

    with pytest.raises(docs.DocumentationGuardError, match="changed since snapshot"):
        docs.prune_markdown(root, snapshot, execute=True)

    assert first.is_file()
    assert changed.is_file()


def test_prune_refuses_unsnapshotted_markdown_and_missing_targets(tmp_path: Path) -> None:
    root = tmp_path / "campaigns"
    root.mkdir()
    _canonical_tree(root)
    snapshot = _snapshot(root, tmp_path / "snapshot.csv")
    unexpected = _write(root, "hkg-tmax/exp-0001/UNTRACKED.md")

    with pytest.raises(docs.DocumentationGuardError, match="absent from snapshot"):
        docs.prune_markdown(root, snapshot, execute=True)
    assert unexpected.is_file()

    unexpected.unlink()
    legacy = _write(root, "hkg-tmax/exp-0001/OLD.md")
    snapshot = _snapshot(root, tmp_path / "snapshot-2.csv")
    (root / "hkg-tmax" / "exp-0001" / "README.md").unlink()
    with pytest.raises(docs.DocumentationGuardError, match="canonical READMEs exist"):
        docs.prune_markdown(root, snapshot, execute=True)
    assert legacy.is_file()


def test_snapshot_rejects_paths_outside_the_supplied_root(tmp_path: Path) -> None:
    root = tmp_path / "campaigns"
    root.mkdir()
    _canonical_tree(root)
    outside = _write(tmp_path, "outside.md")
    snapshot = tmp_path / "malicious.csv"
    with snapshot.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=docs.SNAPSHOT_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "source_commit": "deadbeef",
                "original_path": "../outside.md",
                "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                "bytes": outside.stat().st_size,
                "target_readme": "README.md",
                "disposition": docs.DISPOSITION_MERGE_PRUNE,
            }
        )

    with pytest.raises(docs.DocumentationGuardError, match="invalid original_path"):
        docs.prune_markdown(root, snapshot, execute=True)

    assert outside.is_file()


def test_residual_evidence_text_has_explicit_retired_disposition(tmp_path: Path) -> None:
    root = tmp_path / "campaigns"
    root.mkdir()
    for relative_path in docs.MERGED_PRUNED_TEXT_PATHS:
        _write(root, relative_path)
    generic_text = _write(root, "residual-modeling/strategy/evidence/retained.txt")
    snapshot = _snapshot(root, tmp_path / "snapshot.csv")

    records = docs.read_snapshot(snapshot)

    for relative_path in docs.MERGED_PRUNED_TEXT_PATHS:
        assert records[relative_path].disposition == docs.DISPOSITION_MERGED_PRUNED_TEXT
    assert records["residual-modeling/strategy/evidence/retained.txt"].disposition == (
        docs.DISPOSITION_RETAIN_TEXT
    )
    assert generic_text.is_file()

    with snapshot.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    retired_row = next(row for row in rows if row["original_path"] in docs.MERGED_PRUNED_TEXT_PATHS)
    retired_row["disposition"] = docs.DISPOSITION_RETAIN_TEXT
    tampered = tmp_path / "tampered.csv"
    with tampered.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=docs.SNAPSHOT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(docs.DocumentationGuardError, match="incorrect disposition"):
        docs.read_snapshot(tampered)


def test_verify_archive_accepts_exact_snapshot_and_cli_prints_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "campaigns"
    root.mkdir()
    first = _write(root, "README.md", "root\n")
    second = _write(root, "hkg-tmax/source.txt", "source\n")
    snapshot = _snapshot(root, tmp_path / "snapshot.csv")
    archive = _archive(
        tmp_path / "docs.zip",
        [("README.md", first.read_bytes()), ("hkg-tmax/source.txt", second.read_bytes())],
    )

    assert docs.verify_archive(archive, snapshot) == 2
    assert (
        docs.main(
            [
                "verify-archive",
                "--campaigns-root",
                str(root),
                "--archive",
                str(archive),
                "--snapshot",
                str(snapshot),
            ]
        )
        == 0
    )
    assert "verified 2 archive entries" in capsys.readouterr().out


def test_verify_archive_rejects_missing_extra_duplicate_and_unsafe_entries(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaigns"
    root.mkdir()
    readme = _write(root, "README.md", "root\n")
    snapshot = _snapshot(root, tmp_path / "snapshot.csv")
    content = readme.read_bytes()

    missing = _archive(tmp_path / "missing.zip", [])
    with pytest.raises(docs.DocumentationGuardError, match="missing entries"):
        docs.verify_archive(missing, snapshot)

    extra = _archive(tmp_path / "extra.zip", [("README.md", content), ("EXTRA.md", b"extra")])
    with pytest.raises(docs.DocumentationGuardError, match="extra entries"):
        docs.verify_archive(extra, snapshot)

    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("README.md", content)
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("README.md", content)
    with pytest.raises(docs.DocumentationGuardError, match="duplicate entry"):
        docs.verify_archive(duplicate, snapshot)

    unsafe = _archive(
        tmp_path / "unsafe.zip", [("README.md", content), ("../escape.md", b"escape")]
    )
    with pytest.raises(docs.DocumentationGuardError, match="invalid archive entry path"):
        docs.verify_archive(unsafe, snapshot)


def test_verify_archive_rejects_byte_count_and_sha256_mismatches(tmp_path: Path) -> None:
    root = tmp_path / "campaigns"
    root.mkdir()
    readme = _write(root, "README.md", "root\n")
    snapshot = _snapshot(root, tmp_path / "snapshot.csv")
    original = readme.read_bytes()

    wrong_size = _archive(tmp_path / "wrong-size.zip", [("README.md", original + b"x")])
    with pytest.raises(docs.DocumentationGuardError, match="byte count"):
        docs.verify_archive(wrong_size, snapshot)

    changed = bytes([original[0] ^ 1]) + original[1:]
    wrong_hash = _archive(tmp_path / "wrong-hash.zip", [("README.md", changed)])
    with pytest.raises(docs.DocumentationGuardError, match="sha256"):
        docs.verify_archive(wrong_hash, snapshot)
