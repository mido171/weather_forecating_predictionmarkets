"""Inventory, snapshot, and safely prune campaign documentation.

The destructive operation is a dedicated ``prune`` subcommand and is a dry
run unless ``--execute`` is supplied.  It only removes Markdown files that are
both outside the canonical README layout and unchanged from a provenance
snapshot.  Non-Markdown files are never removed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import os
import re
import stat
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGNS_ROOT = (PROJECT_ROOT / "experiments" / "campaigns").resolve()
DEFAULT_MANIFEST_NAME = "DOCUMENT_PROVENANCE.csv"
SNAPSHOT_FIELDS = (
    "source_commit",
    "original_path",
    "sha256",
    "bytes",
    "target_readme",
    "disposition",
)
DISPOSITION_RETAIN_README = "retain_canonical_readme"
DISPOSITION_MERGE_PRUNE = "merge_then_prune"
DISPOSITION_RETAIN_TEXT = "retain_non_markdown_source"
DISPOSITION_MERGED_PRUNED_TEXT = "merged_then_pruned_text"
MERGED_PRUNED_TEXT_PATHS = frozenset(
    {
        "residual-modeling/strategy/gpt_pro_handoff_20260705_144937/evidence/commands_run.txt",
        "residual-modeling/strategy/gpt_pro_handoff_20260705_144937/evidence/git_diff_stat.txt",
        "residual-modeling/strategy/gpt_pro_handoff_20260705_144937/evidence/git_status_short.txt",
        "residual-modeling/strategy/gpt_pro_handoff_20260705_144937/evidence/tracked_name_status.txt",
    }
)
VALID_DISPOSITIONS = {
    DISPOSITION_RETAIN_README,
    DISPOSITION_MERGE_PRUNE,
    DISPOSITION_RETAIN_TEXT,
    DISPOSITION_MERGED_PRUNED_TEXT,
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class DocumentationGuardError(RuntimeError):
    """Raised when a campaign documentation operation cannot proceed safely."""


@dataclass(frozen=True)
class DocumentRecord:
    original_path: str
    sha256: str
    bytes: int
    target_readme: str
    disposition: str


@dataclass(frozen=True)
class LayoutReport:
    expected: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.missing and not self.unexpected


@dataclass(frozen=True)
class PrunePlan:
    files: tuple[str, ...]
    directories: tuple[str, ...]
    executed: bool


def resolve_campaigns_root(value: Path | str | None = None) -> Path:
    """Return an existing, resolved campaigns root."""

    candidate = DEFAULT_CAMPAIGNS_ROOT if value is None else Path(value).expanduser()
    try:
        root = candidate.resolve(strict=True)
    except OSError as exc:
        raise DocumentationGuardError(f"campaigns root is unavailable: {candidate}") from exc
    if not root.is_dir():
        raise DocumentationGuardError(f"campaigns root is not a directory: {root}")
    return root


def _relative_posix(root: Path, path: Path) -> str:
    try:
        absolute = Path(os.path.abspath(path))
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise DocumentationGuardError(f"path escapes campaigns root: {path}") from exc
    return relative.as_posix()


def _filesystem_path(path: Path) -> str:
    """Return a Windows long-path-safe absolute path for filesystem calls."""

    absolute = os.path.abspath(path)
    if os.name != "nt" or absolute.startswith("\\\\?\\"):
        return absolute
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def _lstat(path: Path) -> os.stat_result:
    return os.stat(_filesystem_path(path), follow_symlinks=False)


def _is_link_or_reparse(status: os.stat_result) -> bool:
    attributes = getattr(status, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(status.st_mode) or bool(attributes & reparse_flag)


def _walk_tree(root: Path) -> tuple[list[Path], list[Path]]:
    """Walk without following links and reject paths that resolve outside root."""

    directories: list[Path] = []
    files: list[Path] = []
    pending = [root]
    while pending:
        current = pending.pop()
        _relative_posix(root, current)
        with os.scandir(_filesystem_path(current)) as scanner:
            entries = sorted(scanner, key=lambda entry: entry.name)
        child_directories: list[Path] = []
        for entry in entries:
            path = current / entry.name
            status = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(status.st_mode):
                if _is_link_or_reparse(status):
                    raise DocumentationGuardError(f"directory links are not supported: {path}")
                _relative_posix(root, path)
                directories.append(path)
                child_directories.append(path)
                continue
            if path.suffix.lower() not in {".md", ".txt"}:
                continue
            if _is_link_or_reparse(status):
                raise DocumentationGuardError(f"file links are not supported: {path}")
            if not stat.S_ISREG(status.st_mode):
                raise DocumentationGuardError(f"non-regular files are not supported: {path}")
            _relative_posix(root, path)
            files.append(path)
        pending.extend(reversed(child_directories))
    return directories, files


def _fingerprint(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with open(_filesystem_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def target_readme_for(original_path: str) -> str:
    """Map a document to the root, campaign, or top-level experiment README."""

    relative = _validated_manifest_path(original_path, field="original_path")
    parts = relative.parts
    if len(parts) == 1:
        return "README.md"
    if len(parts) == 2:
        return f"{parts[0]}/README.md"
    return f"{parts[0]}/{parts[1]}/README.md"


def _disposition(original_path: str, target_readme: str) -> str:
    if original_path in MERGED_PRUNED_TEXT_PATHS:
        return DISPOSITION_MERGED_PRUNED_TEXT
    suffix = PurePosixPath(original_path).suffix.lower()
    if suffix == ".txt":
        return DISPOSITION_RETAIN_TEXT
    if original_path == target_readme:
        return DISPOSITION_RETAIN_README
    return DISPOSITION_MERGE_PRUNE


def inventory_documents(campaigns_root: Path | str | None = None) -> list[DocumentRecord]:
    """Inventory every Markdown and text document beneath ``campaigns_root``."""

    root = resolve_campaigns_root(campaigns_root)
    _, files = _walk_tree(root)
    records: list[DocumentRecord] = []
    for path in files:
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        original_path = _relative_posix(root, path)
        target_readme = target_readme_for(original_path)
        sha256, byte_count = _fingerprint(path)
        records.append(
            DocumentRecord(
                original_path=original_path,
                sha256=sha256,
                bytes=byte_count,
                target_readme=target_readme,
                disposition=_disposition(original_path, target_readme),
            )
        )
    return sorted(records, key=lambda record: record.original_path)


def write_snapshot(records: list[DocumentRecord], output: Path | str, source_commit: str) -> Path:
    """Atomically write a provenance CSV for an inventory."""

    if not source_commit.strip():
        raise DocumentationGuardError("source_commit must not be empty")
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=SNAPSHOT_FIELDS)
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "source_commit": source_commit,
                        "original_path": record.original_path,
                        "sha256": record.sha256,
                        "bytes": record.bytes,
                        "target_readme": record.target_readme,
                        "disposition": record.disposition,
                    }
                )
        os.replace(temporary_name, output_path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return output_path


def _validated_manifest_path(value: str, *, field: str) -> PurePosixPath:
    if not value or "\\" in value:
        raise DocumentationGuardError(f"invalid {field}: {value!r}")
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in {"", ".", ".."} for part in posix_path.parts)
        or posix_path.as_posix() != value
    ):
        raise DocumentationGuardError(f"invalid {field}: {value!r}")
    return posix_path


def read_snapshot(path: Path | str) -> dict[str, DocumentRecord]:
    """Read and strictly validate a provenance snapshot."""

    manifest_path = Path(path).expanduser().resolve(strict=True)
    records: dict[str, DocumentRecord] = {}
    normalized_paths: set[str] = set()
    with open(_filesystem_path(manifest_path), encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_fields = set(SNAPSHOT_FIELDS).difference(reader.fieldnames or ())
        if missing_fields:
            raise DocumentationGuardError(
                "snapshot is missing columns: " + ", ".join(sorted(missing_fields))
            )
        for line_number, row in enumerate(reader, start=2):
            original_path = str(row["original_path"])
            _validated_manifest_path(original_path, field="original_path")
            if PurePosixPath(original_path).suffix.lower() not in {".md", ".txt"}:
                raise DocumentationGuardError(
                    f"snapshot line {line_number} is not a Markdown or text document"
                )
            target_readme = str(row["target_readme"])
            _validated_manifest_path(target_readme, field="target_readme")
            if target_readme != target_readme_for(original_path):
                raise DocumentationGuardError(
                    f"snapshot line {line_number} has an incorrect target_readme"
                )
            sha256 = str(row["sha256"]).lower()
            if SHA256_RE.fullmatch(sha256) is None:
                raise DocumentationGuardError(f"snapshot line {line_number} has invalid sha256")
            try:
                byte_count = int(str(row["bytes"]))
            except ValueError as exc:
                raise DocumentationGuardError(
                    f"snapshot line {line_number} has invalid bytes"
                ) from exc
            if byte_count < 0:
                raise DocumentationGuardError(f"snapshot line {line_number} has invalid bytes")
            disposition = str(row["disposition"])
            expected_disposition = _disposition(original_path, target_readme)
            if disposition not in VALID_DISPOSITIONS or disposition != expected_disposition:
                raise DocumentationGuardError(
                    f"snapshot line {line_number} has an incorrect disposition"
                )
            if not str(row["source_commit"]).strip():
                raise DocumentationGuardError(
                    f"snapshot line {line_number} has an empty source_commit"
                )
            if original_path in records:
                raise DocumentationGuardError(f"snapshot contains duplicate path: {original_path}")
            normalized_path = original_path.casefold()
            if normalized_path in normalized_paths:
                raise DocumentationGuardError(
                    f"snapshot contains case-colliding path: {original_path}"
                )
            normalized_paths.add(normalized_path)
            records[original_path] = DocumentRecord(
                original_path=original_path,
                sha256=sha256,
                bytes=byte_count,
                target_readme=target_readme,
                disposition=disposition,
            )
    return records


def verify_archive(archive_path: Path | str, snapshot_path: Path | str) -> int:
    """Verify that a ZIP is an exact, byte-identical copy of a provenance snapshot."""

    snapshot = read_snapshot(snapshot_path)
    resolved_archive = Path(archive_path).expanduser().resolve(strict=True)
    try:
        with zipfile.ZipFile(_filesystem_path(resolved_archive), "r") as archive:
            archive_entries: dict[str, zipfile.ZipInfo] = {}
            normalized_paths: set[str] = set()
            for info in archive.infolist():
                entry_path = info.filename
                _validated_manifest_path(entry_path, field="archive entry path")
                normalized_path = entry_path.casefold()
                if entry_path in archive_entries or normalized_path in normalized_paths:
                    raise DocumentationGuardError(f"archive contains duplicate entry: {entry_path}")
                archive_entries[entry_path] = info
                normalized_paths.add(normalized_path)

            missing = sorted(set(snapshot).difference(archive_entries))
            extra = sorted(set(archive_entries).difference(snapshot))
            if missing or extra:
                problems: list[str] = []
                if missing:
                    problems.append("missing entries: " + ", ".join(missing))
                if extra:
                    problems.append("extra entries: " + ", ".join(extra))
                raise DocumentationGuardError(
                    "archive does not match snapshot; " + "; ".join(problems)
                )

            for entry_path in sorted(archive_entries):
                info = archive_entries[entry_path]
                record = snapshot[entry_path]
                unix_file_type = stat.S_IFMT(info.external_attr >> 16)
                if info.is_dir() or unix_file_type not in {0, stat.S_IFREG}:
                    raise DocumentationGuardError(
                        f"archive entry is not a regular file: {entry_path}"
                    )
                if info.flag_bits & 0x1:
                    raise DocumentationGuardError(f"archive entry is encrypted: {entry_path}")
                if info.file_size != record.bytes:
                    raise DocumentationGuardError(
                        f"archive entry byte count differs from snapshot: {entry_path}"
                    )
                digest = hashlib.sha256()
                byte_count = 0
                with archive.open(info, "r") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                        byte_count += len(chunk)
                if byte_count != record.bytes:
                    raise DocumentationGuardError(
                        f"archive entry byte count differs from snapshot: {entry_path}"
                    )
                if not hmac.compare_digest(digest.hexdigest(), record.sha256):
                    raise DocumentationGuardError(
                        f"archive entry sha256 differs from snapshot: {entry_path}"
                    )
    except DocumentationGuardError:
        raise
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError, OSError) as exc:
        raise DocumentationGuardError(f"archive cannot be verified: {resolved_archive}") from exc
    return len(snapshot)


def expected_readmes(campaigns_root: Path | str | None = None) -> set[str]:
    """Return the only Markdown paths allowed by the consolidated layout."""

    root = resolve_campaigns_root(campaigns_root)
    directories, _ = _walk_tree(root)
    direct_children = sorted(path for path in directories if path.parent == root)
    expected = {"README.md"}
    for campaign in direct_children:
        campaign_path = _relative_posix(root, campaign)
        expected.add(f"{campaign_path}/README.md")
        for experiment in sorted(path for path in directories if path.parent == campaign):
            experiment_path = _relative_posix(root, experiment)
            expected.add(f"{experiment_path}/README.md")
    return expected


def check_layout(campaigns_root: Path | str | None = None) -> LayoutReport:
    """Check that canonical READMEs are the only Markdown documents present."""

    root = resolve_campaigns_root(campaigns_root)
    expected = expected_readmes(root)
    _, files = _walk_tree(root)
    actual = {_relative_posix(root, path) for path in files if path.suffix.lower() == ".md"}
    return LayoutReport(
        expected=tuple(sorted(expected)),
        missing=tuple(sorted(expected - actual)),
        unexpected=tuple(sorted(actual - expected)),
    )


def _planned_empty_directories(root: Path, files_to_delete: set[Path]) -> list[Path]:
    directories, _ = _walk_tree(root)
    candidates: set[Path] = set()
    for path in files_to_delete:
        parent = path.parent
        while parent != root:
            candidates.add(parent)
            parent = parent.parent
    removable: set[Path] = set()
    for directory in sorted(
        (path for path in directories if path in candidates),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        with os.scandir(_filesystem_path(directory)) as scanner:
            entries = [directory / entry.name for entry in scanner]
        if not entries or all(entry in files_to_delete or entry in removable for entry in entries):
            removable.add(directory)
    return sorted(removable, key=lambda path: (-len(path.parts), path.as_posix()))


def prune_markdown(
    campaigns_root: Path | str,
    snapshot_path: Path | str,
    *,
    execute: bool = False,
) -> PrunePlan:
    """Delete unchanged, snapshotted non-canonical Markdown and empty folders."""

    root = resolve_campaigns_root(campaigns_root)
    snapshot = read_snapshot(snapshot_path)
    report = check_layout(root)
    if report.missing:
        raise DocumentationGuardError(
            "refusing to prune before canonical READMEs exist: " + ", ".join(report.missing)
        )

    files_to_delete: list[Path] = []
    for relative_path in report.unexpected:
        path = root.joinpath(*PurePosixPath(relative_path).parts)
        _relative_posix(root, path)
        if path.suffix.lower() != ".md":
            raise DocumentationGuardError(f"refusing to prune non-Markdown file: {relative_path}")
        record = snapshot.get(relative_path)
        if record is None or record.disposition != DISPOSITION_MERGE_PRUNE:
            raise DocumentationGuardError(
                f"refusing to prune Markdown absent from snapshot: {relative_path}"
            )
        sha256, byte_count = _fingerprint(path)
        if byte_count != record.bytes or not hmac.compare_digest(sha256, record.sha256):
            raise DocumentationGuardError(
                f"refusing to prune Markdown changed since snapshot: {relative_path}"
            )
        target = root.joinpath(*PurePosixPath(record.target_readme).parts)
        target_status = _lstat(target)
        if _is_link_or_reparse(target_status) or not stat.S_ISREG(target_status.st_mode):
            raise DocumentationGuardError(
                f"refusing to prune without target README: {record.target_readme}"
            )
        files_to_delete.append(path)

    empty_directories = _planned_empty_directories(root, set(files_to_delete))
    file_names = tuple(_relative_posix(root, path) for path in files_to_delete)
    directory_names = tuple(_relative_posix(root, path) for path in empty_directories)
    if execute:
        for path in files_to_delete:
            os.unlink(_filesystem_path(path))
        for directory in empty_directories:
            os.rmdir(_filesystem_path(directory))
    return PrunePlan(files=file_names, directories=directory_names, executed=execute)


def _source_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_inventory(records: list[DocumentRecord], stream: TextIO) -> None:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("original_path", "sha256", "bytes", "target_readme", "disposition"))
    for record in records:
        writer.writerow(
            (
                record.original_path,
                record.sha256,
                record.bytes,
                record.target_readme,
                record.disposition,
            )
        )


def _add_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--campaigns-root",
        type=Path,
        default=None,
        help=f"campaigns directory (default: {DEFAULT_CAMPAIGNS_ROOT})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inventory_parser = commands.add_parser("inventory", help="print the document inventory")
    _add_root_argument(inventory_parser)

    snapshot_parser = commands.add_parser("snapshot", help="write a provenance CSV")
    _add_root_argument(snapshot_parser)
    snapshot_parser.add_argument("--output", type=Path, default=None)
    snapshot_parser.add_argument("--source-commit", default=None)

    check_parser = commands.add_parser("check", help="validate the consolidated layout")
    _add_root_argument(check_parser)

    prune_parser = commands.add_parser("prune", help="remove snapshotted legacy Markdown")
    _add_root_argument(prune_parser)
    prune_parser.add_argument("--snapshot", type=Path, default=None)
    mode = prune_parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="perform the deletions")
    mode.add_argument("--dry-run", action="store_true", help="report only (the default)")

    verify_parser = commands.add_parser(
        "verify-archive", help="verify a documentation ZIP against provenance"
    )
    _add_root_argument(verify_parser)
    verify_parser.add_argument("--archive", type=Path, required=True)
    verify_parser.add_argument("--snapshot", "--manifest", dest="snapshot", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = resolve_campaigns_root(args.campaigns_root)
        if args.command == "inventory":
            _write_inventory(inventory_documents(root), sys.stdout)
            return 0
        if args.command == "snapshot":
            output = args.output or (root / DEFAULT_MANIFEST_NAME)
            source_commit = args.source_commit or _source_commit()
            written = write_snapshot(inventory_documents(root), output, source_commit)
            print(f"wrote {written}")
            return 0
        if args.command == "check":
            report = check_layout(root)
            for path in report.missing:
                print(f"missing: {path}")
            for path in report.unexpected:
                print(f"unexpected: {path}")
            if report.is_valid:
                print(f"layout ok: {len(report.expected)} canonical README files")
                return 0
            return 1
        if args.command == "prune":
            snapshot_path = args.snapshot or (root / DEFAULT_MANIFEST_NAME)
            plan = prune_markdown(root, snapshot_path, execute=args.execute)
            action = "deleted" if plan.executed else "would delete"
            for path in plan.files:
                print(f"{action}: {path}")
            directory_action = "removed directory" if plan.executed else "would remove directory"
            for path in plan.directories:
                print(f"{directory_action}: {path}")
            print(
                f"{'executed' if plan.executed else 'dry-run'}: "
                f"{len(plan.files)} files, {len(plan.directories)} directories"
            )
            return 0
        if args.command == "verify-archive":
            snapshot_path = args.snapshot or (root / DEFAULT_MANIFEST_NAME)
            verified_count = verify_archive(args.archive, snapshot_path)
            print(f"verified {verified_count} archive entries")
            return 0
        raise AssertionError(f"unhandled command: {args.command}")
    except (DocumentationGuardError, OSError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
