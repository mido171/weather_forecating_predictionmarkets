from __future__ import annotations

import importlib
import json
import os
import re
import secrets
import shutil
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ConfigError, load_yaml
from .experiment_index import generate_index
from .experiment_registry import (
    CAMPAIGN_TITLES,
    EXPERIMENT_CONTROL_DIR,
    EXPERIMENT_TEMPLATE_DIR,
    ExperimentError,
    _is_reparse_path,
    _require_resolved_within,
    _require_title,
    _validate_rendered_experiment,
    _write_text_atomic,
    _write_yaml_atomic,
    require_campaign,
    slugify,
    validate_registry_state,
)

_PLACEHOLDER_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
_TRANSACTION_TOKEN_RE = re.compile(r"[0-9a-f]{32}")
_TRANSACTION_MARKER = ".experiment-transaction.json"
_CAMPAIGN_TRANSACTION_MARKER = ".experiment-campaign-transaction.json"
_TRANSACTION_PHASES = {
    "prepared",
    "promoting",
    "promoted",
    "registry_committing",
    "registry_committed",
    "index_committing",
    "index_committed",
}
_MAX_TEMPLATE_TREE_ENTRIES = 512
_MAX_TEMPLATE_TREE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class _CampaignProvision:
    path: Path
    campaigns_directory_created: bool
    campaign_directory_created: bool
    readme_created: bool
    marker_created: bool


@dataclass(frozen=True)
class _JournalPaths:
    destination: Path
    staging: Path
    campaign_root: Path


def _ensure_campaign_root(
    experiments_root: Path,
    campaign: str,
    journal: Mapping[str, Any],
) -> _CampaignProvision:
    campaigns_root = experiments_root / "campaigns"
    campaigns_directory_created = not campaigns_root.exists()
    if campaigns_root.exists() and not campaigns_root.is_dir():
        raise ExperimentError(f"Campaigns path is not a directory: {campaigns_root}")
    if campaigns_root.exists() and _is_reparse_path(campaigns_root):
        raise ExperimentError(f"Campaigns path must not be a reparse point: {campaigns_root}")
    campaigns_root.mkdir(parents=True, exist_ok=True)
    _require_resolved_within(campaigns_root, experiments_root, "Campaigns directory")

    campaign_root = experiments_root / "campaigns" / campaign
    campaign_directory_created = not campaign_root.exists()
    if campaign_root.exists() and not campaign_root.is_dir():
        raise ExperimentError(f"Campaign path is not a directory: {campaign_root}")
    if campaign_root.exists() and _is_reparse_path(campaign_root):
        raise ExperimentError(f"Campaign path must not be a reparse point: {campaign_root}")
    campaign_root.mkdir(parents=True, exist_ok=True)
    if campaign_root.resolve() != campaigns_root.resolve() / campaign:
        raise ExperimentError(
            f"Campaign path resolves outside its canonical location: {campaign_root}"
        )

    readme = campaign_root / "README.md"
    if readme.is_file():
        if readme.is_symlink():
            raise ExperimentError(f"Campaign README must not be a symlink: {readme}")
        campaign_marker = _campaign_transaction_marker_path(campaign_root)
        _write_marker(campaign_marker, journal, scope="campaign-transaction")
        return _CampaignProvision(
            campaign_root,
            campaigns_directory_created,
            campaign_directory_created,
            False,
            True,
        )
    unexpected = sorted(path.name for path in campaign_root.iterdir())
    if unexpected:
        raise ExperimentError(f"Campaign is missing README.md and is not empty: {campaign_root}")
    campaign_marker = _campaign_transaction_marker_path(campaign_root)
    try:
        _write_marker(campaign_marker, journal, scope="campaign-transaction")
    except BaseException:
        if campaign_directory_created:
            with suppress(OSError):
                campaign_root.rmdir()
        if campaigns_directory_created:
            with suppress(OSError):
                campaigns_root.rmdir()
        raise
    try:
        readme.write_text(
            f"# {CAMPAIGN_TITLES[campaign]}\n\n"
            "This campaign contains governed experiments created from the standard "
            "template. Add each experiment to this index after its hypothesis is "
            "predeclared.\n",
            encoding="utf-8",
        )
    except OSError:
        with suppress(OSError):
            readme.unlink(missing_ok=True)
        with suppress(OSError):
            campaign_marker.unlink(missing_ok=True)
        if campaign_directory_created:
            with suppress(OSError):
                campaign_root.rmdir()
        if campaigns_directory_created:
            with suppress(OSError):
                campaigns_root.rmdir()
        raise
    return _CampaignProvision(
        campaign_root,
        campaigns_directory_created,
        campaign_directory_created,
        True,
        True,
    )


def _try_lock_descriptor(descriptor: int) -> bool:
    os.lseek(descriptor, 0, os.SEEK_SET)
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    fcntl = importlib.import_module("fcntl")
    fcntl_api = vars(fcntl)
    flock = fcntl_api["flock"]

    try:
        flock(descriptor, fcntl_api["LOCK_EX"] | fcntl_api["LOCK_NB"])
    except BlockingIOError:
        return False
    return True


def _unlock_descriptor(descriptor: int) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return

    fcntl = importlib.import_module("fcntl")
    fcntl_api = vars(fcntl)
    flock = fcntl_api["flock"]

    flock(descriptor, fcntl_api["LOCK_UN"])


@contextmanager
def _registry_lock(lock_path: Path, timeout_seconds: float = 15.0) -> Iterator[None]:
    """Serialize creators with an OS lock that is released automatically on crash."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    try:
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
        deadline = time.monotonic() + timeout_seconds
        while not acquired:
            acquired = _try_lock_descriptor(descriptor)
            if acquired:
                break
            if time.monotonic() >= deadline:
                raise ExperimentError(f"Timed out waiting for registry lock: {lock_path}")
            time.sleep(0.1)
        yield
    finally:
        if acquired:
            _unlock_descriptor(descriptor)
        os.close(descriptor)


def _replace_placeholders(path: Path, values: dict[str, str]) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        try:
            return values[key]
        except KeyError as exc:
            raise ExperimentError(f"Unknown template placeholder {key!r} in {path}") from exc

    path.write_text(_PLACEHOLDER_RE.sub(replace, text), encoding="utf-8")


def _require_regular_path_chain(boundary: Path, path: Path, label: str) -> None:
    _require_resolved_within(path, boundary, label)
    try:
        relative_parts = path.relative_to(boundary).parts
    except ValueError as exc:
        raise ExperimentError(f"{label} escapes its governed root: {path}") from exc
    current = boundary
    for part in (None, *relative_parts):
        if part is not None:
            current /= part
        if current.is_symlink() or not current.exists() or _is_reparse_path(current):
            raise ExperimentError(f"{label} crosses an unavailable reparse path: {current}")


def _regular_tree_files(root: Path, label: str) -> list[Path]:
    """List a bounded tree without following links or Windows reparse points."""

    if root.is_symlink() or not root.is_dir() or _is_reparse_path(root):
        raise ExperimentError(f"{label} must be a regular directory: {root}")
    files: list[Path] = []
    pending = [root]
    entry_count = 0
    total_bytes = 0
    while pending:
        directory = pending.pop()
        try:
            entries = os.scandir(directory)
        except OSError as exc:
            raise ExperimentError(f"Could not inspect {label}: {directory}") from exc
        with entries:
            for entry in entries:
                entry_count += 1
                if entry_count > _MAX_TEMPLATE_TREE_ENTRIES:
                    raise ExperimentError(
                        f"{label} exceeds {_MAX_TEMPLATE_TREE_ENTRIES} bounded entries"
                    )
                path = Path(entry.path)
                if entry.is_symlink() or _is_reparse_path(path):
                    raise ExperimentError(f"{label} contains a symlink or reparse point: {path}")
                try:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                    elif entry.is_file(follow_symlinks=False):
                        total_bytes += entry.stat(follow_symlinks=False).st_size
                        if total_bytes > _MAX_TEMPLATE_TREE_BYTES:
                            raise ExperimentError(
                                f"{label} exceeds {_MAX_TEMPLATE_TREE_BYTES} bounded bytes"
                            )
                        files.append(path)
                    else:
                        raise ExperimentError(f"{label} contains a non-file entry: {path}")
                except OSError as exc:
                    raise ExperimentError(f"Could not inspect {label} entry: {path}") from exc
    return sorted(files)


def _write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    _write_text_atomic(
        path,
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _journal_parts(value: object, label: str) -> list[str]:
    if not isinstance(value, str) or "\\" in value:
        raise ExperimentError(f"Experiment transaction journal has invalid {label}")
    parts = value.split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ExperimentError(f"Experiment transaction journal has unsafe {label}")
    return parts


def _journal_path(root: Path, parts: list[str], label: str) -> Path:
    path = root.joinpath(*parts)
    _require_resolved_within(path, root, f"Transaction journal {label}")
    current = root
    for part in parts:
        current /= part
        if current.is_symlink() or (current.exists() and _is_reparse_path(current)):
            raise ExperimentError(f"Transaction journal {label} crosses a reparse point: {current}")
    return path


def _require_local_runtime_paths(root: Path) -> None:
    for path in (root / "var", root / "var" / "run", root / "var" / "tmp"):
        if not path.exists():
            continue
        _require_resolved_within(path, root, "Experiment runtime path")
        if _is_reparse_path(path):
            raise ExperimentError(f"Experiment runtime path must not be a reparse point: {path}")


def _require_runtime_control_file(root: Path, path: Path, label: str) -> None:
    """Reject links, reparse points, and non-files before opening runtime controls."""

    _require_resolved_within(path, root, label)
    if path.is_symlink():
        raise ExperimentError(f"{label} must not be a symlink or reparse point: {path}")
    if not path.exists():
        return
    if not path.is_file() or _is_reparse_path(path):
        raise ExperimentError(f"{label} must be a regular non-reparse file: {path}")


def _load_transaction_journal(path: Path) -> dict[str, Any]:
    if _is_reparse_path(path):
        raise ExperimentError(f"Experiment transaction journal must be a regular file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"Invalid experiment transaction journal: {path}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ExperimentError(f"Unsupported experiment transaction journal: {path}")
    if not isinstance(data.get("original_registry"), dict) or not isinstance(
        data.get("updated_registry"), dict
    ):
        raise ExperimentError(f"Experiment transaction journal lacks registry snapshots: {path}")
    return data


def _validate_transaction_journal(root: Path, journal: Mapping[str, Any]) -> _JournalPaths:
    phase = journal.get("phase")
    token = journal.get("transaction_token")
    experiment_id = journal.get("experiment_id")
    if phase not in _TRANSACTION_PHASES:
        raise ExperimentError("Experiment transaction journal has invalid phase")
    if not isinstance(token, str) or _TRANSACTION_TOKEN_RE.fullmatch(token) is None:
        raise ExperimentError("Experiment transaction journal has invalid ownership token")
    if not isinstance(experiment_id, str):
        raise ExperimentError("Experiment transaction journal has invalid experiment_id")
    for field in (
        "campaigns_root_preexisting",
        "campaign_root_preexisting",
        "campaign_readme_preexisting",
    ):
        if not isinstance(journal.get(field), bool):
            raise ExperimentError(f"Experiment transaction journal has invalid {field}")
    if not journal["campaigns_root_preexisting"] and journal["campaign_root_preexisting"]:
        raise ExperimentError("Experiment transaction journal has inconsistent campaign roots")
    if not journal["campaign_root_preexisting"] and journal["campaign_readme_preexisting"]:
        raise ExperimentError("Experiment transaction journal has inconsistent campaign README")
    original_index = journal.get("original_index")
    if original_index is not None and not isinstance(original_index, str):
        raise ExperimentError("Experiment transaction journal has invalid index snapshot")

    destination_parts = _journal_parts(journal.get("destination"), "destination")
    if len(destination_parts) != 4 or destination_parts[:2] != [
        "experiments",
        "campaigns",
    ]:
        raise ExperimentError(
            "Experiment transaction destination must be experiments/campaigns/{campaign}/{id-slug}"
        )
    campaign = require_campaign(destination_parts[2])
    directory_name = destination_parts[3]
    prefix = f"{experiment_id}-"
    if not directory_name.startswith(prefix):
        raise ExperimentError("Experiment transaction destination does not match experiment_id")
    slug = directory_name.removeprefix(prefix)
    if not slug or slugify(slug) != slug:
        raise ExperimentError("Experiment transaction destination has a noncanonical slug")

    campaign_parts = _journal_parts(journal.get("campaign_root"), "campaign_root")
    if campaign_parts != destination_parts[:3]:
        raise ExperimentError("Experiment transaction campaign_root is not the destination parent")
    staging_parts = _journal_parts(journal.get("staging"), "staging")
    expected_staging_name = f"{experiment_id}-{token}"
    if staging_parts != ["var", "tmp", "experiment-creation", expected_staging_name]:
        raise ExperimentError("Experiment transaction staging must be the exact owned var/tmp path")

    original_registry = journal["original_registry"]
    updated_registry = journal["updated_registry"]
    original_entries = original_registry.get("experiments")
    updated_entries = updated_registry.get("experiments")
    original_next_id = original_registry.get("next_id")
    updated_next_id = updated_registry.get("next_id")
    if (
        original_registry.get("registry_version") != 2
        or updated_registry.get("registry_version") != 2
        or not isinstance(original_entries, list)
        or not isinstance(updated_entries, list)
        or isinstance(original_next_id, bool)
        or not isinstance(original_next_id, int)
        or original_next_id < 1
        or updated_next_id != original_next_id + 1
        or len(updated_entries) != len(original_entries) + 1
        or updated_entries[:-1] != original_entries
    ):
        raise ExperimentError("Experiment transaction journal has an invalid registry transition")
    expected_id = f"EXP-{original_next_id:04d}"
    if experiment_id != expected_id:
        raise ExperimentError("Experiment transaction experiment_id does not match next_id")
    appended = updated_entries[-1]
    expected_directory = "/".join(destination_parts[1:])
    if (
        not isinstance(appended, Mapping)
        or appended.get("id") != experiment_id
        or appended.get("campaign") != campaign
        or appended.get("directory") != expected_directory
        or appended.get("status") != "PLANNED"
        or not isinstance(appended.get("created_at_utc"), str)
        or not appended["created_at_utc"].strip()
    ):
        raise ExperimentError(
            "Experiment transaction appended registry entry does not match its owned paths"
        )
    title = appended.get("title")
    if _require_title(title) != title:
        raise ExperimentError("Experiment transaction appended registry title is noncanonical")
    if slug != slugify(title):
        raise ExperimentError("Experiment transaction destination slug does not match its title")

    return _JournalPaths(
        destination=_journal_path(root, destination_parts, "destination"),
        staging=_journal_path(root, staging_parts, "staging"),
        campaign_root=_journal_path(root, campaign_parts, "campaign_root"),
    )


def _transaction_marker_path(directory: Path) -> Path:
    return directory / _TRANSACTION_MARKER


def _campaign_transaction_marker_path(directory: Path) -> Path:
    return directory / _CAMPAIGN_TRANSACTION_MARKER


def _marker_payload(journal: Mapping[str, Any], scope: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scope": scope,
        "transaction_token": journal["transaction_token"],
        "experiment_id": journal["experiment_id"],
        "destination": journal["destination"],
    }


def _marker_pending_path(path: Path, journal: Mapping[str, Any]) -> Path:
    return path.with_name(f"{path.name}.{journal['transaction_token']}.pending")


def _write_marker(path: Path, journal: Mapping[str, Any], scope: str) -> None:
    if path.exists() or path.is_symlink():
        raise ExperimentError(f"Transaction ownership marker already exists: {path}")
    pending = _marker_pending_path(path, journal)
    if pending.exists() or pending.is_symlink():
        raise ExperimentError(f"Transaction ownership marker publication exists: {pending}")
    payload = json.dumps(
        _marker_payload(journal, scope),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    try:
        with pending.open("x", encoding="utf-8") as marker_file:
            marker_file.write(payload + "\n")
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _write_transaction_marker(directory: Path, journal: Mapping[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    try:
        _write_marker(_transaction_marker_path(directory), journal, scope="experiment-directory")
    except BaseException:
        with suppress(OSError):
            directory.rmdir()
        raise


def _require_transaction_marker(directory: Path, journal: Mapping[str, Any]) -> Path:
    if not directory.is_dir() or _is_reparse_path(directory):
        raise ExperimentError(f"Transaction-owned directory is unavailable: {directory}")
    marker_path = _transaction_marker_path(directory)
    if not marker_path.is_file() or _is_reparse_path(marker_path):
        raise ExperimentError(f"Transaction ownership marker is missing: {marker_path}")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"Invalid transaction ownership marker: {marker_path}") from exc
    expected = _marker_payload(journal, "experiment-directory")
    if marker != expected:
        raise ExperimentError(f"Transaction ownership marker does not match journal: {marker_path}")
    return marker_path


def _require_campaign_transaction_marker(
    directory: Path,
    journal: Mapping[str, Any],
) -> Path:
    if not directory.is_dir() or _is_reparse_path(directory):
        raise ExperimentError(f"Transaction-owned campaign directory is unavailable: {directory}")
    marker_path = _campaign_transaction_marker_path(directory)
    if not marker_path.is_file() or _is_reparse_path(marker_path):
        raise ExperimentError(f"Campaign transaction ownership marker is missing: {marker_path}")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"Invalid campaign transaction marker: {marker_path}") from exc
    if marker != _marker_payload(journal, "campaign-transaction"):
        raise ExperimentError(f"Campaign transaction marker does not match journal: {marker_path}")
    return marker_path


def _contains_only_incomplete_marker(
    directory: Path,
    marker_path: Path,
    journal: Mapping[str, Any],
) -> bool:
    """Recognize the two tiny crash windows before a marker is fully published."""

    if directory.is_symlink() or not directory.is_dir() or _is_reparse_path(directory):
        return False
    entries = list(directory.iterdir())
    if not entries:
        return True
    pending = _marker_pending_path(marker_path, journal)
    if len(entries) != 1 or entries[0] != pending:
        return False
    candidate = entries[0]
    return not candidate.is_symlink() and candidate.is_file() and not _is_reparse_path(candidate)


def _require_experiment_directory_ownership(
    directory: Path,
    journal: Mapping[str, Any],
    *,
    allow_prepared_marker_gap: bool,
    allow_recovery_pending: bool,
) -> Path | None:
    if directory.is_symlink() or not directory.is_dir() or _is_reparse_path(directory):
        raise ExperimentError(f"Transaction-owned directory is unavailable: {directory}")
    marker_path = _transaction_marker_path(directory)
    pending_path = _marker_pending_path(marker_path, journal)
    try:
        return _require_transaction_marker(directory, journal)
    except ExperimentError:
        if marker_path.exists() or marker_path.is_symlink():
            raise
        if allow_recovery_pending and (pending_path.exists() or pending_path.is_symlink()):
            if (
                pending_path.is_symlink()
                or not pending_path.is_file()
                or _is_reparse_path(pending_path)
            ):
                raise
            return pending_path
        if (
            allow_prepared_marker_gap
            and journal["phase"] == "prepared"
            and _contains_only_incomplete_marker(directory, marker_path, journal)
        ):
            return None
        raise


def _directory_identity(directory: Path) -> tuple[int, int]:
    if directory.is_symlink() or not directory.is_dir() or _is_reparse_path(directory):
        raise ExperimentError(f"Transaction-owned directory is unavailable: {directory}")
    stat_result = directory.stat(follow_symlinks=False)
    if stat_result.st_ino == 0:
        raise ExperimentError(f"Transaction-owned directory has no stable identity: {directory}")
    return stat_result.st_dev, stat_result.st_ino


def _same_directory_identity(directory: Path, identity: tuple[int, int]) -> bool:
    try:
        return _directory_identity(directory) == identity
    except (OSError, ExperimentError):
        return False


def _remove_verified_transaction_directory(
    directory: Path,
    journal: Mapping[str, Any],
    proof: Path | None = None,
) -> None:
    """Delete one owned tree and restore its proof if deletion stops partway."""

    marker = _transaction_marker_path(directory)
    if proof is None or proof == marker:
        _require_transaction_marker(directory, journal)
    else:
        expected_pending = _marker_pending_path(marker, journal)
        if (
            proof != expected_pending
            or proof.is_symlink()
            or not proof.is_file()
            or _is_reparse_path(proof)
        ):
            raise ExperimentError(f"Invalid transaction ownership publication: {proof}")
    identity = _directory_identity(directory)
    try:
        shutil.rmtree(directory)
    except OSError as exc:
        if _same_directory_identity(directory, identity) and not marker.exists():
            try:
                _write_marker(marker, journal, scope="experiment-directory")
            except (OSError, ExperimentError) as marker_exc:
                raise ExperimentError(
                    f"Could not restore transaction ownership after partial cleanup: {marker_exc}"
                ) from exc
        raise


def _campaign_ownership_state(
    paths: _JournalPaths,
    journal: Mapping[str, Any],
) -> tuple[Path | None, Path | None]:
    """Return a verified marker or a safe incomplete marker from phase prepared."""

    campaign_root = paths.campaign_root
    marker_candidate = _campaign_transaction_marker_path(campaign_root)
    pending_candidate = _marker_pending_path(marker_candidate, journal)
    if pending_candidate.exists() or pending_candidate.is_symlink():
        if (
            journal["phase"] != "prepared"
            or marker_candidate.exists()
            or marker_candidate.is_symlink()
            or pending_candidate.is_symlink()
            or not pending_candidate.is_file()
            or _is_reparse_path(pending_candidate)
        ):
            raise ExperimentError(
                f"Invalid campaign transaction marker publication: {pending_candidate}"
            )
        return None, pending_candidate
    if marker_candidate.exists() or marker_candidate.is_symlink():
        return _require_campaign_transaction_marker(campaign_root, journal), None

    needs_campaign_cleanup = any(
        journal[field] is False
        for field in (
            "campaigns_root_preexisting",
            "campaign_root_preexisting",
            "campaign_readme_preexisting",
        )
    )
    if not needs_campaign_cleanup or not campaign_root.exists():
        return None, None
    marker_gap_is_safe = (
        journal["phase"] == "prepared" or journal["campaign_root_preexisting"] is False
    ) and _contains_only_incomplete_marker(campaign_root, marker_candidate, journal)
    if marker_gap_is_safe:
        return None, None
    raise ExperimentError(
        "Interrupted experiment campaign state lacks its transaction ownership marker: "
        f"{marker_candidate}"
    )


def _registry_snapshot_state(
    registry_path: Path,
    original_registry: Mapping[str, Any],
    updated_registry: Mapping[str, Any],
) -> str:
    try:
        current = load_yaml(registry_path)
    except ConfigError:
        return "unknown"
    if current == original_registry:
        return "original"
    if current == updated_registry:
        return "updated"
    return "unknown"


def _restore_index_snapshot(root: Path, journal: Mapping[str, Any]) -> None:
    index_path = root / "EXPERIMENT_INDEX.md"
    original_index = journal.get("original_index")
    if original_index is None:
        index_path.unlink(missing_ok=True)
        return
    if not isinstance(original_index, str):
        raise ExperimentError("Experiment transaction journal has invalid index snapshot")
    _write_text_atomic(index_path, original_index)


def _remove_journal_owned_state(
    root: Path,
    journal: Mapping[str, Any],
    paths: _JournalPaths,
) -> tuple[list[str], bool]:
    """Remove rollback state only after every existing target proves ownership."""

    errors: list[str] = []
    campaign_marker, incomplete_campaign_marker = _campaign_ownership_state(paths, journal)
    owned_directories = [
        path for path in (paths.staging, paths.destination) if path.exists() or path.is_symlink()
    ]
    owned_markers: dict[Path, Path | None] = {}
    for path in owned_directories:
        owned_markers[path] = _require_experiment_directory_ownership(
            path,
            journal,
            allow_prepared_marker_gap=path == paths.staging,
            allow_recovery_pending=campaign_marker is not None,
        )

    marker_gap_owned = incomplete_campaign_marker is not None or (
        paths.campaign_root.exists()
        and journal["campaign_root_preexisting"] is False
        and _contains_only_incomplete_marker(
            paths.campaign_root,
            _campaign_transaction_marker_path(paths.campaign_root),
            journal,
        )
    )
    owned_state_found = bool(owned_directories or campaign_marker or marker_gap_owned)
    if not owned_state_found:
        return errors, False

    for path in owned_directories:
        try:
            if owned_markers[path] is None:
                shutil.rmtree(path)
            else:
                _remove_verified_transaction_directory(path, journal, owned_markers[path])
        except (OSError, ExperimentError) as exc:
            errors.append(f"could not remove {path}: {exc}")
    if errors:
        return errors, True

    campaign_root = paths.campaign_root
    campaigns_root = campaign_root.parent
    readme = campaign_root / "README.md"
    if journal["campaign_readme_preexisting"] is False and campaign_marker is not None:
        try:
            readme.unlink(missing_ok=True)
        except OSError as exc:
            errors.append(f"could not remove {readme}: {exc}")
            return errors, True

    marker_to_remove = campaign_marker or incomplete_campaign_marker
    if journal["campaign_root_preexisting"] is False and campaign_root.exists():
        allowed = {marker_to_remove.name} if marker_to_remove is not None else set()
        unexpected = sorted(
            path.name for path in campaign_root.iterdir() if path.name not in allowed
        )
        if unexpected:
            errors.append(
                f"refused to remove campaign directory with unowned contents: {unexpected}"
            )
            return errors, True
    if marker_to_remove is not None:
        try:
            marker_to_remove.unlink()
        except OSError as exc:
            errors.append(f"could not remove {marker_to_remove}: {exc}")
            return errors, True
    if journal["campaign_root_preexisting"] is False and campaign_root.exists():
        try:
            campaign_root.rmdir()
        except OSError as exc:
            if marker_to_remove is not None and not marker_to_remove.exists():
                with suppress(OSError):
                    _write_marker(
                        _campaign_transaction_marker_path(campaign_root),
                        journal,
                        scope="campaign-transaction",
                    )
            errors.append(f"could not remove {campaign_root}: {exc}")
    if journal["campaigns_root_preexisting"] is False and campaigns_root.exists():
        with suppress(OSError):
            campaigns_root.rmdir()
    _cleanup_empty_directories([root / "var" / "tmp", root / "var" / "tmp" / "experiment-creation"])
    return errors, True


def _clear_transaction_journal(journal_path: Path) -> None:
    journal_path.unlink(missing_ok=True)


def _recover_interrupted_creation(
    root: Path,
    registry_path: Path,
    journal_path: Path,
) -> None:
    if not journal_path.is_file():
        return
    journal = _load_transaction_journal(journal_path)
    paths = _validate_transaction_journal(root, journal)
    original_registry = journal["original_registry"]
    updated_registry = journal["updated_registry"]
    state = _registry_snapshot_state(registry_path, original_registry, updated_registry)
    if state == "original":
        errors, owned_state_found = _remove_journal_owned_state(root, journal, paths)
        if owned_state_found:
            try:
                _restore_index_snapshot(root, journal)
            except (OSError, ExperimentError) as exc:
                errors.append(f"could not restore experiment index: {exc}")
        else:
            try:
                validate_registry_state(root, original_registry)
                generate_index(root)
            except (OSError, ExperimentError) as exc:
                errors.append(f"could not regenerate experiment index: {exc}")
        if errors:
            raise ExperimentError(
                "Interrupted experiment cleanup is incomplete: " + "; ".join(errors)
            )
        _clear_transaction_journal(journal_path)
        return
    if state != "updated":
        raise ExperimentError(
            "Experiment transaction journal cannot be reconciled with the live registry; "
            f"preserved for recovery at {journal_path}"
        )

    destination = paths.destination
    staging = paths.staging
    phase = journal["phase"]
    campaign_marker_path = _campaign_transaction_marker_path(paths.campaign_root)
    if campaign_marker_path.exists() or campaign_marker_path.is_symlink():
        _require_campaign_transaction_marker(paths.campaign_root, journal)
    elif phase != "index_committed":
        raise ExperimentError(
            "Registered interrupted campaign lacks its transaction ownership marker: "
            f"{campaign_marker_path}"
        )

    destination_marker = _transaction_marker_path(destination)
    if destination.exists() or destination.is_symlink():
        if (
            destination_marker.exists()
            or destination_marker.is_symlink()
            or phase != "index_committed"
        ):
            _require_experiment_directory_ownership(
                destination,
                journal,
                allow_prepared_marker_gap=False,
                allow_recovery_pending=False,
            )
        elif destination.is_symlink() or not destination.is_dir() or _is_reparse_path(destination):
            raise ExperimentError(f"Transaction-owned directory is unavailable: {destination}")
    if staging.exists() or staging.is_symlink():
        _require_experiment_directory_ownership(
            staging,
            journal,
            allow_prepared_marker_gap=False,
            allow_recovery_pending=False,
        )

    if not destination.exists():
        if not staging.is_dir():
            raise ExperimentError(
                "Registered interrupted experiment has neither destination nor staging tree; "
                f"preserved journal at {journal_path}"
            )
        _validate_rendered_experiment(staging)
        if not paths.campaign_root.is_dir():
            raise ExperimentError(
                "Registered interrupted experiment is missing its canonical campaign directory: "
                f"{paths.campaign_root}"
            )
        os.replace(staging, destination)
    validate_registry_state(root, updated_registry)
    generate_index(root)
    journal["phase"] = "index_committed"
    _write_json_atomic(journal_path, journal)
    if staging.exists():
        _require_experiment_directory_ownership(
            staging,
            journal,
            allow_prepared_marker_gap=False,
            allow_recovery_pending=False,
        )
        _remove_verified_transaction_directory(staging, journal)
    destination_marker = _transaction_marker_path(destination)
    if destination_marker.exists() or destination_marker.is_symlink():
        _require_transaction_marker(destination, journal).unlink()
    if campaign_marker_path.exists() or campaign_marker_path.is_symlink():
        _require_campaign_transaction_marker(paths.campaign_root, journal).unlink()
    _clear_transaction_journal(journal_path)
    _cleanup_empty_directories([root / "var" / "tmp", root / "var" / "tmp" / "experiment-creation"])


def _cleanup_failed_creation(
    destination: Path,
    destination_owned: bool,
    staging: Path,
    staging_owned: bool,
    provision: _CampaignProvision | None,
    created_runtime_directories: list[Path],
    journal: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    staging_marker = _transaction_marker_path(staging)
    staging_proven = staging_owned or staging_marker.exists() or staging_marker.is_symlink()
    if staging_proven and (staging.exists() or staging.is_symlink()):
        try:
            _remove_verified_transaction_directory(staging, journal)
        except (OSError, ExperimentError) as exc:
            errors.append(f"could not remove {staging}: {exc}")
    destination_marker = _transaction_marker_path(destination)
    destination_proven = (
        destination_owned or destination_marker.exists() or destination_marker.is_symlink()
    )
    if destination_proven and (destination.exists() or destination.is_symlink()):
        try:
            _remove_verified_transaction_directory(destination, journal)
        except (OSError, ExperimentError) as exc:
            errors.append(f"could not remove {destination}: {exc}")
    if provision is None or errors:
        return errors
    readme = provision.path / "README.md"
    marker = _campaign_transaction_marker_path(provision.path)
    if provision.campaign_directory_created:
        allowed = {marker.name}
        if provision.readme_created:
            allowed.add(readme.name)
        unexpected = sorted(
            path.name for path in provision.path.iterdir() if path.name not in allowed
        )
        if unexpected:
            try:
                _require_campaign_transaction_marker(provision.path, journal).unlink()
            except (OSError, ExperimentError) as exc:
                errors.append(f"could not remove {marker}: {exc}")
            _cleanup_empty_directories(created_runtime_directories)
            return errors
    if provision.readme_created:
        try:
            readme.unlink(missing_ok=True)
        except OSError as exc:
            errors.append(f"could not remove {readme}: {exc}")
            return errors
    if provision.marker_created:
        try:
            marker.unlink(missing_ok=True)
        except OSError as exc:
            errors.append(f"could not remove {marker}: {exc}")
            return errors
    if provision.campaign_directory_created:
        try:
            provision.path.rmdir()
        except OSError as exc:
            if provision.marker_created and not marker.exists():
                with suppress(OSError):
                    _write_marker(marker, journal, scope="campaign-transaction")
            errors.append(f"could not remove {provision.path}: {exc}")
    if provision.campaigns_directory_created:
        with suppress(OSError):
            provision.path.parent.rmdir()
    _cleanup_empty_directories(created_runtime_directories)
    return errors


def _cleanup_empty_directories(paths: list[Path]) -> None:
    for path in reversed(paths):
        with suppress(OSError):
            path.rmdir()


def _transaction_residue_errors(root: Path, journal: Mapping[str, Any]) -> list[str]:
    """Prevent journal loss whenever exact transaction residue still exists."""

    try:
        paths = _validate_transaction_journal(root, journal)
    except ExperimentError as exc:
        return [f"could not verify transaction residue: {exc}"]
    candidates = [
        paths.staging,
        _transaction_marker_path(paths.destination),
        _marker_pending_path(_transaction_marker_path(paths.destination), journal),
        _campaign_transaction_marker_path(paths.campaign_root),
        _marker_pending_path(_campaign_transaction_marker_path(paths.campaign_root), journal),
    ]
    errors = [
        f"transaction residue remains at {path}"
        for path in candidates
        if path.exists() or path.is_symlink()
    ]
    if (
        journal["campaign_root_preexisting"] is False
        and paths.campaign_root.exists()
        and not (paths.campaign_root / "README.md").is_file()
    ):
        errors.append(f"incomplete transaction campaign remains at {paths.campaign_root}")
    return errors


def create_experiment(
    root: Path,
    title: str,
    campaign: str,
) -> Path:
    title = _require_title(title)
    campaign = require_campaign(campaign)

    experiments_root = root / "experiments"
    control_root = experiments_root / EXPERIMENT_CONTROL_DIR
    registry_path = control_root / "registry.yaml"
    lock_path = root / "var" / "run" / "experiment-registry.lock"
    journal_path = root / "var" / "run" / "experiment-creation-transaction.json"
    template = experiments_root / EXPERIMENT_TEMPLATE_DIR
    if not template.is_dir():
        raise ExperimentError(f"Missing experiment template: {template}")
    _require_regular_path_chain(experiments_root, template, "Experiment template")
    _regular_tree_files(template, "Experiment template")
    if not registry_path.is_file():
        raise ExperimentError(f"Missing experiment registry: {registry_path}")

    _require_local_runtime_paths(root)
    _require_runtime_control_file(root, lock_path, "Experiment registry lock")
    _require_runtime_control_file(root, journal_path, "Experiment transaction journal")
    with _registry_lock(lock_path):
        _recover_interrupted_creation(root, registry_path, journal_path)
        try:
            registry = load_yaml(registry_path)
        except ConfigError as exc:
            raise ExperimentError(str(exc)) from exc
        validate_registry_state(root, registry)
        original_registry = deepcopy(registry)
        next_id = registry["next_id"]

        experiment_id = f"EXP-{next_id:04d}"
        directory_name = f"{experiment_id}-{slugify(title)}"
        transaction_token = secrets.token_hex(16)
        created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        destination = experiments_root / "campaigns" / campaign / directory_name
        staging = (
            root / "var" / "tmp" / "experiment-creation" / f"{experiment_id}-{transaction_token}"
        )
        provision: _CampaignProvision | None = None
        destination_owned = False
        staging_owned = False
        runtime_directories = [root / "var", root / "var" / "tmp", staging.parent]
        created_runtime_directories = [path for path in runtime_directories if not path.exists()]
        updated_registry = deepcopy(original_registry)
        updated_registry["next_id"] = next_id + 1
        updated_registry["experiments"].append(
            {
                "id": experiment_id,
                "title": title,
                "campaign": campaign,
                "directory": destination.relative_to(experiments_root).as_posix(),
                "created_at_utc": created_at,
                "status": "PLANNED",
            }
        )
        index_path = root / "EXPERIMENT_INDEX.md"
        try:
            original_index = (
                index_path.read_text(encoding="utf-8") if index_path.is_file() else None
            )
        except OSError as exc:
            raise ExperimentError(f"Could not snapshot experiment index: {index_path}") from exc
        campaigns_root = experiments_root / "campaigns"
        campaign_root = campaigns_root / campaign
        journal: dict[str, Any] = {
            "schema_version": 1,
            "phase": "prepared",
            "transaction_token": transaction_token,
            "experiment_id": experiment_id,
            "destination": destination.relative_to(root).as_posix(),
            "staging": staging.relative_to(root).as_posix(),
            "campaign_root": campaign_root.relative_to(root).as_posix(),
            "campaigns_root_preexisting": campaigns_root.exists(),
            "campaign_root_preexisting": campaign_root.exists(),
            "campaign_readme_preexisting": (campaign_root / "README.md").is_file(),
            "original_registry": original_registry,
            "updated_registry": updated_registry,
            "original_index": original_index,
        }
        journal_written = False
        commit_complete = False
        try:
            _validate_transaction_journal(root, journal)
            _write_json_atomic(journal_path, journal)
            journal_written = True
            if staging.exists() or staging.is_symlink():
                raise ExperimentError(f"Experiment staging directory already exists: {staging}")
            staging.parent.mkdir(parents=True, exist_ok=True)
            _write_transaction_marker(staging, journal)
            staging_owned = True

            provision = _ensure_campaign_root(experiments_root, campaign, journal)
            destination = provision.path / directory_name
            if destination.exists() or destination.is_symlink():
                raise ExperimentError(f"Experiment directory already exists: {destination}")

            shutil.copytree(template, staging, dirs_exist_ok=True, symlinks=True)
            yaml_values = {
                "EXPERIMENT_ID_YAML": experiment_id.replace("'", "''"),
                "TITLE_YAML": title.replace("'", "''"),
                "CREATED_AT_UTC_YAML": created_at.replace("'", "''"),
            }
            values = {
                "EXPERIMENT_ID": experiment_id,
                "TITLE": title,
                "CREATED_AT_UTC": created_at,
                **yaml_values,
            }
            for path in _regular_tree_files(staging, "Rendered experiment staging"):
                _replace_placeholders(path, values)
            _validate_rendered_experiment(staging)
            destination.parent.mkdir(parents=True, exist_ok=True)
            journal["phase"] = "promoting"
            _write_json_atomic(journal_path, journal)
            os.replace(staging, destination)
            destination_owned = True
            staging_owned = False
            journal["phase"] = "promoted"
            _write_json_atomic(journal_path, journal)

            validate_registry_state(root, updated_registry)
            journal["phase"] = "registry_committing"
            _write_json_atomic(journal_path, journal)
            _write_yaml_atomic(registry_path, updated_registry)
            journal["phase"] = "registry_committed"
            _write_json_atomic(journal_path, journal)
            journal["phase"] = "index_committing"
            _write_json_atomic(journal_path, journal)
            generate_index(root)
            journal["phase"] = "index_committed"
            _write_json_atomic(journal_path, journal)
            commit_complete = True
            _require_transaction_marker(destination, journal).unlink()
            if provision.marker_created:
                _require_campaign_transaction_marker(provision.path, journal).unlink()
            _clear_transaction_journal(journal_path)
            _cleanup_empty_directories(created_runtime_directories)
        except BaseException as exc:
            rollback_errors: list[str] = []
            state = _registry_snapshot_state(
                registry_path,
                original_registry,
                updated_registry,
            )
            preserve_forward_state = commit_complete
            if state == "updated" and not preserve_forward_state:
                restore_error: BaseException | None = None
                try:
                    _write_yaml_atomic(registry_path, original_registry)
                except BaseException as rollback_exc:
                    restore_error = rollback_exc
                state = _registry_snapshot_state(
                    registry_path,
                    original_registry,
                    updated_registry,
                )
                if restore_error is not None and state != "original":
                    rollback_errors.append(f"could not restore registry: {restore_error}")
            if state == "unknown":
                rollback_errors.append("live registry no longer matches either journal snapshot")
                preserve_forward_state = True
            elif state == "updated":
                preserve_forward_state = True

            if preserve_forward_state:
                message = (
                    "Experiment creation did not finish cleanly; preserved the destination, "
                    "live registry, and transaction journal for forward index repair"
                )
                if rollback_errors:
                    message += ": " + "; ".join(rollback_errors)
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    exc.add_note(message)
                    raise
                raise ExperimentError(message) from exc

            try:
                _restore_index_snapshot(root, journal)
            except BaseException as rollback_exc:
                rollback_errors.append(f"could not restore experiment index: {rollback_exc}")
            rollback_errors.extend(
                _cleanup_failed_creation(
                    destination,
                    destination_owned,
                    staging,
                    staging_owned,
                    provision,
                    created_runtime_directories,
                    journal,
                )
            )
            if not rollback_errors:
                rollback_errors.extend(_transaction_residue_errors(root, journal))
            if not rollback_errors and journal_written:
                try:
                    _clear_transaction_journal(journal_path)
                except OSError as rollback_exc:
                    rollback_errors.append(f"could not clear transaction journal: {rollback_exc}")
            if rollback_errors:
                details = "; ".join(rollback_errors)
                message = f"Experiment creation failed and rollback was incomplete: {details}"
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    exc.add_note(message)
                    raise
                raise ExperimentError(message) from exc
            if isinstance(exc, ExperimentError):
                raise
            if isinstance(exc, Exception):
                raise ExperimentError(f"Experiment creation failed: {exc}") from exc
            raise
    return destination


__all__ = ["create_experiment"]
