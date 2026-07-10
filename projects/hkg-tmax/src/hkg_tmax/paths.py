from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


class ProjectPathError(RuntimeError):
    """Raised when the HKG project or one of its configured roots is invalid."""


class ArchivePathResolutionError(FileNotFoundError):
    """Raised when an archived object cannot be found after relocation."""


_DATA_ANCHORS = frozenset(
    {
        "raw",
        "bronze",
        "silver",
        "gold",
        "metadata",
        "manifests",
        "state",
        "logs",
        "quarantine",
        "cache",
        "datasets",
        "runs",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_PATH_ENV_NAMES = frozenset(
    {
        "HKG_TMAX_DATA_ROOT",
        "HKG_TMAX_RUN_ROOT",
        "HKG_TMAX_CONFIG_ROOT",
        "HKG_TMAX_DB_ROOT",
        "HKG_TMAX_INPUT_ROOT",
        "HKG_TMAX_STORAGE_ROOT_ID",
    }
)


def _path_environment(
    project_root: Path,
    environ: Mapping[str, str] | None,
) -> Mapping[str, str]:
    """Return path-only settings, optionally supplemented from the local `.env`.

    Direct scripts do not pass through the CLI's dotenv loader. Reading only the
    allowlisted path keys keeps those scripts relocatable without importing provider,
    database, or trading credentials as a side effect. Tests that pass an explicit mapping
    remain hermetic and never read the workstation `.env`.
    """

    if environ is not None:
        return environ
    values = dict(os.environ)
    env_path = project_root / ".env"
    if not env_path.is_file():
        return values
    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        name = key.strip()
        if name not in _PATH_ENV_NAMES or values.get(name, "").strip():
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[name] = value
    return values


def _configured_path(
    environ: Mapping[str, str],
    name: str,
    *,
    project_root: Path,
    default: Path,
) -> Path:
    configured = environ.get(name, "").strip()
    path = Path(configured).expanduser() if configured else default
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def find_project_root(
    start: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Find the nearest HKG project root without depending on package depth."""

    env = os.environ if environ is None else environ
    configured = env.get("HKG_TMAX_ROOT", "").strip()
    if configured:
        configured_root = Path(configured).expanduser()
        if not configured_root.is_absolute():
            configured_root = Path.cwd() / configured_root
        candidate = configured_root.resolve()
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "config").is_dir()
            and (candidate / "src" / "hkg_tmax").is_dir()
        ):
            return candidate
        raise ProjectPathError(
            f"HKG_TMAX_ROOT does not identify an HKG project root: {candidate}"
        )

    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "config").is_dir()
            and (candidate / "src" / "hkg_tmax").is_dir()
        ):
            return candidate
    raise ProjectPathError(
        f"Could not find the HKG project root from {current}. "
        "Expected pyproject.toml, config/, and src/hkg_tmax/."
    )


@dataclass(frozen=True)
class ProjectPaths:
    """Authoritative filesystem roots for the HKG project and its runtime data."""

    project_root: Path
    data_root: Path
    run_root: Path
    config_root: Path
    db_root: Path
    storage_root_id: str

    @classmethod
    def discover(
        cls,
        start: Path | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> ProjectPaths:
        return cls.from_project_root(
            find_project_root(start, environ=environ),
            environ=environ,
        )

    @classmethod
    def from_project_root(
        cls,
        project_root: Path,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> ProjectPaths:
        root = project_root.resolve()
        env = _path_environment(root, environ)
        data_root = _configured_path(
            env,
            "HKG_TMAX_DATA_ROOT",
            project_root=root,
            default=root / "data",
        )
        run_root = _configured_path(
            env,
            "HKG_TMAX_RUN_ROOT",
            project_root=root,
            default=data_root / "runs",
        )
        config_root = _configured_path(
            env,
            "HKG_TMAX_CONFIG_ROOT",
            project_root=root,
            default=root / "config",
        )
        default_db_root = root / "db" if (root / "db").is_dir() else root
        db_root = _configured_path(
            env,
            "HKG_TMAX_DB_ROOT",
            project_root=root,
            default=default_db_root,
        )
        storage_root_id = env.get("HKG_TMAX_STORAGE_ROOT_ID", "hkg-tmax-data").strip()
        if not storage_root_id:
            raise ProjectPathError("HKG_TMAX_STORAGE_ROOT_ID must not be empty")
        return cls(
            project_root=root,
            data_root=data_root,
            run_root=run_root,
            config_root=config_root,
            db_root=db_root,
            storage_root_id=storage_root_id,
        )


def configured_input_path(
    paths: ProjectPaths,
    env_name: str,
    filename: str,
    *,
    environ: Mapping[str, str] | None = None,
    legacy_home_relative: Path | None = None,
    home: Path | None = None,
) -> Path:
    """Resolve a relocatable operator-supplied input without user-specific defaults.

    A file-specific environment variable takes precedence, followed by
    ``HKG_TMAX_INPUT_ROOT`` (defaulting to ``run_root/inputs``). A home-relative
    legacy location may be supplied as a read-only transition fallback; it is
    used only when the preferred external input does not yet exist.
    """

    env = os.environ if environ is None else environ
    configured = env.get(env_name, "").strip()
    if configured:
        return _configured_path(
            env,
            env_name,
            project_root=paths.project_root,
            default=paths.run_root / "inputs" / filename,
        )

    input_root = _configured_path(
        env,
        "HKG_TMAX_INPUT_ROOT",
        project_root=paths.project_root,
        default=paths.run_root / "inputs",
    )
    preferred = (input_root / filename).resolve()
    if preferred.is_file() or legacy_home_relative is None:
        return preferred

    legacy = ((home or Path.home()) / legacy_home_relative).resolve()
    return legacy if legacy.is_file() else preferred


def infer_storage_root(raw_root: Path) -> Path:
    """Return the data root for a raw archive root while preserving test fixtures."""

    root = raw_root.resolve()
    return root.parent if root.name.casefold() == "raw" else root


def _relative_to_root(path: Path, root: Path, *, field: str) -> str:
    resolved_root = root.resolve()
    resolved = path.resolve(strict=False)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ProjectPathError(f"{field} is outside storage root {resolved_root}: {resolved}") from exc
    return relative.as_posix()


def archive_reference_fields(
    *,
    data_root: Path,
    content_path: Path,
    sidecar_path: Path,
    storage_root_id: str | None = None,
) -> dict[str, object]:
    """Build relocation-safe fields while retaining legacy absolute provenance."""

    root = data_root.resolve()
    absolute_content = content_path.resolve(strict=False)
    absolute_sidecar = sidecar_path.resolve(strict=False)
    configured_root_id = storage_root_id or os.getenv("HKG_TMAX_STORAGE_ROOT_ID") or "hkg-tmax-data"
    root_id = configured_root_id.strip()
    if not root_id:
        raise ProjectPathError("storage_root_id must not be empty")
    return {
        "storage_schema_version": 2,
        "storage_root_id": root_id,
        "content_path": str(absolute_content),
        "sidecar_path": str(absolute_sidecar),
        "content_relpath": _relative_to_root(absolute_content, root, field="content_path"),
        "sidecar_relpath": _relative_to_root(absolute_sidecar, root, field="sidecar_path"),
        "legacy_content_path": str(absolute_content),
        "legacy_sidecar_path": str(absolute_sidecar),
    }


def _portable_path_parts(value: str) -> tuple[str, ...]:
    if "\\" in value or re.match(r"^[A-Za-z]:", value):
        return PureWindowsPath(value).parts
    return PurePosixPath(value).parts


def _portable_name(value: str) -> str:
    parts = _portable_path_parts(value)
    return parts[-1] if parts else ""


def _native_absolute_path(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.is_absolute() else None


def _safe_relative_path(value: object) -> Path | None:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return None
    if re.match(r"^[A-Za-z]:", text):
        raise ArchivePathResolutionError(f"Unsafe archive relative path: {text!r}")
    pure = PurePosixPath(text)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ArchivePathResolutionError(f"Unsafe archive relative path: {text!r}")
    return Path(*pure.parts)


def _relative_from_legacy_path(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    parts = _portable_path_parts(text)
    for index, part in enumerate(parts):
        if part.casefold() in _DATA_ANCHORS:
            return _safe_relative_path("/".join(parts[index:]))
    return None


def _contained_candidate(data_root: Path, relative: Path) -> Path:
    root = data_root.resolve()
    candidate = (root / relative).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ArchivePathResolutionError(
            f"Archive path escapes configured data root: {relative}"
        ) from exc
    return candidate


def _extension(record: Mapping[str, Any], legacy_value: object) -> str:
    direct = str(record.get("extension", "")).strip().lstrip(".")
    metadata = record.get("metadata")
    inferred = ""
    if isinstance(metadata, Mapping):
        inferred = str(metadata.get("extension_inferred", "")).strip().lstrip(".")
    if direct or inferred:
        candidate = direct or inferred
        return candidate if candidate.replace("_", "").isalnum() else ""
    name = _portable_name(str(legacy_value or ""))
    suffix = PurePosixPath(name).suffix.lstrip(".")
    return suffix if suffix.replace("_", "").isalnum() else ""


def _first_existing_file(candidates: list[Path]) -> Path | None:
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def resolve_archive_content_path(
    record: Mapping[str, Any],
    *,
    data_root: Path,
    sidecar_path: Path | None = None,
) -> Path:
    """Resolve an archived content object after a data-root relocation.

    Resolution is deliberately ordered: an existing legacy absolute path,
    a configured-root relative path, a content-addressed object, then a file
    adjacent to the relocated sidecar. Missing or ambiguous data fails closed.
    """

    root = data_root.resolve()
    legacy_values = (record.get("content_path"), record.get("legacy_content_path"))
    legacy_value = next((value for value in legacy_values if str(value or "").strip()), "")

    for value in legacy_values:
        absolute = _native_absolute_path(value)
        if absolute is not None and absolute.is_file():
            return absolute

    relative = _safe_relative_path(record.get("content_relpath"))
    if relative is None:
        for value in legacy_values:
            relative = _relative_from_legacy_path(value)
            if relative is not None:
                break
    if relative is not None:
        candidate = _contained_candidate(root, relative)
        if candidate.is_file():
            return candidate

    digest = str(record.get("content_sha256", "")).strip().lower()
    extension = _extension(record, legacy_value)
    if _SHA256_RE.fullmatch(digest):
        object_dir = root / "raw" / "objects" / digest[:2]
        if extension:
            candidate = object_dir / f"{digest}.{extension}"
            if candidate.is_file():
                return candidate
        matches = sorted(
            path
            for path in object_dir.glob(f"{digest}.*")
            if path.is_file() and not path.name.endswith(".metadata.json")
        )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ArchivePathResolutionError(
                f"Ambiguous content-addressed archive object for sha256={digest} under {object_dir}"
            )

    sidecar_candidates: list[Path] = []
    if sidecar_path is not None:
        sidecar_candidates.append(sidecar_path.resolve(strict=False))
    sidecar_values = (record.get("sidecar_path"), record.get("legacy_sidecar_path"))
    for value in sidecar_values:
        recorded_sidecar = _native_absolute_path(value)
        if recorded_sidecar is not None:
            sidecar_candidates.append(recorded_sidecar)
    sidecar_relative = _safe_relative_path(record.get("sidecar_relpath"))
    if sidecar_relative is None:
        for value in sidecar_values:
            sidecar_relative = _relative_from_legacy_path(value)
            if sidecar_relative is not None:
                break
    if sidecar_relative is not None:
        sidecar_candidates.append(_contained_candidate(root, sidecar_relative))

    content_name = _portable_name(str(legacy_value or ""))
    sibling_candidates: list[Path] = []
    for sidecar in sidecar_candidates:
        if content_name:
            sibling_candidates.append(sidecar.with_name(content_name))
        if sidecar.name.endswith(".metadata.json") and extension:
            sibling_candidates.append(
                sidecar.with_name(sidecar.name.removesuffix(".metadata.json") + f".{extension}")
            )
    sibling = _first_existing_file(sibling_candidates)
    if sibling is not None:
        return sibling

    raise ArchivePathResolutionError(
        "Unable to resolve archived content under configured data root "
        f"{root} (sha256={digest or 'missing'}, relpath={record.get('content_relpath') or 'missing'})"
    )
