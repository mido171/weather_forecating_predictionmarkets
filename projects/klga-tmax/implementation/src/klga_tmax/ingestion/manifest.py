from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from klga_tmax.ingestion.hash_keys import canonical_json, sha256_hex
from klga_tmax.utils.git import current_git_sha


@dataclass(frozen=True)
class SourceManifest:
    job_id: str
    source_name: str
    code_version_git_sha: str
    config_hash: str
    started_at_utc: datetime
    finished_at_utc: datetime | None = None
    row_counts_bronze: int = 0
    row_counts_silver: int = 0
    row_counts_gold: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)


def build_manifest(
    *,
    job_id: str,
    source_name: str,
    source_config: dict[str, Any],
    started_at_utc: datetime,
    finished_at_utc: datetime | None = None,
    row_counts_bronze: int = 0,
    row_counts_silver: int = 0,
    row_counts_gold: int = 0,
    errors: list[dict[str, Any]] | None = None,
    warnings: list[dict[str, Any]] | None = None,
) -> SourceManifest:
    if started_at_utc.tzinfo is None:
        raise ValueError("started_at_utc must be timezone-aware")
    if finished_at_utc is not None and finished_at_utc.tzinfo is None:
        raise ValueError("finished_at_utc must be timezone-aware")
    return SourceManifest(
        job_id=job_id,
        source_name=source_name,
        code_version_git_sha=current_git_sha(),
        config_hash=sha256_hex(canonical_json(source_config)),
        started_at_utc=started_at_utc.astimezone(timezone.utc),
        finished_at_utc=finished_at_utc.astimezone(timezone.utc) if finished_at_utc else None,
        row_counts_bronze=row_counts_bronze,
        row_counts_silver=row_counts_silver,
        row_counts_gold=row_counts_gold,
        errors=errors or [],
        warnings=warnings or [],
    )


def write_manifest(manifest: SourceManifest, artifact_root: Path) -> Path:
    path = artifact_root / "manifests" / f"{manifest.job_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(manifest)
    payload["started_at_utc"] = manifest.started_at_utc.isoformat()
    payload["finished_at_utc"] = (
        manifest.finished_at_utc.isoformat() if manifest.finished_at_utc else None
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
