from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .paths import ProjectPathError, find_project_root


class ConfigError(RuntimeError):
    """Raised when repository configuration is missing or invalid."""


def find_repo_root(start: Path | None = None) -> Path:
    """Backward-compatible alias for centralized HKG project discovery."""

    try:
        return find_project_root(start)
    except ProjectPathError as exc:
        raise ConfigError(str(exc)) from exc


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Missing YAML file: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {path}")
    return data


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class Source:
    id: str
    provider: str
    role: str
    access: dict[str, Any]
    point_in_time_status: str
    raw: dict[str, Any]

    @property
    def url(self) -> str:
        value = self.access.get("url")
        if not isinstance(value, str) or not value:
            raise ConfigError(f"Source {self.id!r} has no usable access.url")
        return value

    @property
    def tags(self) -> tuple[str, ...]:
        value = self.raw.get("tags", [])
        if value is None:
            return ()
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise ConfigError(f"Source {self.id!r} tags must be a list of strings")
        return tuple(value)


@dataclass(frozen=True)
class SourceCatalog:
    sources: tuple[Source, ...]
    required_fields: tuple[str, ...]

    @classmethod
    def from_path(cls, path: Path) -> SourceCatalog:
        data = load_yaml(path)
        required = tuple(data.get("required_fields", ()))
        raw_sources = data.get("sources")
        if not isinstance(raw_sources, list):
            raise ConfigError(f"{path}: sources must be a list")

        seen: set[str] = set()
        parsed: list[Source] = []
        for index, raw in enumerate(raw_sources):
            if not isinstance(raw, dict):
                raise ConfigError(f"{path}: sources[{index}] must be a mapping")
            missing = [field for field in required if field not in raw]
            if missing:
                raise ConfigError(
                    f"{path}: source at index {index} missing required fields: {missing}"
                )
            source_id = raw.get("id")
            if not isinstance(source_id, str) or not source_id:
                raise ConfigError(f"{path}: source id at index {index} is invalid")
            if source_id in seen:
                raise ConfigError(f"{path}: duplicate source id {source_id!r}")
            seen.add(source_id)
            access = raw.get("access")
            if not isinstance(access, dict):
                raise ConfigError(f"{path}: source {source_id!r} access must be a mapping")
            parsed.append(
                Source(
                    id=source_id,
                    provider=str(raw["provider"]),
                    role=str(raw["role"]),
                    access=access,
                    point_in_time_status=str(raw["point_in_time_status"]),
                    raw=raw,
                )
            )
        return cls(tuple(parsed), required)

    def get(self, source_id: str) -> Source:
        for source in self.sources:
            if source.id == source_id:
                return source
        raise ConfigError(f"Unknown source id: {source_id}")

    def tagged(self, tag: str) -> tuple[Source, ...]:
        return tuple(source for source in self.sources if tag in source.tags)

    def select(
        self,
        source_ids: Iterable[str] | None = None,
        tag: str | None = None,
    ) -> tuple[Source, ...]:
        if source_ids:
            selected = tuple(self.get(source_id) for source_id in source_ids)
        elif tag:
            selected = self.tagged(tag)
        else:
            selected = self.sources
        return selected
