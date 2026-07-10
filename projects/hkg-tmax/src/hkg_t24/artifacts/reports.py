"""Report writers for the HKG-T24-001 foundation slice."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from hkg_t24.constants import REPORT_NAMES
from hkg_t24.utils.sql import csv_line


@dataclass(frozen=True)
class ReportPathSet:
    repo_root: Path

    @property
    def reports_dir(self) -> Path:
        return self.repo_root / "reports"

    @property
    def hkg_reports_dir(self) -> Path:
        return self.repo_root / "reports" / "hkg_t24"

    @property
    def context_dir(self) -> Path:
        return self.repo_root / "documentation" / "strategy_implementation_documentation" / "context"


class ReportWriter:
    """Simple deterministic report writer."""

    def __init__(self, repo_root: Path) -> None:
        self.paths = ReportPathSet(repo_root=repo_root)
        self.paths.reports_dir.mkdir(parents=True, exist_ok=True)
        self.paths.hkg_reports_dir.mkdir(parents=True, exist_ok=True)
        self.paths.context_dir.mkdir(parents=True, exist_ok=True)

    def write_root_report(self, name: str, title: str, sections: Sequence[tuple[str, str]]) -> Path:
        if name not in REPORT_NAMES and not name.endswith("_report.md"):
            raise ValueError(f"Unexpected HKG-T24 report name: {name}")
        path = self.paths.reports_dir / name
        lines = [f"# {title}", ""]
        for heading, body in sections:
            lines.extend([f"## {heading}", "", body.rstrip(), ""])
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return path

    def write_csv(self, name: str, headers: Sequence[str], rows: Iterable[Sequence[object | None]]) -> Path:
        path = self.paths.reports_dir / name
        lines = [csv_line(headers)]
        lines.extend(csv_line(row) for row in rows)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def write_hkg_report(self, name: str, title: str, sections: Sequence[tuple[str, str]]) -> Path:
        path = self.paths.hkg_reports_dir / name
        lines = [f"# {title}", ""]
        for heading, body in sections:
            lines.extend([f"## {heading}", "", body.rstrip(), ""])
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return path

    def write_context_doc(self, name: str, title: str, sections: Sequence[tuple[str, str]]) -> Path:
        path = self.paths.context_dir / name
        lines = [f"# {title}", ""]
        for heading, body in sections:
            lines.extend([f"## {heading}", "", body.rstrip(), ""])
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return path
