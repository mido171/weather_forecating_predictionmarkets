#!/usr/bin/env python3
"""Bounded, read-only repository health checks for the weather-markets workspace.

The doctor uses only the Python standard library and read-only Git commands. It never follows
reparse points, contacts a provider, starts a service, mutates a database, or changes Git state.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Sequence


ALL_CHECKS = (
    "root",
    "docs",
    "filesystem",
    "tracked-runtime",
    "large-files",
    "secrets",
    "stale-paths",
    "unsafe-defaults",
)

PRUNED_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "target",
    "build",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    "catboost_info",
}

RUNTIME_DIRECTORY_NAMES = {
    ".venv",
    "venv",
    "node_modules",
    "target",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    "catboost_info",
    "run_logs",
    "run-output",
    "run_outputs",
}

RUNTIME_FILE_SUFFIXES = {".log", ".pid", ".pyc", ".pyo", ".tmp"}
ROOT_RUNTIME_DIRECTORIES = {"var", "artifacts", "exports", "reports"}

TEXT_EXTENSIONS = {
    "",
    ".bat",
    ".cfg",
    ".cmd",
    ".conf",
    ".env",
    ".gradle",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".kts",
    ".md",
    ".mjs",
    ".properties",
    ".ps1",
    ".py",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

OPERATIONAL_EXTENSIONS = {
    ".bat",
    ".cfg",
    ".cmd",
    ".conf",
    ".env",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".kts",
    ".mjs",
    ".properties",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".xml",
    ".yaml",
    ".yml",
}

EVIDENCE_DIRECTORY_NAMES = {
    "archive",
    "documentation",
    "docs",
    "experiments",
    "metadata",
    "planning",
    "reports",
    "research",
    "strategy_spec",
}

REQUIRED_ROOT_DOCUMENTS = (
    "AGENTS.md",
    "README.md",
    "docs/START_HERE.md",
    "docs/architecture/REPOSITORY_MAP.md",
    "docs/operations/SAFE_COMMANDS.md",
    "docs/operations/RUNTIME_SAFETY.md",
    "docs/security/SECURITY_BASELINE.md",
)

TRACKED_RUNTIME_DOCUMENTATION = frozenset({"var/README.md"})

STALE_PATH_PROVENANCE_PREFIXES = (
    "legacy/",
    "projects/hkg-tmax/docs/archive/",
    "projects/hkg-tmax/docs/evidence/",
    "projects/hkg-tmax/experiments/",
    "projects/klga-tmax/docs/archive/",
)

POLICY_FIXTURE_PATHS = ("tools/repo/doctor.py", "tools/repo/tests/")

HIGH_CONFIDENCE_SECRET_PATTERNS = (
    ("secret.aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("secret.github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")),
    ("secret.slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    (
        "secret.private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
)

CREDENTIAL_URL = re.compile(
    r"(?i)\b[a-z][a-z0-9+.-]*://(?P<username>[^\s/:@]+):(?P<password>[^\s/@]+)@"
)

CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)^\s*[\"']?"
    r"(?P<key>(?:api[_-]?key|access[_-]?key|secret(?:[_-]?key)?|"
    r"client[_-]?secret|password|passwd|pwd|auth[_-]?token|bearer[_-]?token))"
    r"[\"']?\s*(?P<delimiter>[:=])\s*(?P<value>.+?)\s*$"
)

STALE_PATH_PATTERNS = (
    (
        "stale.bootstrap_path",
        re.compile(
            r"(?i)weather_data_extraction[\\/]+bootstrap[\\/]+"
            r"hkg_tmax_elite_codex_bootstrap[\\/]+hkg_tmax_elite_codex"
        ),
    ),
    (
        "stale.absolute_workspace_path",
        re.compile(
            r"(?i)C:[\\/]+Users[\\/]+ahmad[\\/]+Desktop[\\/]+generalFiles[\\/]+git"
            r"[\\/]+weather_markets[\\/]+weather_data_extraction(?:[\\/]|\b)"
        ),
    ),
)


@dataclass(frozen=True, order=True)
class Finding:
    severity: str
    code: str
    message: str
    path: str | None = None
    line: int | None = None

    def display_location(self) -> str:
        if not self.path:
            return ""
        return f" {self.path}:{self.line}" if self.line else f" {self.path}"


class DoctorError(RuntimeError):
    """Raised when a bounded repository check cannot run reliably."""


def _decode_path(raw: bytes) -> str:
    return raw.decode("utf-8", errors="surrogateescape")


def _is_placeholder(value: str) -> bool:
    candidate = value.strip().strip("'\"").rstrip(",")
    lowered = candidate.lower()
    if not candidate:
        return True
    if candidate.startswith(("${", "$env:", "{{", "<")):
        return True
    if lowered.startswith(("os.getenv", "os.environ", "system.getenv", "env(", "secret(")):
        return True
    if lowered in {
        "null",
        "none",
        "false",
        "true",
        "redacted",
        "***",
        "xxxxx",
        "password",
        "secret",
        "token",
        "localhost",
    }:
        return True
    markers = (
        "change_me",
        "change-me",
        "changeme",
        "example",
        "placeholder",
        "dummy",
        "your_",
        "test_",
        "test-",
        "fake_",
        "fake-",
        "sample_",
        "sample-",
    )
    return any(marker in lowered for marker in markers)


def _looks_binary(data: bytes) -> bool:
    return b"\x00" in data


def _is_evidence_path(relative: str) -> bool:
    parts = {part.lower() for part in PurePosixPath(relative).parts[:-1]}
    return bool(parts & EVIDENCE_DIRECTORY_NAMES)


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(80).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    file_attribute_reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(info, "st_file_attributes", 0) & file_attribute_reparse)


class RepositoryDoctor:
    def __init__(
        self,
        root: Path,
        *,
        scope: str | None = None,
        max_file_bytes: int = 10 * 1024 * 1024,
        full: bool = False,
    ) -> None:
        self.root = root.resolve()
        self.scope = self._normalize_scope(scope)
        self.max_file_bytes = max_file_bytes
        self.full = full
        self._tracked_cache: list[str] | None = None
        self._text_cache: list[tuple[str, str]] | None = None

    def _normalize_scope(self, scope: str | None) -> str | None:
        if scope in (None, "", "."):
            return None
        normalized = PurePosixPath(scope.replace("\\", "/"))
        if normalized.is_absolute() or ".." in normalized.parts:
            raise DoctorError("--scope must be a repository-relative path")
        return normalized.as_posix().rstrip("/")

    def _git(self, *args: str, timeout: int = 60) -> bytes:
        env = os.environ.copy()
        env["GIT_OPTIONAL_LOCKS"] = "0"
        process = subprocess.run(
            ["git", "-C", str(self.root), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
            env=env,
        )
        if process.returncode != 0:
            detail = process.stderr.decode("utf-8", errors="replace").strip()
            raise DoctorError(f"git {' '.join(args)} failed: {detail or process.returncode}")
        return process.stdout

    def tracked_files(self) -> list[str]:
        if self._tracked_cache is None:
            output = self._git("ls-files", "-z")
            paths = [_decode_path(item) for item in output.split(b"\0") if item]
            self._tracked_cache = sorted(path for path in paths if self._in_scope(path))
        return self._tracked_cache

    def _in_scope(self, path: str) -> bool:
        if self.scope is None:
            return True
        normalized = path.replace("\\", "/")
        return normalized == self.scope or normalized.startswith(f"{self.scope}/")

    def _scan_root(self) -> Path:
        return self.root if self.scope is None else self.root / self.scope

    def run(self, checks: Sequence[str] = ALL_CHECKS) -> list[Finding]:
        unknown = sorted(set(checks) - set(ALL_CHECKS))
        if unknown:
            raise DoctorError(f"unknown check(s): {', '.join(unknown)}")
        findings: list[Finding] = []
        dispatch = {
            "root": self.check_root,
            "docs": self.check_documents,
            "filesystem": self.check_filesystem_boundaries,
            "tracked-runtime": self.check_tracked_runtime,
            "large-files": self.check_large_files,
            "secrets": self.check_secrets,
            "stale-paths": self.check_stale_paths,
            "unsafe-defaults": self.check_unsafe_defaults,
        }
        for check in checks:
            findings.extend(dispatch[check]())
        return sorted(findings, key=lambda f: (f.severity, f.code, f.path or "", f.line or 0))

    def check_root(self) -> list[Finding]:
        findings: list[Finding] = []
        dot_git = self.root / ".git"
        if not dot_git.is_dir():
            kind = "missing" if not dot_git.exists() else "not a directory"
            findings.append(
                Finding("error", "git.not_standalone", f"root .git is {kind}; expected a directory")
            )
        try:
            reported = Path(
                self._git("rev-parse", "--show-toplevel").decode("utf-8", errors="replace").strip()
            ).resolve()
            if reported != self.root:
                findings.append(
                    Finding(
                        "error",
                        "git.wrong_root",
                        f"Git reports a different root: {reported}",
                    )
                )
        except DoctorError as exc:
            findings.append(Finding("error", "git.unavailable", str(exc)))
        return findings

    def check_documents(self) -> list[Finding]:
        if self.scope is not None:
            return []
        return [
            Finding("error", "docs.missing", "required root document is missing", path)
            for path in REQUIRED_ROOT_DOCUMENTS
            if not (self.root / path).is_file()
        ]

    def check_filesystem_boundaries(self) -> list[Finding]:
        findings: list[Finding] = []
        scan_root = self._scan_root()
        if not scan_root.exists():
            return [Finding("error", "scope.missing", "scope does not exist", self.scope)]

        max_directories = 250_000 if self.full else 50_000
        visited = 0
        for current, directories, files in os.walk(scan_root, topdown=True, followlinks=False):
            current_path = Path(current)
            visited += 1
            if visited > max_directories:
                findings.append(
                    Finding(
                        "error",
                        "scan.directory_budget",
                        f"directory scan exceeded its {max_directories:,} directory budget",
                    )
                )
                break

            retained: list[str] = []
            for name in directories:
                child = current_path / name
                relative = child.relative_to(self.root).as_posix()
                if name == ".git":
                    if child != self.root / ".git":
                        findings.append(
                            Finding("error", "git.nested", "nested .git directory", relative)
                        )
                    continue
                if _is_reparse_or_symlink(child):
                    findings.append(
                        Finding(
                            "error",
                            "filesystem.reparse_point",
                            "junction, symlink, or reparse-point directory is prohibited",
                            relative,
                        )
                    )
                    continue
                if name in PRUNED_DIRECTORY_NAMES:
                    continue
                retained.append(name)
            directories[:] = retained

            if ".git" in files and current_path != self.root:
                marker = (current_path / ".git").relative_to(self.root).as_posix()
                findings.append(Finding("error", "git.nested", "nested .git file", marker))

        return findings

    def check_tracked_runtime(self) -> list[Finding]:
        findings: list[Finding] = []
        for relative in self.tracked_files():
            if relative in TRACKED_RUNTIME_DOCUMENTATION:
                continue
            if not (self.root / relative).exists():
                # A tracked path deleted in the working tree is being removed, not retained.
                continue
            pure = PurePosixPath(relative)
            lowered_parts = {part.lower() for part in pure.parts}
            reason: str | None = None
            if pure.parts and pure.parts[0].lower() in ROOT_RUNTIME_DIRECTORIES:
                reason = f"root runtime directory '{pure.parts[0]}'"
            elif lowered_parts & RUNTIME_DIRECTORY_NAMES:
                reason = f"runtime/build directory '{sorted(lowered_parts & RUNTIME_DIRECTORY_NAMES)[0]}'"
            elif pure.suffix.lower() in RUNTIME_FILE_SUFFIXES:
                reason = f"runtime file suffix '{pure.suffix.lower()}'"
            if reason:
                findings.append(
                    Finding("error", "tracked.runtime", f"tracked file is in {reason}", relative)
                )
        return findings

    def check_large_files(self) -> list[Finding]:
        findings: list[Finding] = []
        for relative in self.tracked_files():
            path = self.root / relative
            try:
                size = path.stat().st_size
            except (FileNotFoundError, OSError):
                continue
            if size > self.max_file_bytes and not _is_lfs_pointer(path):
                findings.append(
                    Finding(
                        "error",
                        "tracked.large_file",
                        f"tracked file is {size:,} bytes; limit is {self.max_file_bytes:,}",
                        relative,
                    )
                )
        return findings

    def _iter_tracked_text(self) -> Iterator[tuple[str, str]]:
        if self._text_cache is not None:
            yield from self._text_cache
            return
        byte_limit = 5 * 1024 * 1024 if self.full else 1 * 1024 * 1024
        loaded: list[tuple[str, str]] = []
        for relative in self.tracked_files():
            path = self.root / relative
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            try:
                if path.stat().st_size > byte_limit:
                    continue
                data = path.read_bytes()
            except (FileNotFoundError, OSError):
                continue
            if _looks_binary(data):
                continue
            loaded.append((relative, data.decode("utf-8", errors="replace")))
        self._text_cache = loaded
        yield from loaded

    def check_secrets(self) -> list[Finding]:
        findings: list[Finding] = []
        for relative, text in self._iter_tracked_text():
            if relative == POLICY_FIXTURE_PATHS[0] or relative.startswith(POLICY_FIXTURE_PATHS[1]):
                continue
            for code, pattern in HIGH_CONFIDENCE_SECRET_PATTERNS:
                for match in pattern.finditer(text):
                    line = text.count("\n", 0, match.start()) + 1
                    findings.append(
                        Finding(
                            "error",
                            code,
                            "tracked file contains a high-confidence credential shape",
                            relative,
                            line,
                        )
                    )
            if Path(relative).suffix.lower() in OPERATIONAL_EXTENSIONS:
                for match in CREDENTIAL_URL.finditer(text):
                    username = match.group("username")
                    password = match.group("password")
                    if _is_placeholder(username) or _is_placeholder(password) or len(password) < 12:
                        continue
                    line = text.count("\n", 0, match.start()) + 1
                    findings.append(
                        Finding(
                            "error",
                            "secret.credential_url",
                            "tracked operational file contains credentials embedded in a URL",
                            relative,
                            line,
                        )
                    )
            for number, line_text in enumerate(text.splitlines(), start=1):
                match = CREDENTIAL_ASSIGNMENT.match(line_text)
                if not match or _is_placeholder(match.group("value")):
                    continue
                extension = Path(relative).suffix.lower()
                if extension not in OPERATIONAL_EXTENSIONS:
                    continue
                value = match.group("value").strip()
                config_like = extension in {
                    ".cfg",
                    ".conf",
                    ".env",
                    ".ini",
                    ".json",
                    ".properties",
                    ".toml",
                    ".yaml",
                    ".yml",
                }
                quoted_literal = value.startswith(("'", '"'))
                if not config_like and not quoted_literal:
                    continue
                findings.append(
                    Finding(
                        "error",
                        "secret.literal_assignment",
                        f"credential-like key '{match.group('key')}' has a literal value",
                        relative,
                        number,
                    )
                )
        return findings

    def check_stale_paths(self) -> list[Finding]:
        findings: list[Finding] = []
        for relative, text in self._iter_tracked_text():
            if relative == POLICY_FIXTURE_PATHS[0] or relative.startswith(POLICY_FIXTURE_PATHS[1]):
                continue
            if Path(relative).suffix.lower() not in OPERATIONAL_EXTENSIONS:
                continue
            if relative.startswith(("docs/migrations/", "docs/archive/")) or relative.startswith(
                STALE_PATH_PROVENANCE_PREFIXES
            ):
                continue
            for code, pattern in STALE_PATH_PATTERNS:
                matches = list(pattern.finditer(text))
                if not matches:
                    continue
                first_line = text.count("\n", 0, matches[0].start()) + 1
                severity = "warning" if _is_evidence_path(relative) else "error"
                findings.append(
                    Finding(
                        severity,
                        code,
                        f"file contains {len(matches)} stale workspace path occurrence(s)",
                        relative,
                        first_line,
                    )
                )
        return findings

    def check_unsafe_defaults(self) -> list[Finding]:
        findings: list[Finding] = []
        for relative, text in self._iter_tracked_text():
            if relative == POLICY_FIXTURE_PATHS[0] or relative.startswith(POLICY_FIXTURE_PATHS[1]):
                continue
            extension = Path(relative).suffix.lower()
            if extension not in OPERATIONAL_EXTENSIONS:
                continue
            lines = text.splitlines()
            conditional_bodies = re.findall(
                r"@ConditionalOnProperty\s*\((.*?)\)", text, flags=re.DOTALL
            )
            fail_closed_startup_guard = any(
                re.search(r'(?i)havingValue\s*=\s*[\"\']true[\"\']', body)
                and not re.search(r"(?i)matchIfMissing\s*=\s*true", body)
                for body in conditional_bodies
            )
            for index, line in enumerate(lines):
                lowered = line.lower()
                if "repo-doctor: allow-unsafe-default" in lowered:
                    continue
                line_number = index + 1
                if re.search(r"(?i)\bn_jobs\s*[:=]\s*-1\b", line):
                    findings.append(
                        Finding("error", "unsafe.unbounded_jobs", "unbounded n_jobs default", relative, line_number)
                    )
                numeric = re.search(
                    r"(?i)\b(?:max[_-]?workers|workers|threads|concurrency)\s*[:=]\s*(\d+)\b",
                    line,
                )
                if numeric and int(numeric.group(1)) > 2:
                    findings.append(
                        Finding(
                            "warning",
                            "unsafe.high_concurrency",
                            f"configured concurrency is {numeric.group(1)}; review its execution boundary",
                            relative,
                            line_number,
                        )
                    )
                if re.search(r"(?i)^\s*(?:environment|environment-name)\s*:\s*prod\s*$", line):
                    findings.append(
                        Finding(
                            "error",
                            "unsafe.production_default",
                            "production environment is configured as a default",
                            relative,
                            line_number,
                        )
                    )
                if re.search(r"(?i)^\s*enabled\s*:\s*true\s*$", line):
                    context = " ".join(lines[max(0, index - 6) : index + 1]).lower()
                    dangerous = (
                        "trading",
                        "authentication",
                        "auth:",
                        "backfill",
                        "ingestion",
                        "collector",
                        "scheduler",
                        "websocket",
                        "live:",
                    )
                    if any(token in context for token in dangerous):
                        findings.append(
                            Finding(
                                "error",
                                "unsafe.enabled_default",
                                "live, ingestion, backfill, scheduler, authentication, or trading capability defaults to enabled",
                                relative,
                                line_number,
                            )
                        )
                if (
                    re.search(r"@EventListener\s*\(\s*ApplicationReadyEvent\.class\s*\)", line)
                    and not fail_closed_startup_guard
                ):
                    context = " ".join(lines[index : min(len(lines), index + 8)]).lower()
                    if "repo-doctor: allow-unsafe-default" not in context:
                        findings.append(
                            Finding(
                                "warning",
                                "unsafe.startup_listener",
                                "startup listener requires a fail-closed configuration guard",
                                relative,
                                line_number,
                            )
                        )
                if re.search(r"(?i)\bwhile\s+(?:true|\(true\))\b", line):
                    findings.append(
                        Finding(
                            "warning",
                            "unsafe.infinite_loop",
                            "infinite loop requires bounded cancellation and failure handling",
                            relative,
                            line_number,
                        )
                    )
        return findings


def _parse_checks(value: str) -> list[str]:
    checks = [item.strip() for item in value.split(",") if item.strip()]
    if not checks:
        raise argparse.ArgumentTypeError("at least one check is required")
    unknown = sorted(set(checks) - set(ALL_CHECKS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown checks: {', '.join(unknown)}")
    return checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--scope", help="repository-relative path to inspect")
    parser.add_argument(
        "--checks",
        type=_parse_checks,
        default=list(ALL_CHECKS),
        help=f"comma-separated checks (default: {','.join(ALL_CHECKS)})",
    )
    parser.add_argument("--max-file-mib", type=float, default=10.0, help="tracked file size limit")
    parser.add_argument("--full", action="store_true", help="raise bounded scan/text limits")
    parser.add_argument("--strict", action="store_true", help="fail on warnings as well as errors")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def _counts(findings: Iterable[Finding]) -> dict[str, int]:
    result = {"error": 0, "warning": 0, "info": 0}
    for finding in findings:
        result[finding.severity] = result.get(finding.severity, 0) + 1
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_file_mib <= 0:
        print("error: --max-file-mib must be positive", file=sys.stderr)
        return 2
    try:
        doctor = RepositoryDoctor(
            args.root,
            scope=args.scope,
            max_file_bytes=int(args.max_file_mib * 1024 * 1024),
            full=args.full,
        )
        findings = doctor.run(args.checks)
    except (DoctorError, OSError, subprocess.SubprocessError) as exc:
        if args.json:
            print(json.dumps({"tool_error": str(exc)}, indent=2))
        else:
            print(f"repository doctor failed: {exc}", file=sys.stderr)
        return 2

    counts = _counts(findings)
    if args.json:
        payload = {
            "root": str(doctor.root),
            "scope": doctor.scope,
            "checks": list(args.checks),
            "counts": counts,
            "findings": [asdict(finding) for finding in findings],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for finding in findings:
            print(
                f"[{finding.severity.upper()}] {finding.code}{finding.display_location()}: "
                f"{finding.message}"
            )
        print(
            "Repository doctor: "
            f"{counts['error']} error(s), {counts['warning']} warning(s), {counts['info']} info"
        )

    if counts["error"] or (args.strict and counts["warning"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
