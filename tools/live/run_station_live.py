from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml
    except Exception:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def _load_dotenv(path: Path) -> dict:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _apply_if_missing(env: dict, key: str, value: object | None) -> None:
    if value is None or value == "":
        return
    env.setdefault(key, str(value))


def _bootstrap_env(repo_root: Path) -> dict:
    env = os.environ.copy()
    _apply_if_missing(env, "PYTHONPATH", str(repo_root))

    live_cfg = _load_yaml(repo_root / "config" / "live_kmia.yaml")
    mysql_cfg = live_cfg.get("mysql", {}) if isinstance(live_cfg, dict) else {}
    grib_cfg = live_cfg.get("gribstream", {}) if isinstance(live_cfg, dict) else {}

    _apply_if_missing(env, "MYSQL_HOST", mysql_cfg.get("host"))
    _apply_if_missing(env, "MYSQL_PORT", mysql_cfg.get("port"))
    _apply_if_missing(env, "MYSQL_DB", mysql_cfg.get("database"))
    _apply_if_missing(env, "MYSQL_USER", mysql_cfg.get("user"))
    _apply_if_missing(env, "MYSQL_PASSWORD", mysql_cfg.get("password"))

    _apply_if_missing(env, "GRIBSTREAM_ACCEPT", grib_cfg.get("accept") or grib_cfg.get("default_accept"))

    secrets_cfg = {}
    for candidate in ("live_env.yaml", "live_secrets.yaml"):
        secrets_cfg = _load_yaml(repo_root / "config" / candidate)
        if secrets_cfg:
            break
    if not secrets_cfg:
        secrets_cfg = _load_dotenv(repo_root / ".env")

    if isinstance(secrets_cfg, dict):
        mysql_secret = secrets_cfg.get("mysql", {}) if isinstance(secrets_cfg.get("mysql", {}), dict) else {}
        grib_secret = secrets_cfg.get("gribstream", {}) if isinstance(secrets_cfg.get("gribstream", {}), dict) else {}
        _apply_if_missing(env, "MYSQL_HOST", mysql_secret.get("host"))
        _apply_if_missing(env, "MYSQL_PORT", mysql_secret.get("port"))
        _apply_if_missing(env, "MYSQL_DB", mysql_secret.get("database"))
        _apply_if_missing(env, "MYSQL_USER", mysql_secret.get("user"))
        _apply_if_missing(env, "MYSQL_PASSWORD", mysql_secret.get("password"))

        token = grib_secret.get("token") or secrets_cfg.get("GRIBSTREAM_TOKEN")
        _apply_if_missing(env, "GRIBSTREAM_TOKEN", token)
        _apply_if_missing(env, "GRIBSTREAM_ACCEPT", grib_secret.get("accept"))

    return env


def _parse_args(raw_args: list[str]) -> list[str]:
    station = None
    target_date = None
    passthrough: list[str] = []
    positional: list[str] = []

    idx = 0
    while idx < len(raw_args):
        arg = raw_args[idx]
        if arg in ("-s", "--station"):
            if idx + 1 < len(raw_args):
                station = raw_args[idx + 1]
            idx += 2
            continue
        if arg in ("-d", "--date", "--target-date"):
            if idx + 1 < len(raw_args):
                target_date = raw_args[idx + 1]
            idx += 2
            continue
        if arg.startswith("-"):
            passthrough.append(arg)
        else:
            positional.append(arg)
        idx += 1

    # If station/date not explicitly set, consume first two positional args.
    if station is None and positional:
        station = positional.pop(0)
    if target_date is None and positional:
        target_date = positional.pop(0)

    normalized: list[str] = []
    if station:
        normalized += ["--station", station]
    if target_date:
        normalized += ["--target-date", target_date]
    normalized += positional
    normalized += passthrough
    return normalized


def main() -> int:
    repo_root = _repo_root()
    env = _bootstrap_env(repo_root)
    script_path = Path(__file__).with_name("run_kmia_live.py")
    argv = _parse_args(sys.argv[1:])

    if "GRIBSTREAM_TOKEN" not in env or not env["GRIBSTREAM_TOKEN"].strip():
        print(
            "Missing GRIBSTREAM_TOKEN. Add it to config/live_env.yaml or set GRIBSTREAM_TOKEN in .env.",
            file=sys.stderr,
        )
        return 2

    completed = subprocess.run([sys.executable, str(script_path), *argv], env=env)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
