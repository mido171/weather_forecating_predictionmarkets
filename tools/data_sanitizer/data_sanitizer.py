from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


REQUIRED_COLUMNS = [
    "request_location_id",
    "valid_time_utc",
    "temp",
    "dew_pt",
    "rh",
    "pressure",
    "vis",
    "wspd",
    "wdir",
    "gust",
    "precip_hrly",
    "clds",
    "wx_phrase",
    "uv_index",
    "uv_desc",
    "wdir_cardinal",
]
NUMERIC_COLUMNS = ["temp", "dew_pt", "rh", "pressure", "vis", "wspd", "wdir", "gust", "precip_hrly", "uv_index"]
NUMERIC_SCORE_COLUMNS = list(NUMERIC_COLUMNS)
CATEGORICAL_COLUMNS = ["clds", "wx_phrase", "uv_desc", "wdir_cardinal"]

WDIR_ALLOWED = {"N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "VAR", "VRB", "CALM"}
WDIR_CENTER = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5, "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5, "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
    "VAR": np.nan, "CALM": np.nan,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean: {v}")


def _default_rules() -> dict[str, Any]:
    return {
        "null_tokens": ["", "null", "none", "nan", "na", "n/a"],
        "bounds": {
            "temp": [-100.0, 140.0],
            "dew_pt": [-100.0, 110.0],
            "rh": [0.0, 100.0],
            "pressure_inhg": [25.0, 35.0],
            "pressure_hpa": [850.0, 1100.0],
            "pressure_unknown": [0.0, 9999.0],
            "vis": [0.0, 60.0],
            "wspd": [0.0, 200.0],
            "wdir": [0.0, 360.0],
            "gust": [0.0, 250.0],
            "precip_hrly": [0.0, 10.0],
            "uv_index": [0.0, 20.0],
        },
        "sentinels": {"wdir": [999.0]},
        "clds_allowed": ["CLR", "SKC", "FEW", "SCT", "BKN", "OVC", "VV", "NSC"],
        "clds_map": {"CLEAR": "CLR", "SCATTERED": "SCT", "BROKEN": "BKN", "OVERCAST": "OVC"},
        "max_phrase_length": 200,
        "convert_pressure_to_inhg": False,
    }


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def load_rules(path: str | Path | None = None) -> dict[str, Any]:
    rules = _default_rules()
    if not path:
        return rules
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Rules file not found: {p}")
    if yaml is None:
        raise RuntimeError("PyYAML is required to parse YAML rules.")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Rules file must parse to dict: {p}")
    return _merge_dict(rules, cfg)


def read_station_universe(path: str | Path | None) -> set[str] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Station universe not found: {p}")
    df = pd.read_csv(p, low_memory=False)
    if "request_location_id" not in df.columns:
        raise ValueError(f"station_universe missing request_location_id: {p}")
    vals = df["request_location_id"].astype(str).str.strip().str.upper()
    vals = vals[vals != ""]
    return set(vals.tolist())


def _normalize_text(series: pd.Series, null_tokens: set[str]) -> pd.Series:
    s = series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    s = s.mask(s.str.lower().isin(null_tokens), pd.NA)
    s = s.mask(s == "", pd.NA)
    return s


def _wdir_to_cardinal(wdir: pd.Series) -> pd.Series:
    arr = pd.to_numeric(wdir, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(arr), pd.NA, dtype=object)
    valid = np.isfinite(arr)
    if valid.any():
        deg = np.mod(arr[valid], 360.0)
        names = np.array(["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"], dtype=object)
        idx = np.floor((deg + 11.25) / 22.5).astype(int) % 16
        out[np.where(valid)[0]] = names[idx]
    return pd.Series(out, index=wdir.index, dtype="string")


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _git_hash() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return r.stdout.strip() or None
    except Exception:
        return None


def sanitize_observations_dataframe(
    df: pd.DataFrame,
    *,
    rules: dict[str, Any] | None = None,
    station_universe: set[str] | None = None,
    emit_flags: bool = False,
    drop_invalid_timestamps: bool = True,
    fill_wdir_from_cardinal: bool = False,
    enforce_30m_grid: bool = False,
    row_order_start: int = 0,
    collect_triggered_rules: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = rules or _default_rules()
    null_tokens = set(str(x).strip().lower() for x in cfg.get("null_tokens", []))
    bounds = cfg["bounds"]
    sentinels = cfg.get("sentinels", {})
    clds_allowed = set(str(x).strip().upper() for x in cfg.get("clds_allowed", []))
    clds_map = {str(k).upper(): str(v).upper() for k, v in (cfg.get("clds_map") or {}).items()}
    max_phrase_len = int(cfg.get("max_phrase_length", 200))
    convert_pressure_to_inhg = bool(cfg.get("convert_pressure_to_inhg", False))

    out = df.copy()
    missing_required = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    rule_counts: Counter[str] = Counter()
    col_invalid: Counter[str] = Counter()
    station_rules: dict[str, Counter[str]] = defaultdict(Counter)
    minute_hist: Counter[str] = Counter()
    clds_unknown: Counter[str] = Counter()
    dropped_rows = 0
    dropped_invalid_ts = 0
    dropped_unknown_station = 0
    dropped_off_grid = 0
    pressure_unit = "unknown"

    triggered = [[] for _ in range(len(out))] if collect_triggered_rules else None
    flags = {f"{c}_was_sanitized": np.zeros(len(out), dtype=bool) for c in NUMERIC_COLUMNS} if emit_flags else {}
    if emit_flags:
        flags["valid_time_utc_was_sanitized"] = np.zeros(len(out), dtype=bool)
        flags["request_location_id_was_sanitized"] = np.zeros(len(out), dtype=bool)

    def record(mask: pd.Series, rule: str, col: str | None = None) -> int:
        nonlocal triggered
        if not isinstance(mask, pd.Series):
            mask = pd.Series(mask, index=out.index)
        n = int(mask.sum())
        if n <= 0:
            return 0
        rule_counts[rule] += n
        if col:
            col_invalid[col] += n
            if emit_flags and f"{col}_was_sanitized" in flags:
                flags[f"{col}_was_sanitized"] |= mask.to_numpy(dtype=bool)
        stations = out.loc[mask, "request_location_id"].astype("string").str.upper().fillna("MISSING")
        for st, cnt in stations.value_counts().to_dict().items():
            station_rules[str(st)][rule] += int(cnt)
        if collect_triggered_rules and triggered is not None:
            idx = np.where(mask.to_numpy(dtype=bool))[0]
            for i in idx:
                triggered[i].append(rule)
        return n

    def apply_bounds(col: str, lo: float, hi: float, rule: str) -> None:
        m = out[col].notna() & ((out[col] < lo) | (out[col] > hi))
        if m.any():
            record(m, rule, col)
            out.loc[m, col] = np.nan

    out["request_location_id"] = _normalize_text(out["request_location_id"], null_tokens).str.upper()
    empty_station = out["request_location_id"].isna()
    if empty_station.any():
        record(empty_station, "station_empty_drop", "request_location_id")
        if emit_flags:
            flags["request_location_id_was_sanitized"] |= empty_station.to_numpy(dtype=bool)

    unknown_station = pd.Series(False, index=out.index)
    if station_universe is not None:
        unknown_station = out["request_location_id"].notna() & ~out["request_location_id"].isin(station_universe)
        if unknown_station.any():
            record(unknown_station, "unknown_station_drop", "request_location_id")
            if emit_flags:
                flags["request_location_id_was_sanitized"] |= unknown_station.to_numpy(dtype=bool)
    drop_station = empty_station | unknown_station
    if drop_station.any():
        dropped_unknown_station += int(drop_station.sum())
        dropped_rows += int(drop_station.sum())
        keep = ~drop_station
        out = out.loc[keep].reset_index(drop=True)
        if triggered is not None:
            triggered = [r for r, k in zip(triggered, keep.to_numpy(dtype=bool)) if k]
        if emit_flags:
            for k in flags:
                flags[k] = flags[k][keep.to_numpy(dtype=bool)]

    ts = pd.to_datetime(out["valid_time_utc"], errors="coerce", utc=True)
    invalid_ts = ts.isna()
    if invalid_ts.any():
        record(invalid_ts, "invalid_timestamp", "valid_time_utc")
        if emit_flags:
            flags["valid_time_utc_was_sanitized"] |= invalid_ts.to_numpy(dtype=bool)
    if drop_invalid_timestamps and invalid_ts.any():
        dropped_invalid_ts += int(invalid_ts.sum())
        dropped_rows += int(invalid_ts.sum())
        keep = ~invalid_ts
        out = out.loc[keep].reset_index(drop=True)
        ts = ts.loc[keep].reset_index(drop=True)
        if triggered is not None:
            triggered = [r for r, k in zip(triggered, keep.to_numpy(dtype=bool)) if k]
        if emit_flags:
            for k in flags:
                flags[k] = flags[k][keep.to_numpy(dtype=bool)]

    if enforce_30m_grid and len(out) > 0:
        off_grid = ((ts.dt.minute % 30) != 0) | (ts.dt.second != 0)
        if off_grid.any():
            record(off_grid, "off_grid_timestamp_drop", "valid_time_utc")
            dropped_off_grid += int(off_grid.sum())
            dropped_rows += int(off_grid.sum())
            keep = ~off_grid
            out = out.loc[keep].reset_index(drop=True)
            ts = ts.loc[keep].reset_index(drop=True)
            if triggered is not None:
                triggered = [r for r, k in zip(triggered, keep.to_numpy(dtype=bool)) if k]
            if emit_flags:
                for k in flags:
                    flags[k] = flags[k][keep.to_numpy(dtype=bool)]

    if len(out) > 0:
        for minute, cnt in ts.dt.minute.value_counts().to_dict().items():
            minute_hist[str(int(minute))] += int(cnt)
    out["valid_time_utc"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.loc[ts.isna(), "valid_time_utc"] = np.nan

    for c in NUMERIC_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    apply_bounds("temp", *map(float, bounds["temp"]), "temp_out_of_bounds")
    apply_bounds("dew_pt", *map(float, bounds["dew_pt"]), "dewpt_out_of_bounds")
    apply_bounds("rh", *map(float, bounds["rh"]), "rh_out_of_bounds")
    apply_bounds("vis", *map(float, bounds["vis"]), "vis_out_of_bounds")
    apply_bounds("wspd", *map(float, bounds["wspd"]), "wspd_out_of_bounds")
    apply_bounds("gust", *map(float, bounds["gust"]), "gust_out_of_bounds")
    apply_bounds("precip_hrly", *map(float, bounds["precip_hrly"]), "precip_out_of_bounds")

    wdir_sentinel = {float(x) for x in sentinels.get("wdir", [])}
    if wdir_sentinel:
        m = out["wdir"].isin(wdir_sentinel)
        if m.any():
            record(m, "wdir_sentinel_999", "wdir")
            out.loc[m, "wdir"] = np.nan
    apply_bounds("wdir", *map(float, bounds["wdir"]), "wdir_out_of_bounds")

    calm = out["wspd"].notna() & (out["wspd"] < 1.0) & out["wdir"].notna()
    if calm.any():
        record(calm, "wdir_cleared_calm", "wdir")
        out.loc[calm, "wdir"] = np.nan

    dew_clip = out["dew_pt"].notna() & out["temp"].notna() & (out["dew_pt"] > (out["temp"] + 2.0))
    if dew_clip.any():
        record(dew_clip, "dewpt_gt_temp_clipped", "dew_pt")
        out.loc[dew_clip, "dew_pt"] = out.loc[dew_clip, "temp"]

    gust_clip = out["gust"].notna() & out["wspd"].notna() & (out["gust"] < out["wspd"])
    if gust_clip.any():
        record(gust_clip, "gust_lt_wspd_clipped", "gust")
        out.loc[gust_clip, "gust"] = out.loc[gust_clip, "wspd"]

    uv_neg = out["uv_index"].notna() & (out["uv_index"] < 0.0)
    if uv_neg.any():
        record(uv_neg, "uv_negative_sentinel", "uv_index")
        out.loc[uv_neg, "uv_index"] = np.nan
    uv_high = out["uv_index"].notna() & (out["uv_index"] > float(bounds["uv_index"][1]))
    if uv_high.any():
        record(uv_high, "uv_out_of_bounds", "uv_index")
        out.loc[uv_high, "uv_index"] = np.nan

    p = out["pressure"]
    p_med = float(np.nanmedian(p.to_numpy(dtype=float))) if p.notna().any() else np.nan
    if np.isfinite(p_med) and 20.0 <= p_med <= 40.0:
        pressure_unit = "inhg"
        plo, phi = bounds["pressure_inhg"]
    elif np.isfinite(p_med) and 850.0 <= p_med <= 1100.0:
        if convert_pressure_to_inhg:
            out["pressure"] = out["pressure"] * 0.029529983071445
            pressure_unit = "hpa_to_inhg"
            plo, phi = bounds["pressure_inhg"]
        else:
            pressure_unit = "hpa"
            plo, phi = bounds["pressure_hpa"]
    else:
        pressure_unit = "unknown"
        plo, phi = bounds["pressure_unknown"]
        if out["pressure"].notna().any():
            record(out["pressure"].notna(), "pressure_unit_unknown", "pressure")
    apply_bounds("pressure", float(plo), float(phi), "pressure_out_of_bounds")

    for c in CATEGORICAL_COLUMNS:
        out[c] = _normalize_text(out[c], null_tokens)

    clds = out["clds"].str.upper().replace(clds_map)
    unknown_clds = clds.notna() & ~clds.isin(clds_allowed)
    if unknown_clds.any():
        record(unknown_clds, "clds_unknown_value", "clds")
        clds_unknown.update(clds[unknown_clds].astype(str).value_counts().to_dict())
    out["clds"] = clds

    card = out["wdir_cardinal"].str.upper().replace({"VRB": "VAR"})
    unknown_card = card.notna() & ~card.isin(WDIR_ALLOWED)
    if unknown_card.any():
        record(unknown_card, "wdir_cardinal_unknown_value", "wdir_cardinal")
        card = card.mask(unknown_card, pd.NA)
    fill_card = card.isna() & out["wdir"].notna()
    if fill_card.any():
        record(fill_card, "wdir_cardinal_filled_from_wdir", "wdir_cardinal")
        card.loc[fill_card] = _wdir_to_cardinal(out.loc[fill_card, "wdir"])
    out["wdir_cardinal"] = card

    if fill_wdir_from_cardinal:
        fill_wdir = out["wdir"].isna() & out["wdir_cardinal"].notna()
        if fill_wdir.any():
            record(fill_wdir, "wdir_filled_from_cardinal", "wdir")
            out.loc[fill_wdir, "wdir"] = out.loc[fill_wdir, "wdir_cardinal"].map(WDIR_CENTER).astype(float)

    long_phrase = out["wx_phrase"].notna() & (out["wx_phrase"].str.len() > max_phrase_len)
    if long_phrase.any():
        record(long_phrase, "wx_phrase_too_long", "wx_phrase")
    long_uv = out["uv_desc"].notna() & (out["uv_desc"].str.len() > max_phrase_len)
    if long_uv.any():
        record(long_uv, "uv_desc_too_long", "uv_desc")

    out["_san_score"] = out[NUMERIC_SCORE_COLUMNS].notna().sum(axis=1).astype(int)
    out["_row_order"] = np.arange(row_order_start, row_order_start + len(out), dtype=np.int64)
    next_row_order = int(row_order_start + len(out))

    if triggered is not None:
        out["triggered_rules"] = [";".join(x) for x in triggered]
    if emit_flags:
        for k, arr in flags.items():
            out[k] = arr

    meta = {
        "rule_counts": dict(rule_counts),
        "column_invalid_counts": dict(col_invalid),
        "station_rule_counts": {k: dict(v) for k, v in station_rules.items()},
        "minute_histogram": dict(minute_hist),
        "rows_in": int(len(df)),
        "rows_out": int(len(out)),
        "rows_dropped": int(dropped_rows),
        "dropped_invalid_timestamp": int(dropped_invalid_ts),
        "dropped_unknown_station": int(dropped_unknown_station),
        "dropped_off_grid_timestamp": int(dropped_off_grid),
        "pressure_unit_detected": pressure_unit,
        "clds_unknown_top": dict(clds_unknown.most_common(50)),
        "next_row_order": next_row_order,
    }
    return out, meta


def dedupe_sanitized_dataframe(df: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, dict[str, int]]:
    if policy not in {"none", "first", "best_non_null"}:
        raise ValueError(f"Unsupported dedupe policy: {policy}")
    if policy == "none":
        return df.copy(), {"duplicate_rows_seen": 0, "duplicate_groups": 0, "rows_dropped_by_dedupe": 0}

    key = ["request_location_id", "valid_time_utc"]
    dup = df.duplicated(subset=key, keep=False)
    dup_rows = int(dup.sum())
    dup_groups = int(df.loc[dup, key].drop_duplicates().shape[0]) if dup_rows else 0

    if policy == "first":
        out = df.sort_values(["_row_order"], ascending=[True]).drop_duplicates(subset=key, keep="first")
    else:
        out = df.sort_values(["_san_score", "_row_order"], ascending=[False, True]).drop_duplicates(subset=key, keep="first")
    out = out.sort_values(["request_location_id", "valid_time_utc", "_row_order"]).reset_index(drop=True)
    return out, {
        "duplicate_rows_seen": dup_rows,
        "duplicate_groups": dup_groups,
        "rows_dropped_by_dedupe": int(len(df) - len(out)),
    }


def _write_csv(df: pd.DataFrame, path: Path, *, mode: str, header: bool, compression: str) -> None:
    if compression == "gzip":
        df.to_csv(path, index=False, mode=mode, header=header, compression="gzip")
    else:
        df.to_csv(path, index=False, mode=mode, header=header)


def _profile_output(output_path: Path, chunksize: int) -> dict[str, Any]:
    null_after = Counter()
    cat_top = {c: Counter() for c in CATEGORICAL_COLUMNS}
    num = {c: {"count": 0, "sum": 0.0, "min": np.inf, "max": -np.inf} for c in NUMERIC_COLUMNS}
    st_rows = Counter()
    st_missing: dict[str, Counter[str]] = defaultdict(Counter)

    for chunk in pd.read_csv(output_path, chunksize=chunksize, low_memory=False, compression="infer"):
        for c in REQUIRED_COLUMNS:
            if c in chunk.columns:
                null_after[c] += int(chunk[c].isna().sum())
        for c in NUMERIC_COLUMNS:
            if c in chunk.columns:
                v = pd.to_numeric(chunk[c], errors="coerce")
                ok = v[np.isfinite(v.to_numpy(dtype=float))]
                num[c]["count"] += int(ok.shape[0])
                if not ok.empty:
                    num[c]["sum"] += float(ok.sum())
                    num[c]["min"] = min(num[c]["min"], float(ok.min()))
                    num[c]["max"] = max(num[c]["max"], float(ok.max()))
        for c in CATEGORICAL_COLUMNS:
            if c in chunk.columns:
                cat_top[c].update(chunk[c].astype("string").fillna("MISSING").value_counts().to_dict())

        st = chunk["request_location_id"].astype("string").fillna("MISSING").str.upper()
        st_rows.update(st.value_counts().to_dict())
        for c in NUMERIC_COLUMNS:
            if c in chunk.columns:
                miss = pd.to_numeric(chunk[c], errors="coerce").isna()
                for sid, cnt in st[miss].value_counts().to_dict().items():
                    st_missing[str(sid)][c] += int(cnt)

    num_after = {}
    for c, s in num.items():
        if s["count"] == 0:
            num_after[c] = {"count": 0, "mean": None, "min": None, "max": None}
        else:
            num_after[c] = {"count": s["count"], "mean": s["sum"] / s["count"], "min": s["min"], "max": s["max"]}

    per_station = {}
    for sid, rows in st_rows.items():
        miss_pct = {c: (st_missing[sid][c] / rows if rows else 0.0) for c in NUMERIC_COLUMNS}
        per_station[sid] = {"rows_output": int(rows), "missing_pct_numeric_after": miss_pct}

    return {
        "per_column_null_after": dict(null_after),
        "numeric_after": num_after,
        "categorical_top_after": {k: dict(v.most_common(50)) for k, v in cat_top.items()},
        "per_station_summary": per_station,
    }


def _schema_profile_required_columns(path: str | Path | None) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Schema profile not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    for key in ("columns", "required_columns", "schema_columns"):
        vals = obj.get(key)
        if isinstance(vals, list):
            return {str(x) for x in vals}
    return set()


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sanitize weather observations CSV.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--report-out", required=True)
    p.add_argument("--samples-out", required=True)
    p.add_argument("--station-universe", default="")
    p.add_argument("--schema-profile", default="")
    p.add_argument("--config", default="")
    p.add_argument("--chunksize", type=int, default=250000)
    p.add_argument("--compression", choices=["gzip", "none"], default="gzip")
    p.add_argument("--emit-flags", default="false")
    p.add_argument("--drop-invalid-timestamps", default="true")
    p.add_argument("--dedupe-policy", choices=["none", "first", "best_non_null"], default="best_non_null")
    p.add_argument("--strict-columns", default="true")
    p.add_argument("--allow-extra-columns", default="false")
    p.add_argument("--enforce-30m-grid", default="false")
    p.add_argument("--fill-wdir-from-cardinal", default="false")
    p.add_argument("--manifest-path", default="data_sanitizer_manifest.jsonl")
    p.add_argument("--max-samples", type=int, default=50000)
    return p


def run_cli(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report_out).resolve()
    samples_path = Path(args.samples_out).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rules = load_rules(args.config or None)
    station_universe = read_station_universe(args.station_universe or None)
    emit_flags = _as_bool(args.emit_flags)
    drop_invalid_ts = _as_bool(args.drop_invalid_timestamps)
    strict_columns = _as_bool(args.strict_columns)
    allow_extra = _as_bool(args.allow_extra_columns)
    enforce_30m_grid = _as_bool(args.enforce_30m_grid)
    fill_wdir_from_cardinal = _as_bool(args.fill_wdir_from_cardinal)
    chunksize = int(args.chunksize)
    max_samples = int(args.max_samples)
    schema_cols = _schema_profile_required_columns(args.schema_profile or None)

    in_hash = _sha256_file(input_path)
    input_rows = 0
    rows_pre_dedupe = 0
    dropped_rows = 0
    before_null = Counter()
    rules_total = Counter()
    col_invalid = Counter()
    st_rules: dict[str, Counter[str]] = defaultdict(Counter)
    minute_hist = Counter()
    clds_unknown = Counter()
    pressure_units = Counter()
    sample_frames: list[pd.DataFrame] = []
    row_order_start = 0

    tmp_fd, tmp_name = tempfile.mkstemp(prefix="data_sanitizer_", suffix=".csv")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    tmp_header = False
    out_header = False

    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunksize, low_memory=False), start=1):
        if i == 1 and strict_columns:
            cols = list(chunk.columns)
            req = set(REQUIRED_COLUMNS).union(schema_cols)
            missing = sorted(req.difference(cols))
            extra = sorted(set(cols).difference(req))
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            if extra and not allow_extra:
                raise ValueError(f"Unexpected columns in strict mode: {extra}")

        input_rows += int(len(chunk))
        for c in REQUIRED_COLUMNS:
            if c in chunk.columns:
                before_null[c] += int(chunk[c].isna().sum())

        sane, meta = sanitize_observations_dataframe(
            chunk,
            rules=rules,
            station_universe=station_universe,
            emit_flags=emit_flags,
            drop_invalid_timestamps=drop_invalid_ts,
            fill_wdir_from_cardinal=fill_wdir_from_cardinal,
            enforce_30m_grid=enforce_30m_grid,
            row_order_start=row_order_start,
            collect_triggered_rules=True,
        )
        row_order_start = int(meta["next_row_order"])
        rows_pre_dedupe += int(meta["rows_out"])
        dropped_rows += int(meta["rows_dropped"])
        rules_total.update(meta.get("rule_counts", {}))
        col_invalid.update(meta.get("column_invalid_counts", {}))
        for sid, d in meta.get("station_rule_counts", {}).items():
            st_rules[sid].update(d)
        minute_hist.update(meta.get("minute_histogram", {}))
        clds_unknown.update(meta.get("clds_unknown_top", {}))
        pressure_units.update({str(meta.get("pressure_unit_detected", "unknown")): 1})

        trig = sane["triggered_rules"].astype(str).str.len() > 0
        if trig.any() and sum(len(x) for x in sample_frames) < max_samples:
            sample_frames.append(sane.loc[trig].copy())

        if args.dedupe_policy == "none":
            keep = [c for c in sane.columns if c not in {"_san_score", "_row_order"}]
            _write_csv(sane[keep], output_path, mode="a", header=not out_header, compression=args.compression)
            out_header = True
        else:
            _write_csv(sane, tmp_path, mode="a", header=not tmp_header, compression="none")
            tmp_header = True

    dedupe_stats = {"duplicate_rows_seen": 0, "duplicate_groups": 0, "rows_dropped_by_dedupe": 0}
    if args.dedupe_policy != "none":
        all_df = pd.read_csv(tmp_path, low_memory=False)
        deduped, dedupe_stats = dedupe_sanitized_dataframe(all_df, args.dedupe_policy)
        keep = [c for c in deduped.columns if c not in {"_san_score", "_row_order"}]
        if not emit_flags:
            keep = [c for c in keep if not c.endswith("_was_sanitized")]
        deduped[keep].to_csv(output_path, index=False, compression=("gzip" if args.compression == "gzip" else None))

    if tmp_path.exists():
        tmp_path.unlink()

    out_hash = _sha256_file(output_path)
    profile = _profile_output(output_path, chunksize=max(10000, min(200000, chunksize)))
    output_rows = int(sum(v["rows_output"] for v in profile["per_station_summary"].values()))

    if sample_frames:
        samples = pd.concat(sample_frames, ignore_index=True).drop_duplicates().head(max_samples)
    else:
        samples = pd.DataFrame(columns=REQUIRED_COLUMNS + ["triggered_rules"])
    samples.to_csv(samples_path, index=False)

    per_col = {}
    for c in REQUIRED_COLUMNS:
        per_col[c] = {
            "null_count_before": int(before_null.get(c, 0)),
            "null_count_after": int(profile["per_column_null_after"].get(c, 0)),
            "invalid_count": int(col_invalid.get(c, 0)),
        }
        if c in profile["numeric_after"]:
            per_col[c]["numeric_after"] = profile["numeric_after"][c]
        if c in profile["categorical_top_after"]:
            per_col[c]["top_categories_after"] = profile["categorical_top_after"][c]

    for sid, info in profile["per_station_summary"].items():
        info["major_rule_hits"] = {
            k: int(v)
            for k, v in st_rules.get(sid, {}).items()
            if k in {"temp_out_of_bounds", "wdir_sentinel_999", "uv_negative_sentinel", "wspd_out_of_bounds"}
        }

    report = {
        "run_metadata": {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "report_out": str(report_path),
            "samples_out": str(samples_path),
            "run_utc": _utc_now_iso(),
            "chunksize": chunksize,
            "compression": args.compression,
            "dedupe_policy": args.dedupe_policy,
            "drop_invalid_timestamps": drop_invalid_ts,
            "strict_columns": strict_columns,
            "emit_flags": emit_flags,
            "git_commit_hash": _git_hash(),
        },
        "row_counts": {
            "input_rows": int(input_rows),
            "rows_after_sanitize_pre_dedupe": int(rows_pre_dedupe),
            "rows_dropped_total_sanitize": int(dropped_rows),
            "rows_output": int(output_rows),
            **{k: int(v) for k, v in dedupe_stats.items()},
        },
        "hashes": {"input_sha256": in_hash, "output_sha256": out_hash},
        "rules": {
            "hit_counts": {k: int(v) for k, v in rules_total.items()},
            "pressure_unit_detected_windows": {k: int(v) for k, v in pressure_units.items()},
            "minute_histogram": {k: int(v) for k, v in minute_hist.items()},
            "clds_unknown_top": {k: int(v) for k, v in clds_unknown.most_common(50)},
        },
        "per_column": per_col,
        "per_station": profile["per_station_summary"],
        "runtime_seconds": float(time.perf_counter() - t0),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    manifest_row = {
        "run_timestamp_utc": _utc_now_iso(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "report_path": str(report_path),
        "samples_path": str(samples_path),
        "input_sha256": in_hash,
        "output_sha256": out_hash,
        "input_rows": int(input_rows),
        "output_rows": int(output_rows),
        "dedupe_policy": args.dedupe_policy,
        "runtime_seconds": float(time.perf_counter() - t0),
    }
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(manifest_row) + "\n")

    return report


def main() -> int:
    args = _parser().parse_args()
    report = run_cli(args)
    print(json.dumps(report["row_counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
