from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Bucket:
    label_raw: str
    lo: Optional[int]
    hi: Optional[int]
    mode: str  # range | or_below | or_above

    def contains(self, temp_f: int) -> bool:
        if self.mode == "or_below" and self.hi is not None:
            return temp_f <= self.hi
        if self.mode == "or_above" and self.lo is not None:
            return temp_f >= self.lo
        if self.mode == "range" and self.lo is not None and self.hi is not None:
            return self.lo <= temp_f <= self.hi
        return False

    def canonical_label(self) -> str:
        if self.mode == "or_below" and self.hi is not None:
            return f"{self.hi}F or below"
        if self.mode == "or_above" and self.lo is not None:
            return f"{self.lo}F or above"
        if self.mode == "range" and self.lo is not None and self.hi is not None:
            return f"{self.lo}F to {self.hi}F"
        return self.label_raw


def normalize_price(v: float) -> float:
    if pd.isna(v):
        return np.nan
    x = float(v)
    if x < 0:
        return np.nan
    if x > 1.0:
        x = x / 100.0
    return float(np.clip(x, 0.0, 1.0))


def parse_bucket_label(label: str) -> Optional[Bucket]:
    s = str(label).strip().lower().replace(" to ", "-")
    s = re.sub(r"\s+", " ", s)
    lt_match = re.search(r"<\s*(\d+)", s)
    if lt_match:
        bound = int(lt_match.group(1)) - 1
        return Bucket(label_raw=str(label), lo=None, hi=bound, mode="or_below")
    le_match = re.search(r"<=\s*(\d+)", s)
    if le_match:
        return Bucket(label_raw=str(label), lo=None, hi=int(le_match.group(1)), mode="or_below")
    gt_match = re.search(r">\s*(\d+)", s)
    if gt_match:
        bound = int(gt_match.group(1)) + 1
        return Bucket(label_raw=str(label), lo=bound, hi=None, mode="or_above")
    ge_match = re.search(r">=\s*(\d+)", s)
    if ge_match:
        return Bucket(label_raw=str(label), lo=int(ge_match.group(1)), hi=None, mode="or_above")
    nums = [int(x) for x in re.findall(r"\d+", s)]
    if ("or below" in s or "or less" in s) and nums:
        return Bucket(label_raw=str(label), lo=None, hi=nums[0], mode="or_below")
    if ("or above" in s or "or higher" in s) and nums:
        return Bucket(label_raw=str(label), lo=nums[0], hi=None, mode="or_above")
    if len(nums) >= 2:
        lo, hi = sorted([nums[0], nums[1]])
        return Bucket(label_raw=str(label), lo=lo, hi=hi, mode="range")
    return None


def cdf_from_quantiles(qmap: Dict[float, float], x: float) -> float:
    taus = np.array(sorted(qmap.keys()), dtype=float)
    qvals = np.array([qmap[t] for t in taus], dtype=float)
    qvals = np.maximum.accumulate(qvals)
    return float(np.interp(x, qvals, taus, left=0.0, right=1.0))


def pmf_int_from_quantiles(qmap: Dict[float, float], support_lo: int = -20, support_hi: int = 130) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for t in range(support_lo, support_hi + 1):
        p = cdf_from_quantiles(qmap, t + 0.5) - cdf_from_quantiles(qmap, t - 0.5)
        out[t] = max(0.0, float(p))
    total = float(sum(out.values()))
    if total <= 0:
        w = support_hi - support_lo + 1
        return {t: 1.0 / w for t in range(support_lo, support_hi + 1)}
    return {k: v / total for k, v in out.items()}


def bucket_prob(pmf: Dict[int, float], b: Bucket) -> float:
    if b.mode == "or_below" and b.hi is not None:
        return float(sum(v for k, v in pmf.items() if k <= b.hi))
    if b.mode == "or_above" and b.lo is not None:
        return float(sum(v for k, v in pmf.items() if k >= b.lo))
    if b.mode == "range" and b.lo is not None and b.hi is not None:
        return float(sum(v for k, v in pmf.items() if b.lo <= k <= b.hi))
    return 0.0


def safe_iso_utc(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).isoformat().replace("+00:00", "Z")


def market_file_date_from_path(path: Path) -> Optional[str]:
    m = re.search(r"_(\d{8})\.csv$", str(path.name))
    if not m:
        return None
    ymd = m.group(1)
    return f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"


def compute_entry_cutoff_utc(target_date_local: pd.Timestamp, entry_hour_z: int, entry_minute_z: int) -> pd.Timestamp:
    return (target_date_local - pd.Timedelta(days=1) + pd.Timedelta(hours=entry_hour_z, minutes=entry_minute_z)).tz_localize(
        "UTC"
    )


def parse_station_ids(value: str) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in str(value or "").split(","):
        sid = raw.strip().upper()
        if not sid or sid in seen:
            continue
        seen.add(sid)
        out.append(sid)
    return out


def parse_json_mapping(value: Optional[str]) -> Dict[str, str]:
    if value is None:
        return {}
    txt = str(value).strip()
    if not txt:
        return {}
    path = Path(txt)
    if path.exists():
        txt = path.read_text(encoding="utf-8")
    payload = json.loads(txt)
    if not isinstance(payload, dict):
        raise ValueError("JSON mapping must be an object")
    out: Dict[str, str] = {}
    for k, v in payload.items():
        key = str(k).strip().upper()
        val = str(v).strip()
        if key and val:
            out[key] = val
    return out


def load_predictions(dev_path: Path, test_path: Path) -> Dict[str, pd.Series]:
    dev = pd.read_parquet(dev_path)
    test = pd.read_parquet(test_path)
    pred = pd.concat([dev, test], ignore_index=True)
    pred["target_date_local"] = pd.to_datetime(pred["target_date_local"]).dt.normalize()
    pred = pred.drop_duplicates(subset=["target_date_local"], keep="last").sort_values("target_date_local")
    return {pd.Timestamp(r["target_date_local"]).strftime("%Y-%m-%d"): r for _, r in pred.iterrows()}


def load_truth_map(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    date_col = "date" if "date" in df.columns else ("target_date_local" if "target_date_local" in df.columns else None)
    value_col = (
        "settled_tmax"
        if "settled_tmax" in df.columns
        else ("y_tmax" if "y_tmax" in df.columns else None)
    )
    if date_col is None or value_col is None:
        raise ValueError(f"Truth CSV missing required columns in {path}. expected date/date_local and settled_tmax/y_tmax.")
    df["target_date_local"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df["y_tmax"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["target_date_local", "y_tmax"]).copy()
    out: Dict[str, float] = {}
    for _, r in df.iterrows():
        out[pd.Timestamp(r["target_date_local"]).strftime("%Y-%m-%d")] = float(r["y_tmax"])
    return out


def _extract_live_report_block(report: Dict, station_id: str) -> Dict:
    report_blocks = report.get("inference_by_station", {}) if isinstance(report.get("inference_by_station"), dict) else {}
    block = report_blocks.get(station_id, {})
    if not block:
        block = report.get(f"inference_{station_id.lower()}", {})
    return block if isinstance(block, dict) else {}


def _missing_live_report_stations(report: Dict, station_ids: List[str]) -> List[str]:
    missing: List[str] = []
    for station_id in station_ids:
        block = _extract_live_report_block(report, station_id)
        quantiles = block.get("quantiles", {}) if isinstance(block, dict) else {}
        if not isinstance(quantiles, dict) or not quantiles:
            missing.append(station_id)
    return missing


def run_live_inference_for_target(
    target_day: pd.Timestamp,
    live_script_path: Path,
    live_root: Path,
    python_bin: str,
    script_log_level: str,
    station_ids: List[str],
) -> Dict:
    day_key = pd.Timestamp(target_day).strftime("%Y-%m-%d")
    out_dir = live_root / f"target_{pd.Timestamp(target_day).strftime('%Y%m%d')}"
    report_path = out_dir / "inference_report.json"
    rerun_reason = ""
    if report_path.exists():
        try:
            existing_report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            existing_report = {}
        missing_station_ids = _missing_live_report_stations(existing_report, station_ids)
        if missing_station_ids:
            rerun_reason = f"missing_station_blocks:{','.join(missing_station_ids)}"
    else:
        rerun_reason = "report_missing"

    if rerun_reason:
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            python_bin,
            str(live_script_path),
            "--target-date",
            day_key,
            "--out-dir",
            str(out_dir),
            "--log-level",
            str(script_log_level),
            "--stdout-json",
            "summary",
        ]
        completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            failure_path = out_dir / "runtime_gate_failure.json"
            failure_excerpt = ""
            if failure_path.exists():
                try:
                    payload = json.loads(failure_path.read_text(encoding="utf-8"))
                    failure_excerpt = json.dumps(
                        {
                            "error": payload.get("error"),
                            "message": payload.get("message"),
                            "lagging_slices": payload.get("lagging_slices", []),
                        },
                        indent=2,
                    )
                except Exception:
                    failure_excerpt = failure_path.read_text(encoding="utf-8")[:2000]
            stderr_tail = (completed.stderr or "")[-2000:]
            stdout_tail = (completed.stdout or "")[-2000:]
            raise RuntimeError(
                f"Live inference script failed for target={day_key} code={completed.returncode}\n"
                f"stderr_tail=\n{stderr_tail}\nstdout_tail=\n{stdout_tail}\n"
                f"runtime_gate_failure_excerpt=\n{failure_excerpt}"
            )
    if not report_path.exists():
        raise FileNotFoundError(f"Missing live inference report after run for target={day_key}: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    missing_station_ids = _missing_live_report_stations(report, station_ids)
    if missing_station_ids:
        raise RuntimeError(
            f"Live inference report missing requested stations for target={day_key}: {','.join(missing_station_ids)}"
        )
    return report


def load_predictions_from_live_script(
    start_date: str,
    end_date: str,
    live_script_path: Path,
    live_root: Path,
    python_bin: str,
    script_log_level: str,
    truth_maps: Dict[str, Dict[str, float]],
    station_ids: List[str],
) -> Tuple[Dict[str, Dict[str, Dict]], Dict]:
    days = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")
    pred_maps: Dict[str, Dict[str, Dict]] = {sid: {} for sid in station_ids}
    stats = {
        "days_requested": int(len(days)),
        "days_with_live_report": 0,
        "days_live_inference_failed": 0,
        "days_missing_truth_any_station": 0,
        "station_prediction_rows": {sid: 0 for sid in station_ids},
        "failed_live_inference_days": [],
        "live_script_path": str(live_script_path),
        "live_root": str(live_root),
    }

    for i, day in enumerate(days, start=1):
        day_key = day.strftime("%Y-%m-%d")
        try:
            report = run_live_inference_for_target(
                target_day=day,
                live_script_path=live_script_path,
                live_root=live_root,
                python_bin=python_bin,
                script_log_level=script_log_level,
                station_ids=station_ids,
            )
        except Exception as exc:
            stats["days_live_inference_failed"] += 1
            if len(stats["failed_live_inference_days"]) < 200:
                stats["failed_live_inference_days"].append(
                    {"target_date_local": day_key, "error": str(exc)}
                )
            print(json.dumps({"warning": "live_inference_failed", "target_date_local": day_key}))
            continue
        stats["days_with_live_report"] += 1

        for station_id in station_ids:
            block = _extract_live_report_block(report, station_id)
            q = block.get("quantiles", {})
            if not q:
                continue
            y = truth_maps[station_id].get(day_key)
            if y is None or not math.isfinite(float(y)):
                stats["days_missing_truth_any_station"] += 1
                continue
            pred_maps[station_id][day_key] = {
                "target_date_local": day_key,
                "y_tmax": float(y),
                "q_0.05": float(q["q_0.05"]),
                "q_0.10": float(q["q_0.10"]),
                "q_0.25": float(q["q_0.25"]),
                "q_0.50": float(q["q_0.50"]),
                "q_0.75": float(q["q_0.75"]),
                "q_0.90": float(q["q_0.90"]),
                "q_0.95": float(q["q_0.95"]),
            }
            stats["station_prediction_rows"][station_id] += 1

        if i % 25 == 0 or i == len(days):
            print(
                json.dumps(
                    {
                        "progress": f"{i}/{len(days)}",
                        "last_target_date": day_key,
                        "station_prediction_rows": stats["station_prediction_rows"],
                    }
                )
            )

    return pred_maps, stats


def build_market_index(root: Path, file_prefix: str) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for p in root.rglob(f"{file_prefix.upper()}_*.csv"):
        m = re.match(rf"^{file_prefix.upper()}_(\d{{8}})\.csv$", p.name)
        if m:
            idx[m.group(1)] = p
    return idx


def load_market_rows_after_gate(path: Path, gate: pd.Timestamp) -> Tuple[Optional[pd.DataFrame], Optional[pd.Timestamp]]:
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        return None, None
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.empty:
        return None, None
    open_ts = pd.Timestamp(df["timestamp"].iloc[0])
    after = df[df["timestamp"] >= gate].copy()
    if after.empty:
        return None, open_ts
    after = after.groupby("timestamp", as_index=False).last()
    return after, open_ts


def candidate_sort_key(c: Dict) -> Tuple:
    return (
        -float(c["model_win_prob"]),
        -float(c["ev"]),
        float(c["market_price"]),
        str(c["station_id"]),
        str(c["bucket_raw"]),
        str(c["side"]),
    )


def build_day_contexts(
    day: pd.Timestamp,
    station_ids: List[str],
    pred_maps: Dict[str, Dict[str, pd.Series]],
    market_indices: Dict[str, Dict[str, Path]],
    entry_hour_z: int,
    entry_minute_z: int,
    min_entry_minutes_after_open: int,
) -> Tuple[List[Dict], Dict]:
    day_key = day.strftime("%Y-%m-%d")
    ymd = day.strftime("%Y%m%d")
    gate = compute_entry_cutoff_utc(day, entry_hour_z, entry_minute_z)
    contexts: List[Dict] = []
    status = {"day": day_key, "gate_cutoff_utc": safe_iso_utc(gate), "station_status": {}}

    for station_id in station_ids:
        st = {
            "has_prediction": False,
            "has_market_file": False,
            "has_rows_after_gate": False,
            "has_rows_after_effective_cutoff": False,
            "market_file": None,
            "market_open_utc": None,
            "effective_cutoff_utc": None,
        }
        pred = pred_maps[station_id].get(day_key)
        if pred is None:
            status["station_status"][station_id] = st
            continue
        st["has_prediction"] = True
        fpath = market_indices[station_id].get(ymd)
        if fpath is None:
            status["station_status"][station_id] = st
            continue
        st["has_market_file"] = True
        st["market_file"] = str(fpath)
        rows_after_gate, open_ts = load_market_rows_after_gate(fpath, gate)
        st["market_open_utc"] = safe_iso_utc(open_ts) if open_ts is not None else None
        if rows_after_gate is None or rows_after_gate.empty:
            status["station_status"][station_id] = st
            continue
        st["has_rows_after_gate"] = True

        open_delay_cutoff = (pd.Timestamp(open_ts) + pd.Timedelta(minutes=int(min_entry_minutes_after_open))) if open_ts is not None else gate
        effective_cutoff = max(gate, open_delay_cutoff)
        st["effective_cutoff_utc"] = safe_iso_utc(effective_cutoff)
        rows_after_effective = rows_after_gate[rows_after_gate["timestamp"] >= effective_cutoff].copy()
        if rows_after_effective.empty:
            status["station_status"][station_id] = st
            continue
        st["has_rows_after_effective_cutoff"] = True
        status["station_status"][station_id] = st

        qmap = {
            0.05: float(pred["q_0.05"]),
            0.10: float(pred["q_0.10"]),
            0.25: float(pred["q_0.25"]),
            0.50: float(pred["q_0.50"]),
            0.75: float(pred["q_0.75"]),
            0.90: float(pred["q_0.90"]),
            0.95: float(pred["q_0.95"]),
        }
        contexts.append(
            {
                "station_id": station_id,
                "target_date_local": day,
                "market_file_date_local": market_file_date_from_path(fpath),
                "y_tmax": int(round(float(pred["y_tmax"]))),
                "pmf": pmf_int_from_quantiles(qmap),
                "market_file": fpath,
                "market_open_utc": open_ts if open_ts is not None else gate,
                "gate_cutoff_utc": gate,
                "effective_cutoff_utc": effective_cutoff,
                "rows_after_gate": rows_after_effective,
            }
        )
    return contexts, status


def candidates_at_timestamp(
    ts: pd.Timestamp,
    contexts: Iterable[Dict],
    ev_min: float,
    win_min: float,
    min_market_price: float,
) -> List[Dict]:
    out: List[Dict] = []
    for ctx in contexts:
        row_df = ctx["rows_after_gate"][ctx["rows_after_gate"]["timestamp"] == ts]
        if row_df.empty:
            continue
        row = row_df.iloc[0]
        for col in ctx["rows_after_gate"].columns:
            if col == "timestamp":
                continue
            bucket = parse_bucket_label(col)
            if bucket is None:
                continue
            p_yes_mkt = normalize_price(row[col])
            if not np.isfinite(p_yes_mkt):
                continue
            p_yes_model = bucket_prob(ctx["pmf"], bucket)
            p_no_model = 1.0 - p_yes_model
            p_no_mkt = 1.0 - p_yes_mkt

            ev_yes = p_yes_model - p_yes_mkt
            if p_yes_mkt >= min_market_price and p_yes_model >= win_min and ev_yes >= ev_min:
                out.append(
                    {
                        "target_date_local": ctx["target_date_local"].strftime("%Y-%m-%d"),
                        "market_file_date_local": str(ctx.get("market_file_date_local") or ""),
                        "station_id": ctx["station_id"],
                        "entry_timestamp_utc": safe_iso_utc(ts),
                        "market_open_utc": safe_iso_utc(ctx["market_open_utc"]),
                        "gate_cutoff_utc": safe_iso_utc(ctx["gate_cutoff_utc"]),
                        "effective_cutoff_utc": safe_iso_utc(ctx["effective_cutoff_utc"]),
                        "market_file": str(ctx["market_file"]),
                        "bucket_raw": col,
                        "bucket": bucket.canonical_label(),
                        "side": "YES",
                        "market_price": float(p_yes_mkt),
                        "model_win_prob": float(p_yes_model),
                        "ev": float(ev_yes),
                        "y_tmax": int(ctx["y_tmax"]),
                        "win": int(bucket.contains(int(ctx["y_tmax"]))),
                    }
                )

            ev_no = p_no_model - p_no_mkt
            if p_no_mkt >= min_market_price and p_no_model >= win_min and ev_no >= ev_min:
                out.append(
                    {
                        "target_date_local": ctx["target_date_local"].strftime("%Y-%m-%d"),
                        "market_file_date_local": str(ctx.get("market_file_date_local") or ""),
                        "station_id": ctx["station_id"],
                        "entry_timestamp_utc": safe_iso_utc(ts),
                        "market_open_utc": safe_iso_utc(ctx["market_open_utc"]),
                        "gate_cutoff_utc": safe_iso_utc(ctx["gate_cutoff_utc"]),
                        "effective_cutoff_utc": safe_iso_utc(ctx["effective_cutoff_utc"]),
                        "market_file": str(ctx["market_file"]),
                        "bucket_raw": col,
                        "bucket": bucket.canonical_label(),
                        "side": "NO",
                        "market_price": float(p_no_mkt),
                        "model_win_prob": float(p_no_model),
                        "ev": float(ev_no),
                        "y_tmax": int(ctx["y_tmax"]),
                        "win": int(not bucket.contains(int(ctx["y_tmax"]))),
                    }
                )
    return out


def select_trade_for_day(
    day: pd.Timestamp,
    station_ids: List[str],
    pred_maps: Dict[str, Dict[str, pd.Series]],
    market_indices: Dict[str, Dict[str, Path]],
    ev_min: float,
    win_min: float,
    min_market_price: float,
    entry_hour_z: int,
    entry_minute_z: int,
    min_entry_minutes_after_open: int,
) -> Tuple[Optional[Dict], Dict]:
    contexts, status = build_day_contexts(
        day,
        station_ids,
        pred_maps,
        market_indices,
        entry_hour_z,
        entry_minute_z,
        min_entry_minutes_after_open,
    )
    status["has_any_station_context"] = len(contexts) > 0
    if not contexts:
        status["first_eligible_timestamp_utc"] = None
        status["chosen_trade_key"] = None
        return None, status

    ts_all: List[pd.Timestamp] = []
    for ctx in contexts:
        ts_all.extend(list(pd.to_datetime(ctx["rows_after_gate"]["timestamp"], utc=True)))
    if not ts_all:
        status["first_eligible_timestamp_utc"] = None
        status["chosen_trade_key"] = None
        return None, status
    for ts in sorted(set(pd.Timestamp(x) for x in ts_all)):
        eligible = candidates_at_timestamp(ts, contexts, ev_min, win_min, min_market_price)
        if not eligible:
            continue
        chosen = sorted(eligible, key=candidate_sort_key)[0]
        chosen["first_eligible_timestamp_utc"] = safe_iso_utc(ts)
        chosen["eligible_count_at_entry_timestamp"] = int(len(eligible))
        chosen["arbitration_policy"] = "model_win_prob desc, ev desc, market_price asc, station_id asc"
        status["first_eligible_timestamp_utc"] = safe_iso_utc(ts)
        status["chosen_trade_key"] = {
            "station_id": chosen["station_id"],
            "bucket_raw": chosen["bucket_raw"],
            "side": chosen["side"],
            "entry_timestamp_utc": chosen["entry_timestamp_utc"],
        }
        return chosen, status
    status["first_eligible_timestamp_utc"] = None
    status["chosen_trade_key"] = None
    return None, status


def run_backtest(
    station_ids: List[str],
    pred_maps: Dict[str, Dict[str, pd.Series]],
    market_indices: Dict[str, Dict[str, Path]],
    start_date: str,
    end_date: str,
    ev_min: float,
    win_min: float,
    min_market_price: float,
    start_balance: float,
    risk_fraction: float,
    stake_cap_usd: float,
    entry_hour_z: int,
    entry_minute_z: int,
    min_entry_minutes_after_open: int,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    trades: List[Dict] = []
    day_debug: Dict[str, Dict] = {}
    days = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")
    counts = {
        "total_days": int(len(days)),
        "days_with_prediction_by_station": {sid: 0 for sid in station_ids},
        "days_with_any_prediction": 0,
        "days_with_market_file_by_station": {sid: 0 for sid in station_ids},
        "days_with_any_market_file": 0,
        "days_with_any_station_context": 0,
        "days_without_trade_candidate": 0,
    }

    for day in days:
        day_key = day.strftime("%Y-%m-%d")
        ymd = day.strftime("%Y%m%d")
        has_any_pred = False
        has_any_file = False
        for sid in station_ids:
            has_pred = day_key in pred_maps.get(sid, {})
            has_file = ymd in market_indices.get(sid, {})
            counts["days_with_prediction_by_station"][sid] += int(has_pred)
            counts["days_with_market_file_by_station"][sid] += int(has_file)
            has_any_pred = has_any_pred or has_pred
            has_any_file = has_any_file or has_file
        counts["days_with_any_prediction"] += int(has_any_pred)
        counts["days_with_any_market_file"] += int(has_any_file)

        chosen, dbg = select_trade_for_day(
            day=day,
            station_ids=station_ids,
            pred_maps=pred_maps,
            market_indices=market_indices,
            ev_min=ev_min,
            win_min=win_min,
            min_market_price=min_market_price,
            entry_hour_z=entry_hour_z,
            entry_minute_z=entry_minute_z,
            min_entry_minutes_after_open=min_entry_minutes_after_open,
        )
        day_debug[day_key] = dbg
        counts["days_with_any_station_context"] += int(dbg.get("has_any_station_context", False))
        if chosen is None:
            counts["days_without_trade_candidate"] += 1
            continue
        trades.append(chosen)

    trades_df = pd.DataFrame(trades)
    bal = float(start_balance)
    peak = bal
    if not trades_df.empty:
        trades_df = trades_df.sort_values(["target_date_local", "entry_timestamp_utc"]).reset_index(drop=True)
        for i, r in trades_df.iterrows():
            stake = min(bal * risk_fraction, stake_cap_usd)
            price = float(r["market_price"])
            shares = stake / price if price > 0 else 0.0
            pnl = shares * (1.0 - price) if int(r["win"]) == 1 else -stake
            bal_before = bal
            bal = bal + pnl
            peak = max(peak, bal)
            dd = 0.0 if peak <= 0 else (peak - bal) / peak
            trades_df.loc[i, "stake"] = stake
            trades_df.loc[i, "shares"] = shares
            trades_df.loc[i, "pnl"] = pnl
            trades_df.loc[i, "balance_before"] = bal_before
            trades_df.loc[i, "balance_after"] = bal
            trades_df.loc[i, "drawdown"] = dd
            trades_df.loc[i, "result"] = "W" if int(r["win"]) == 1 else "L"

        entry_days = pd.to_datetime(trades_df["entry_timestamp_utc"], utc=True).dt.tz_convert(ZoneInfo("Europe/Stockholm")).dt.date
        entry_day_counts = entry_days.value_counts()
        days_with_multi_entry = int((entry_day_counts > 1).sum())
        max_trades_per_entry_day = int(entry_day_counts.max()) if not entry_day_counts.empty else 0
    else:
        days_with_multi_entry = 0
        max_trades_per_entry_day = 0

    wins = trades_df[trades_df["win"] == 1] if not trades_df.empty else pd.DataFrame()
    losses = trades_df[trades_df["win"] == 0] if not trades_df.empty else pd.DataFrame()
    gp = float(wins["pnl"].sum()) if not wins.empty else 0.0
    gl = float(-losses["pnl"].sum()) if not losses.empty else 0.0
    summary = {
        "period_start": str(start_date),
        "period_end": str(end_date),
        "entry_hour_z": int(entry_hour_z),
        "entry_minute_z": int(entry_minute_z),
        "min_entry_minutes_after_open": int(min_entry_minutes_after_open),
        "ev_min": float(ev_min),
        "win_min": float(win_min),
        "min_market_price": float(min_market_price),
        "risk_fraction": float(risk_fraction),
        "stake_cap_usd": float(stake_cap_usd),
        **counts,
        "trades": int(len(trades_df)),
        "wins": int((trades_df["win"] == 1).sum()) if not trades_df.empty else 0,
        "losses": int((trades_df["win"] == 0).sum()) if not trades_df.empty else 0,
        "win_rate": float(trades_df["win"].mean()) if not trades_df.empty else 0.0,
        "profit_factor": float(gp / gl) if gl > 0 else None,
        "start_balance": float(start_balance),
        "final_balance": float(bal),
        "total_pnl": float(bal - start_balance),
        "avg_ev_at_trade": float(trades_df["ev"].mean()) if not trades_df.empty else 0.0,
        "median_ev_at_trade": float(trades_df["ev"].median()) if not trades_df.empty else 0.0,
        "max_drawdown": float(trades_df["drawdown"].max()) if not trades_df.empty else 0.0,
        "station_counts": trades_df["station_id"].value_counts().to_dict() if not trades_df.empty else {},
        "side_counts": trades_df["side"].value_counts().to_dict() if not trades_df.empty else {},
        # Informational only: multiple target dates can legitimately enter on the same Stockholm calendar day.
        "entry_stockholm_days_with_multiple_trades": int(days_with_multi_entry),
        "max_trades_per_entry_stockholm_day": int(max_trades_per_entry_day),
    }
    return trades_df, summary, day_debug


def _eq(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


def run_sanity_audit(
    station_ids: List[str],
    trades_df: pd.DataFrame,
    pred_maps: Dict[str, Dict[str, pd.Series]],
    market_indices: Dict[str, Dict[str, Path]],
    day_debug: Dict[str, Dict],
    ev_min: float,
    win_min: float,
    min_market_price: float,
    entry_hour_z: int,
    entry_minute_z: int,
    min_entry_minutes_after_open: int,
    stake_cap_usd: float,
) -> Dict:
    failures = {
        "more_than_one_trade_per_day_global": 0,
        "entry_before_gate": 0,
        "entry_before_effective_cutoff": 0,
        "entry_not_first_eligible_timestamp_globally": 0,
        "tie_break_policy_violation": 0,
        "market_file_missing": 0,
        "market_file_date_mismatch_target_date": 0,
        "bucket_not_found": 0,
        "bucket_unparseable": 0,
        "entry_bucket_price_missing_at_timestamp": 0,
        "market_price_mismatch": 0,
        "market_price_below_min_market_price": 0,
        "model_prob_mismatch": 0,
        "ev_mismatch": 0,
        "win_label_mismatch": 0,
        "stake_cap_breach": 0,
        "pnl_mismatch": 0,
    }
    if not trades_df.empty:
        failures["more_than_one_trade_per_day_global"] = int(trades_df["target_date_local"].duplicated().sum())

    checked = 0
    for _, tr in trades_df.iterrows():
        checked += 1
        day = pd.Timestamp(tr["target_date_local"]).normalize()
        day_key = day.strftime("%Y-%m-%d")
        gate = compute_entry_cutoff_utc(day, entry_hour_z, entry_minute_z)
        entry_ts = pd.Timestamp(pd.to_datetime(tr["entry_timestamp_utc"], utc=True))
        market_open_ts = pd.Timestamp(pd.to_datetime(tr["market_open_utc"], utc=True))
        open_delay_cutoff = market_open_ts + pd.Timedelta(minutes=int(min_entry_minutes_after_open))
        effective_cutoff = max(gate, open_delay_cutoff)
        if entry_ts < gate:
            failures["entry_before_gate"] += 1
        if entry_ts < effective_cutoff:
            failures["entry_before_effective_cutoff"] += 1

        first_ts = day_debug.get(day_key, {}).get("first_eligible_timestamp_utc")
        if first_ts is None or str(first_ts) != str(tr["entry_timestamp_utc"]):
            failures["entry_not_first_eligible_timestamp_globally"] += 1

        chosen, _ = select_trade_for_day(
            day=day,
            station_ids=station_ids,
            pred_maps=pred_maps,
            market_indices=market_indices,
            ev_min=ev_min,
            win_min=win_min,
            min_market_price=min_market_price,
            entry_hour_z=entry_hour_z,
            entry_minute_z=entry_minute_z,
            min_entry_minutes_after_open=min_entry_minutes_after_open,
        )
        if chosen is None or (
            str(chosen["station_id"]),
            str(chosen["bucket_raw"]),
            str(chosen["side"]),
            str(chosen["entry_timestamp_utc"]),
        ) != (
            str(tr["station_id"]),
            str(tr["bucket_raw"]),
            str(tr["side"]),
            str(tr["entry_timestamp_utc"]),
        ):
            failures["tie_break_policy_violation"] += 1

        fpath = Path(str(tr["market_file"]))
        if not fpath.exists():
            failures["market_file_missing"] += 1
            continue
        mkt_day = market_file_date_from_path(fpath)
        if mkt_day is None or str(mkt_day) != str(tr["target_date_local"]):
            failures["market_file_date_mismatch_target_date"] += 1
        df = pd.read_csv(fpath)
        if "timestamp" not in df.columns:
            failures["market_file_missing"] += 1
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        row_df = df[(df["timestamp"] == entry_ts)]
        if row_df.empty:
            failures["market_file_missing"] += 1
            continue
        row = row_df.iloc[0]
        raw_bucket = str(tr["bucket_raw"])
        if raw_bucket not in df.columns:
            failures["bucket_not_found"] += 1
            continue
        bucket = parse_bucket_label(raw_bucket)
        if bucket is None:
            failures["bucket_unparseable"] += 1
            continue
        if pd.isna(row[raw_bucket]):
            failures["entry_bucket_price_missing_at_timestamp"] += 1
            continue
        p_yes = normalize_price(row[raw_bucket])
        p_mkt_expected = p_yes if str(tr["side"]) == "YES" else (1.0 - p_yes)
        if not _eq(float(tr["market_price"]), p_mkt_expected, tol=1e-8):
            failures["market_price_mismatch"] += 1
        if float(tr["market_price"]) + 1e-12 < float(min_market_price):
            failures["market_price_below_min_market_price"] += 1

        pred = pred_maps[str(tr["station_id"])].get(day_key)
        if pred is None:
            failures["model_prob_mismatch"] += 1
            failures["ev_mismatch"] += 1
            continue
        qmap = {0.05: float(pred["q_0.05"]), 0.10: float(pred["q_0.10"]), 0.25: float(pred["q_0.25"]), 0.50: float(pred["q_0.50"]), 0.75: float(pred["q_0.75"]), 0.90: float(pred["q_0.90"]), 0.95: float(pred["q_0.95"])}
        pmf = pmf_int_from_quantiles(qmap)
        p_yes_model = bucket_prob(pmf, bucket)
        p_model_expected = p_yes_model if str(tr["side"]) == "YES" else (1.0 - p_yes_model)
        if not _eq(float(tr["model_win_prob"]), p_model_expected, tol=1e-8):
            failures["model_prob_mismatch"] += 1
        if not _eq(float(tr["ev"]), float(p_model_expected - p_mkt_expected), tol=1e-8):
            failures["ev_mismatch"] += 1

        y = int(round(float(tr["y_tmax"])))
        win_expected = 1 if ((str(tr["side"]) == "YES" and bucket.contains(y)) or (str(tr["side"]) == "NO" and not bucket.contains(y))) else 0
        if int(tr["win"]) != win_expected:
            failures["win_label_mismatch"] += 1
        if float(tr["stake"]) > float(stake_cap_usd) + 1e-9:
            failures["stake_cap_breach"] += 1
        pnl_expected = float(tr["shares"]) * (1.0 - float(tr["market_price"])) if int(tr["win"]) == 1 else -float(tr["stake"])
        if not _eq(float(tr["pnl"]), pnl_expected, tol=1e-6):
            failures["pnl_mismatch"] += 1

    return {"checked_trades": int(checked), "passes_all_checks": not any(v > 0 for v in failures.values()), "failures": failures}


def build_sideaware_with_balance_table(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=[
            "Target date (Local)", "Market file day (Local)", "Entry date (Stockholm)", "Entry time (Stockholm)",
            "Station", "Bucket", "Bucket raw (market)", "Side", "Market win % (side)", "Model win %",
            "EV", "Amount invested ($)", "Profit made ($)", "Result", "Balance after trade ($)",
            "Market open (UTC)", "Gate cutoff (UTC)", "Effective cutoff (UTC)", "Market file"
        ])
    t = trades_df.copy()
    t["entry_ts_utc"] = pd.to_datetime(t["entry_timestamp_utc"], utc=True)
    t["entry_stockholm_date"] = t["entry_ts_utc"].dt.tz_convert(ZoneInfo("Europe/Stockholm")).dt.strftime("%Y-%m-%d")
    t["entry_stockholm"] = t["entry_ts_utc"].dt.tz_convert(ZoneInfo("Europe/Stockholm")).dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    t["bucket_raw_display"] = t["bucket_raw"].map(lambda raw: parse_bucket_label(str(raw)).canonical_label() if parse_bucket_label(str(raw)) is not None else str(raw))
    return pd.DataFrame({
        "Target date (Local)": t["target_date_local"],
        "Market file day (Local)": t["market_file_date_local"],
        "Entry date (Stockholm)": t["entry_stockholm_date"],
        "Entry time (Stockholm)": t["entry_stockholm"],
        "Station": t["station_id"],
        "Bucket": t["bucket"],
        "Bucket raw (market)": t["bucket_raw_display"],
        "Side": t["side"],
        "Market win % (side)": (t["market_price"] * 100.0).round(2),
        "Model win %": (t["model_win_prob"] * 100.0).round(2),
        "EV": t["ev"].round(4),
        "Amount invested ($)": t["stake"].round(2),
        "Profit made ($)": t["pnl"].round(2),
        "Result": t["result"].map({"W": "Win", "L": "Loss"}).fillna(t["result"]),
        "Balance after trade ($)": t["balance_after"].round(2),
        "Market open (UTC)": t["market_open_utc"],
        "Gate cutoff (UTC)": t["gate_cutoff_utc"],
        "Effective cutoff (UTC)": t["effective_cutoff_utc"],
        "Market file": t["market_file"],
    })


def build_stockholm_display_table(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=[
            "Target date (Local)", "Market file day (Local)", "Entry date (Stockholm)", "Entry time (Stockholm)",
            "Station", "Bucket", "Bucket raw (market)", "Side", "Market win % (side)",
            "Model win %", "EV", "Amount invested ($)", "Profit made ($)", "Result", "Balance after trade ($)"
        ])
    t = trades_df.copy()
    t["entry_ts_utc"] = pd.to_datetime(t["entry_timestamp_utc"], utc=True)
    t["entry_stockholm_date"] = t["entry_ts_utc"].dt.tz_convert(ZoneInfo("Europe/Stockholm")).dt.strftime("%Y-%m-%d")
    t["entry_stockholm"] = t["entry_ts_utc"].dt.tz_convert(ZoneInfo("Europe/Stockholm")).dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    t["bucket_raw_display"] = t["bucket_raw"].map(lambda raw: parse_bucket_label(str(raw)).canonical_label() if parse_bucket_label(str(raw)) is not None else str(raw))
    return pd.DataFrame({
        "Target date (Local)": t["target_date_local"],
        "Market file day (Local)": t["market_file_date_local"],
        "Entry date (Stockholm)": t["entry_stockholm_date"],
        "Entry time (Stockholm)": t["entry_stockholm"],
        "Station": t["station_id"],
        "Bucket": t["bucket"],
        "Bucket raw (market)": t["bucket_raw_display"],
        "Side": t["side"],
        "Market win % (side)": (t["market_price"] * 100.0).round(2),
        "Model win %": (t["model_win_prob"] * 100.0).round(2),
        "EV": t["ev"].round(4),
        "Amount invested ($)": t["stake"].round(2),
        "Profit made ($)": t["pnl"].round(2),
        "Result": t["result"].map({"W": "Win", "L": "Loss"}).fillna(t["result"]),
        "Balance after trade ($)": t["balance_after"].round(2),
    })


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audited MOS blend_12 backtest (single-station or cojoined).")
    p.add_argument("--mode", choices=["single", "cojoined"], default="single")
    p.add_argument("--stations", default="KNYC", help="Comma-separated station ids.")
    p.add_argument("--prediction-source", choices=["parquet", "live-script"], default="parquet")
    p.add_argument("--pred-dev-by-station-json", default=None, help="JSON map station->dev parquet path (or path to JSON file).")
    p.add_argument("--pred-test-by-station-json", default=None, help="JSON map station->test parquet path (or path to JSON file).")
    p.add_argument("--truth-csv-by-station-json", default=None, help="JSON map station->truth csv path (or path to JSON file).")
    p.add_argument("--kalshi-root-by-station-json", default=None, help="JSON map station->kalshi root path (or path to JSON file).")
    p.add_argument("--file-prefix-by-station-json", default=None, help="JSON map station->market file prefix (or path to JSON file).")
    p.add_argument("--pred-dev-knyc", default=r"D:\Ahmed\data\kalshi\Experiments\MOS\03_blends\blend_12\dev_predictions.parquet")
    p.add_argument("--pred-test-knyc", default=r"D:\Ahmed\data\kalshi\Experiments\MOS\03_blends\blend_12\test_predictions.parquet")
    p.add_argument("--pred-dev-kmia", default=r"D:\Ahmed\data\kalshi\Experiments\MOS_KMIA\03_blends\blend_12\dev_predictions.parquet")
    p.add_argument("--pred-test-kmia", default=r"D:\Ahmed\data\kalshi\Experiments\MOS_KMIA\03_blends\blend_12\test_predictions.parquet")
    p.add_argument("--pred-dev-kmdw", default=r"D:\Ahmed\data\kalshi\Experiments\MOS_KMDW\03_blends\blend_12\dev_predictions.parquet")
    p.add_argument("--pred-test-kmdw", default=r"D:\Ahmed\data\kalshi\Experiments\MOS_KMDW\03_blends\blend_12\test_predictions.parquet")
    p.add_argument("--pred-dev-klax", default=r"D:\Ahmed\data\kalshi\Experiments\MOS_KLAX\03_blends\blend_12\dev_predictions.parquet")
    p.add_argument("--pred-test-klax", default=r"D:\Ahmed\data\kalshi\Experiments\MOS_KLAX\03_blends\blend_12\test_predictions.parquet")
    p.add_argument(
        "--live-script-path",
        default=str((Path(__file__).resolve().parents[1] / "tools" / "live" / "mos_quantile_live_inference.py")),
    )
    p.add_argument("--live-inference-root", default=r"D:\Ahmed\data\live\mos_quantile_live_inference\backtest_replay")
    p.add_argument("--live-script-python", default=sys.executable)
    p.add_argument("--live-script-log-level", default="ERROR")
    p.add_argument("--truth-csv-knyc", default=r"D:\Ahmed\data\kalshi\training_data\02_truth\KNYC_settled_tmax.csv")
    p.add_argument("--truth-csv-kmia", default=r"D:\Ahmed\data\kalshi\training_data\02_truth\KMIA_settled_tmax.csv")
    p.add_argument("--truth-csv-kmdw", default=r"D:\Ahmed\data\kalshi\training_data\02_truth\KMDW_settled_tmax_2002_2026.csv")
    p.add_argument("--truth-csv-klax", default=r"D:\Ahmed\data\kalshi\training_data\02_truth\KLAX_settled_tmax_2002_2026.csv")
    p.add_argument("--kalshi-root-knyc", default=r"D:\Ahmed\data\kalshi\kalshi_history\kxhighny_2024_10_01_to_2025_12_31")
    p.add_argument("--kalshi-root-kmia", default=r"D:\Ahmed\data\kalshi\kalshi_history\kxhighmia_2024_10_01_to_2025_12_31")
    p.add_argument("--kalshi-root-kmdw", default=r"D:\Ahmed\data\kalshi\kalshi_history\kxhighchi_2024_10_01_to_2026_03_03")
    p.add_argument("--kalshi-root-klax", default=r"D:\Ahmed\data\kalshi\kalshi_history\kxhighlax_2025_01_01_to_2026_03_05")
    p.add_argument("--start-date", default="2024-10-01")
    p.add_argument("--end-date", default="2025-12-31")
    p.add_argument("--entry-hour-z", type=int, default=12)
    p.add_argument("--entry-minute-z", type=int, default=0)
    p.add_argument("--min-entry-minutes-after-open", type=int, default=0)
    p.add_argument("--ev-min", type=float, default=0.15)
    p.add_argument("--win-min", type=float, default=0.65)
    p.add_argument("--min-market-price", type=float, default=0.0)
    p.add_argument("--start-balance", type=float, default=2700.0)
    p.add_argument("--risk-fraction", type=float, default=0.065)
    p.add_argument("--stake-cap-usd", type=float, default=400.0)
    p.add_argument("--out-dir", default=r"D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest")
    p.add_argument("--out-prefix", default="cojoined_blend12_knyc_kmia_tminus1_1200z_ev0p15_win65_risk6p5_cap400")
    p.add_argument("--table-out", default=r"D:\Ahmed\data\kalshi\plots\cojoined_blend12_knyc_kmia_stockholm_table.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if float(args.min_market_price) < 0.0 or float(args.min_market_price) > 1.0:
        raise ValueError("--min-market-price must be in [0, 1]")
    station_ids = parse_station_ids(args.stations)
    if not station_ids:
        raise ValueError("--stations must contain at least one station id")
    if args.mode == "single" and len(station_ids) != 1:
        raise ValueError("--mode single requires exactly one station id in --stations")

    pred_dev_map = parse_json_mapping(args.pred_dev_by_station_json)
    pred_test_map = parse_json_mapping(args.pred_test_by_station_json)
    truth_csv_map = parse_json_mapping(args.truth_csv_by_station_json)
    kalshi_root_map = parse_json_mapping(args.kalshi_root_by_station_json)
    file_prefix_map = parse_json_mapping(args.file_prefix_by_station_json)

    legacy_pred_dev = {
        "KNYC": str(args.pred_dev_knyc),
        "KMIA": str(args.pred_dev_kmia),
        "KMDW": str(args.pred_dev_kmdw),
        "KLAX": str(args.pred_dev_klax),
    }
    legacy_pred_test = {
        "KNYC": str(args.pred_test_knyc),
        "KMIA": str(args.pred_test_kmia),
        "KMDW": str(args.pred_test_kmdw),
        "KLAX": str(args.pred_test_klax),
    }
    legacy_truth = {
        "KNYC": str(args.truth_csv_knyc),
        "KMIA": str(args.truth_csv_kmia),
        "KMDW": str(args.truth_csv_kmdw),
        "KLAX": str(args.truth_csv_klax),
    }
    legacy_kalshi_root = {
        "KNYC": str(args.kalshi_root_knyc),
        "KMIA": str(args.kalshi_root_kmia),
        "KMDW": str(args.kalshi_root_kmdw),
        "KLAX": str(args.kalshi_root_klax),
    }
    legacy_file_prefix = {"KNYC": "KNYC", "KMIA": "KMIA", "KMDW": "KMDW", "KLAX": "KLAX"}

    if str(args.prediction_source) == "parquet":
        pred_maps: Dict[str, Dict[str, pd.Series]] = {}
        used_pred_dev: Dict[str, str] = {}
        used_pred_test: Dict[str, str] = {}
        for sid in station_ids:
            dev_path = pred_dev_map.get(sid) or legacy_pred_dev.get(sid)
            test_path = pred_test_map.get(sid) or legacy_pred_test.get(sid)
            if not dev_path or not test_path:
                raise ValueError(f"Missing parquet prediction paths for station={sid}")
            used_pred_dev[sid] = str(dev_path)
            used_pred_test[sid] = str(test_path)
            pred_maps[sid] = load_predictions(Path(dev_path), Path(test_path))
        source_meta: Dict = {
            "prediction_source": "parquet",
            "pred_dev_by_station": used_pred_dev,
            "pred_test_by_station": used_pred_test,
        }
    else:
        live_script_path = Path(args.live_script_path)
        if not live_script_path.exists():
            raise FileNotFoundError(f"Live script not found: {live_script_path}")
        truth_maps: Dict[str, Dict[str, float]] = {}
        used_truth_csv: Dict[str, str] = {}
        for sid in station_ids:
            truth_path = truth_csv_map.get(sid) or legacy_truth.get(sid)
            if not truth_path:
                raise ValueError(f"Missing truth CSV path for station={sid}")
            used_truth_csv[sid] = str(truth_path)
            truth_maps[sid] = load_truth_map(Path(truth_path))
        pred_maps, live_stats = load_predictions_from_live_script(
            start_date=str(args.start_date),
            end_date=str(args.end_date),
            live_script_path=live_script_path,
            live_root=Path(args.live_inference_root),
            python_bin=str(args.live_script_python),
            script_log_level=str(args.live_script_log_level),
            truth_maps=truth_maps,
            station_ids=station_ids,
        )
        source_meta = {
            "prediction_source": "live-script",
            "live_script_path": str(live_script_path),
            "live_inference_root": str(args.live_inference_root),
            "live_script_python": str(args.live_script_python),
            "live_script_log_level": str(args.live_script_log_level),
            "truth_csv_by_station": used_truth_csv,
            "live_loader_stats": live_stats,
        }

    market_indices: Dict[str, Dict[str, Path]] = {}
    used_kalshi_roots: Dict[str, str] = {}
    used_file_prefixes: Dict[str, str] = {}
    for sid in station_ids:
        root = kalshi_root_map.get(sid) or legacy_kalshi_root.get(sid)
        prefix = file_prefix_map.get(sid) or legacy_file_prefix.get(sid) or sid
        if not root:
            raise ValueError(f"Missing Kalshi root for station={sid}")
        used_kalshi_roots[sid] = str(root)
        used_file_prefixes[sid] = str(prefix).upper()
        market_indices[sid] = build_market_index(Path(root), str(prefix).upper())

    trades_df, summary, day_debug = run_backtest(
        station_ids=station_ids,
        pred_maps=pred_maps,
        market_indices=market_indices,
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        ev_min=float(args.ev_min),
        win_min=float(args.win_min),
        min_market_price=float(args.min_market_price),
        start_balance=float(args.start_balance),
        risk_fraction=float(args.risk_fraction),
        stake_cap_usd=float(args.stake_cap_usd),
        entry_hour_z=int(args.entry_hour_z),
        entry_minute_z=int(args.entry_minute_z),
        min_entry_minutes_after_open=int(args.min_entry_minutes_after_open),
    )
    sanity = run_sanity_audit(
        station_ids=station_ids,
        trades_df=trades_df,
        pred_maps=pred_maps,
        market_indices=market_indices,
        day_debug=day_debug,
        ev_min=float(args.ev_min),
        win_min=float(args.win_min),
        min_market_price=float(args.min_market_price),
        entry_hour_z=int(args.entry_hour_z),
        entry_minute_z=int(args.entry_minute_z),
        min_entry_minutes_after_open=int(args.min_entry_minutes_after_open),
        stake_cap_usd=float(args.stake_cap_usd),
    )
    summary["prediction_source_meta"] = source_meta
    summary["mode"] = str(args.mode)
    summary["stations"] = station_ids
    summary["kalshi_roots_by_station"] = used_kalshi_roots
    summary["file_prefix_by_station"] = used_file_prefixes

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = Path(args.table_out)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir / f"trades_{args.out_prefix}.csv"
    summary_path = out_dir / f"summary_{args.out_prefix}.json"
    sanity_path = out_dir / f"sanity_{args.out_prefix}.json"
    debug_path = out_dir / f"day_debug_{args.out_prefix}.json"
    sideaware_path = table_path.parent / f"all_trades_sideaware_{args.out_prefix}_with_balance.csv"

    trades_df.to_csv(trades_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sanity_path.write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    debug_path.write_text(json.dumps(day_debug, indent=2), encoding="utf-8")
    build_sideaware_with_balance_table(trades_df).to_csv(sideaware_path, index=False)
    build_stockholm_display_table(trades_df).to_csv(table_path, index=False)

    print("WROTE_TRADES", trades_path)
    print("WROTE_SUMMARY", summary_path)
    print("WROTE_SANITY", sanity_path)
    print("WROTE_DAY_DEBUG", debug_path)
    print("WROTE_SIDEAWARE", sideaware_path)
    print("WROTE_TABLE", table_path)
    print(json.dumps({
        "trades": summary.get("trades"),
        "final_balance": summary.get("final_balance"),
        "profit_factor": summary.get("profit_factor"),
        "max_drawdown": summary.get("max_drawdown"),
        "station_counts": summary.get("station_counts", {}),
    }, indent=2))


if __name__ == "__main__":
    main()
