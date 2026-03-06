from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import mos_blend12_knyc_kmia_cojoined_audit as cojoined


DEFAULT_STATIONS = ["KNYC", "KMIA", "KMDW", "KLAX"]
DEFAULT_TRUTH = {
    "KNYC": r"D:\Ahmed\data\kalshi\training_data\02_truth\KNYC_settled_tmax.csv",
    "KMIA": r"D:\Ahmed\data\kalshi\training_data\02_truth\KMIA_settled_tmax.csv",
    "KMDW": r"D:\Ahmed\data\kalshi\training_data\02_truth\KMDW_settled_tmax_2002_2026.csv",
    "KLAX": r"D:\Ahmed\data\kalshi\training_data\02_truth\KLAX_settled_tmax_2002_2026.csv",
}
DEFAULT_KALSHI_ROOT = {
    "KNYC": r"D:\Ahmed\data\kalshi\kalshi_history\kxhighny_2024_10_01_to_2025_12_31",
    "KMIA": r"D:\Ahmed\data\kalshi\kalshi_history\kxhighmia_2024_10_01_to_2025_12_31",
    "KMDW": r"D:\Ahmed\data\kalshi\kalshi_history\kxhighchi_2024_10_01_to_2026_03_03",
    "KLAX": r"D:\Ahmed\data\kalshi\kalshi_history\kxhighlax_2025_01_01_to_2026_03_05",
}
DEFAULT_FILE_PREFIX = {sid: sid for sid in DEFAULT_STATIONS}
DEFAULT_LIVE_ROOT = r"D:\Ahmed\data\live\mos_quantile_live_inference\backtest_replay_knyc_kmia_kmdw_klax_2024_2025"
DEFAULT_OUT_DIR = r"D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\live_replay_grid_4station_2024_2025"
DEFAULT_SQLITE_DIR = r"D:\Ahmed\data\sqlite"


def _sanitize_for_json(payload: Dict) -> Dict:
    return json.loads(json.dumps(payload, default=str))


def _build_float_grid(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("grid step must be > 0")
    d_start = Decimal(str(start))
    d_end = Decimal(str(end))
    d_step = Decimal(str(step))
    if d_end < d_start:
        raise ValueError("grid end must be >= grid start")
    vals: List[float] = []
    cur = d_start
    guard = 0
    while cur <= d_end + Decimal("1e-12"):
        vals.append(float(cur))
        cur += d_step
        guard += 1
        if guard > 20000:
            raise RuntimeError("grid generation guard triggered")
    return vals


def _tag(x: float) -> str:
    return str(x).replace(".", "p")


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _simulate_bankroll(
    trades_template: pd.DataFrame,
    *,
    start_balance: float,
    stake_cap_usd: float,
    sizing_mode: str,
    risk_fraction: Optional[float],
    kelly_fraction: Optional[float],
) -> Tuple[pd.DataFrame, Dict]:
    t = trades_template.copy()
    t = t.sort_values(["target_date_local", "entry_timestamp_utc", "station_id"], kind="stable").reset_index(drop=True)

    balance = float(start_balance)
    peak = float(start_balance)
    out_rows: List[Dict] = []

    for _, row in t.iterrows():
        price = float(row["market_price"])
        model_win_prob = float(row["model_win_prob"])
        win = int(row["win"])

        if sizing_mode == "fixed_risk":
            if risk_fraction is None:
                raise ValueError("risk_fraction is required for fixed_risk")
            full_kelly = None
            risk_fraction_used = float(risk_fraction)
        elif sizing_mode == "fractional_kelly":
            if kelly_fraction is None:
                raise ValueError("kelly_fraction is required for fractional_kelly")
            if price <= 0.0 or price >= 1.0:
                full_kelly = 0.0
            else:
                full_kelly = (model_win_prob - price) / (1.0 - price)
                full_kelly = max(0.0, min(1.0, float(full_kelly)))
            risk_fraction_used = float(kelly_fraction) * full_kelly
        else:
            raise ValueError(f"Unsupported sizing_mode={sizing_mode}")

        stake = min(balance * risk_fraction_used, float(stake_cap_usd))
        shares = stake / price if price > 0.0 else 0.0
        pnl = shares * (1.0 - price) if win == 1 else -stake

        balance_before = balance
        balance = balance + pnl
        peak = max(peak, balance)
        drawdown = 0.0 if peak <= 0.0 else (peak - balance) / peak

        out_row = row.to_dict()
        out_row["risk_fraction_used"] = float(risk_fraction_used)
        out_row["kelly_full_fraction"] = None if full_kelly is None else float(full_kelly)
        out_row["stake"] = float(stake)
        out_row["shares"] = float(shares)
        out_row["pnl"] = float(pnl)
        out_row["balance_before"] = float(balance_before)
        out_row["balance_after"] = float(balance)
        out_row["drawdown"] = float(drawdown)
        out_row["result"] = "W" if win == 1 else "L"
        out_rows.append(out_row)

    out = pd.DataFrame(out_rows)
    if out.empty:
        gross_profit = 0.0
        gross_loss = 0.0
        wins = 0
        losses = 0
        avg_ev = 0.0
        median_ev = 0.0
        max_drawdown = 0.0
        risk_fraction_avg = 0.0
        risk_fraction_min = 0.0
        risk_fraction_max = 0.0
    else:
        gross_profit = float(out.loc[out["win"] == 1, "pnl"].sum())
        gross_loss = float(-out.loc[out["win"] == 0, "pnl"].sum())
        wins = int((out["win"] == 1).sum())
        losses = int((out["win"] == 0).sum())
        avg_ev = float(out["ev"].mean())
        median_ev = float(out["ev"].median())
        max_drawdown = float(out["drawdown"].max())
        risk_fraction_avg = float(out["risk_fraction_used"].mean())
        risk_fraction_min = float(out["risk_fraction_used"].min())
        risk_fraction_max = float(out["risk_fraction_used"].max())

    return out, {
        "sizing_mode": sizing_mode,
        "risk_fraction": None if risk_fraction is None else float(risk_fraction),
        "kelly_fraction": None if kelly_fraction is None else float(kelly_fraction),
        "stake_cap_usd": float(stake_cap_usd),
        "start_balance": float(start_balance),
        "trades": int(len(out)),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": (float(wins) / float(len(out))) if len(out) > 0 else 0.0,
        "profit_factor": (float(gross_profit) / float(gross_loss)) if gross_loss > 0 else None,
        "final_balance": float(balance),
        "total_pnl": float(balance - start_balance),
        "avg_ev_at_trade": avg_ev,
        "median_ev_at_trade": median_ev,
        "max_drawdown": max_drawdown,
        "station_counts": out["station_id"].value_counts().to_dict() if not out.empty else {},
        "side_counts": out["side"].value_counts().to_dict() if not out.empty else {},
        "risk_fraction_used_avg": risk_fraction_avg,
        "risk_fraction_used_min": risk_fraction_min,
        "risk_fraction_used_max": risk_fraction_max,
        "stake_cap_breach_count": int((out["stake"] > (float(stake_cap_usd) + 1e-9)).sum()) if not out.empty else 0,
    }


def _resolve_mapping(cli_json: Optional[str], legacy: Dict[str, str]) -> Dict[str, str]:
    merged = dict(legacy)
    if cli_json:
        merged.update(cojoined.parse_json_mapping(cli_json))
    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate 4-station live-script replay forecasts for the 2024-2025 backtest window, "
            "persist them into SQLite, and run the EV/win/risk grid into the same SQLite database."
        )
    )
    p.add_argument("--stations", default="KNYC,KMIA,KMDW,KLAX")
    p.add_argument("--start-date", default="2024-10-01")
    p.add_argument("--end-date", default="2025-12-31")
    p.add_argument("--entry-hour-z", type=int, default=12)
    p.add_argument("--entry-minute-z", type=int, default=0)
    p.add_argument("--min-entry-minutes-after-open", type=int, default=30)
    p.add_argument("--min-market-price", type=float, default=0.25)
    p.add_argument("--start-balance", type=float, default=2700.0)
    p.add_argument("--stake-cap-usd", type=float, default=700.0)
    p.add_argument("--selection-risk-fraction", type=float, default=0.075)
    p.add_argument("--ev-start", type=float, default=0.15)
    p.add_argument("--ev-end", type=float, default=0.55)
    p.add_argument("--ev-step", type=float, default=0.05)
    p.add_argument("--win-start", type=float, default=0.65)
    p.add_argument("--win-end", type=float, default=0.90)
    p.add_argument("--win-step", type=float, default=0.05)
    p.add_argument("--fixed-risk-start", type=float, default=0.045)
    p.add_argument("--fixed-risk-end", type=float, default=0.085)
    p.add_argument("--fixed-risk-step", type=float, default=0.01)
    p.add_argument("--kelly-start", type=float, default=0.10)
    p.add_argument("--kelly-end", type=float, default=0.25)
    p.add_argument("--kelly-step", type=float, default=0.01)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--run-prefix", default="cojoined_blend12_knyc_kmia_kmdw_klax_tminus1_1200z_openplus30m_live_script_grid_2024_2025")
    p.add_argument("--sqlite-dir", default=DEFAULT_SQLITE_DIR)
    p.add_argument("--sqlite-path", default=None)
    p.add_argument("--overwrite-db", action="store_true")
    p.add_argument("--live-inference-root", default=DEFAULT_LIVE_ROOT)
    p.add_argument(
        "--live-script-path",
        default=str((Path(__file__).resolve().parents[1] / "tools" / "live" / "mos_quantile_live_inference.py")),
    )
    p.add_argument("--live-script-python", default=sys.executable)
    p.add_argument("--live-script-log-level", default="ERROR")
    p.add_argument("--truth-csv-by-station-json", default=None)
    p.add_argument("--kalshi-root-by-station-json", default=None)
    p.add_argument("--file-prefix-by-station-json", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    station_ids = cojoined.parse_station_ids(args.stations)
    if not station_ids:
        raise ValueError("--stations must contain at least one station id")

    truth_map_paths = _resolve_mapping(args.truth_csv_by_station_json, DEFAULT_TRUTH)
    kalshi_root_paths = _resolve_mapping(args.kalshi_root_by_station_json, DEFAULT_KALSHI_ROOT)
    file_prefix_paths = _resolve_mapping(args.file_prefix_by_station_json, DEFAULT_FILE_PREFIX)

    for sid in station_ids:
        if sid not in truth_map_paths:
            raise ValueError(f"Missing truth path for station={sid}")
        if sid not in kalshi_root_paths:
            raise ValueError(f"Missing Kalshi root for station={sid}")

    truth_maps = {sid: cojoined.load_truth_map(Path(truth_map_paths[sid])) for sid in station_ids}
    market_indices = {
        sid: cojoined.build_market_index(Path(kalshi_root_paths[sid]), str(file_prefix_paths.get(sid, sid)).upper())
        for sid in station_ids
    }

    ev_values = _build_float_grid(args.ev_start, args.ev_end, args.ev_step)
    win_values = _build_float_grid(args.win_start, args.win_end, args.win_step)
    fixed_risk_values = _build_float_grid(args.fixed_risk_start, args.fixed_risk_end, args.fixed_risk_step)
    kelly_values = _build_float_grid(args.kelly_start, args.kelly_end, args.kelly_step)

    live_root = Path(args.live_inference_root)
    live_root.mkdir(parents=True, exist_ok=True)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    sqlite_dir = Path(args.sqlite_dir)
    sqlite_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_root / f"{args.run_prefix}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = Path(args.sqlite_path) if args.sqlite_path else sqlite_dir / f"{args.run_prefix}_{run_id}.sqlite"
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite_db and sqlite_path.exists():
        sqlite_path.unlink()

    entry_rule_text = (
        f"entry_timestamp_utc >= max(T-1 {int(args.entry_hour_z):02d}:{int(args.entry_minute_z):02d}Z, "
        f"market_open_utc + {int(args.min_entry_minutes_after_open)}m); "
        "enter at the first eligible global timestamp across all configured stations; "
        "tie-break: model_win_prob desc, ev desc, market_price asc, station_id asc."
    )
    side_price_rule_text = "YES side uses normalized bucket price; NO side uses 1-YES."
    fixed_rule_text = "stake=min(balance_before*risk_fraction, stake_cap_usd)"
    kelly_rule_text = (
        "full_kelly=clamp((q-p)/(1-p),0,1); "
        "risk_fraction_used=kelly_fraction*full_kelly; "
        "stake=min(balance_before*risk_fraction_used, stake_cap_usd)"
    )

    pred_maps: Dict[str, Dict[str, Dict]] = {sid: {} for sid in station_ids}
    day_rows: List[Dict] = []
    forecast_rows: List[Dict] = []
    requested_days = pd.date_range(pd.Timestamp(args.start_date), pd.Timestamp(args.end_date), freq="D")

    for i, day in enumerate(requested_days, start=1):
        day_key = day.strftime("%Y-%m-%d")
        ymd = day.strftime("%Y%m%d")
        target_dir = live_root / f"target_{ymd}"
        report_path = target_dir / "inference_report.json"
        failure_path = target_dir / "runtime_gate_failure.json"

        try:
            report = cojoined.run_live_inference_for_target(
                target_day=day,
                live_script_path=Path(args.live_script_path),
                live_root=live_root,
                python_bin=str(args.live_script_python),
                script_log_level=str(args.live_script_log_level),
                station_ids=station_ids,
            )
            day_status = "ok"
            failure_json = None
            error_text = None
        except Exception as exc:
            report = {}
            day_status = "live_inference_failed"
            error_text = str(exc)
            failure_json = None
            if failure_path.exists():
                try:
                    failure_json = json.loads(failure_path.read_text(encoding="utf-8"))
                except Exception:
                    failure_json = {"raw_excerpt": failure_path.read_text(encoding="utf-8")[:4000]}

        report_sha = _sha256_file(report_path)
        leakage = report.get("leakage_proof", {}) if isinstance(report, dict) else {}
        counters = leakage.get("global_guardrail_counters", {}) if isinstance(leakage, dict) else {}

        day_rows.append(
            {
                "run_id": run_id,
                "target_date_local": day_key,
                "target_ymd": ymd,
                "status": day_status,
                "report_path": str(report_path),
                "report_sha256": report_sha,
                "runtime_gate_failure_path": str(failure_path) if failure_path.exists() else None,
                "runtime_gate_failure_json": None if failure_json is None else _json_text(failure_json),
                "error_text": error_text,
                "live_run_id": report.get("run_id") if isinstance(report, dict) else None,
                "quote_asof_utc": report.get("quote_asof_utc") if isinstance(report, dict) else None,
                "requested_station_count": int(len(station_ids)),
                "available_station_count": int(len(report.get("inference_by_station", {}))) if isinstance(report.get("inference_by_station"), dict) else 0,
                "global_guardrail_counters_json": _json_text(counters),
            }
        )

        for sid in station_ids:
            block = cojoined._extract_live_report_block(report, sid) if report else {}
            quantiles = block.get("quantiles", {}) if isinstance(block, dict) else {}
            evidence_all = leakage.get("per_station_evidence", {}) if isinstance(leakage, dict) else {}
            evidence = evidence_all.get(sid, {}) if isinstance(evidence_all, dict) else {}
            truth_value = truth_maps[sid].get(day_key)
            row_status = "ok" if quantiles else ("live_inference_failed" if day_status != "ok" else "missing_station_block")
            forecast_rows.append(
                {
                    "run_id": run_id,
                    "target_date_local": day_key,
                    "station_id": sid,
                    "status": row_status,
                    "report_path": str(report_path),
                    "report_sha256": report_sha,
                    "truth_available": 1 if truth_value is not None else 0,
                    "y_tmax": None if truth_value is None else float(truth_value),
                    "runtime_utc": block.get("runtime_utc"),
                    "runtime_expected_policy_utc": evidence.get("runtime_expected_from_policy_utc"),
                    "runtime_equals_expected_policy_runtime": evidence.get("runtime_equals_expected_policy_runtime"),
                    "runtime_lte_quote_asof": evidence.get("runtime_lte_quote_asof"),
                    "quantiles_monotonic": evidence.get("inference_quantiles_monotonic"),
                    "prediction_point_tmax_f": block.get("prediction_point_tmax_f"),
                    "q_0_05": quantiles.get("q_0.05"),
                    "q_0_10": quantiles.get("q_0.10"),
                    "q_0_25": quantiles.get("q_0.25"),
                    "q_0_50": quantiles.get("q_0.50"),
                    "q_0_75": quantiles.get("q_0.75"),
                    "q_0_90": quantiles.get("q_0.90"),
                    "q_0_95": quantiles.get("q_0.95"),
                    "bundle_dir": evidence.get("bundle_dir"),
                    "station_block_json": _json_text(block),
                    "station_evidence_json": _json_text(evidence),
                }
            )
            if quantiles and truth_value is not None:
                pred_maps[sid][day_key] = {
                    "target_date_local": day_key,
                    "y_tmax": float(truth_value),
                    "q_0.05": float(quantiles["q_0.05"]),
                    "q_0.10": float(quantiles["q_0.10"]),
                    "q_0.25": float(quantiles["q_0.25"]),
                    "q_0.50": float(quantiles["q_0.50"]),
                    "q_0.75": float(quantiles["q_0.75"]),
                    "q_0.90": float(quantiles["q_0.90"]),
                    "q_0.95": float(quantiles["q_0.95"]),
                }

        if i % 25 == 0 or i == len(requested_days):
            stored_rows = {sid: len(pred_maps[sid]) for sid in station_ids}
            print(json.dumps({"phase": "forecast_replay", "progress": f"{i}/{len(requested_days)}", "stored_rows": stored_rows}))

    base_rows: List[Dict] = []
    combo_rows: List[Dict] = []

    for ev_min in ev_values:
        for win_min in win_values:
            trades_df, summary, day_debug = cojoined.run_backtest(
                station_ids=station_ids,
                pred_maps=pred_maps,
                market_indices=market_indices,
                start_date=str(args.start_date),
                end_date=str(args.end_date),
                ev_min=float(ev_min),
                win_min=float(win_min),
                min_market_price=float(args.min_market_price),
                start_balance=float(args.start_balance),
                risk_fraction=float(args.selection_risk_fraction),
                stake_cap_usd=float(args.stake_cap_usd),
                entry_hour_z=int(args.entry_hour_z),
                entry_minute_z=int(args.entry_minute_z),
                min_entry_minutes_after_open=int(args.min_entry_minutes_after_open),
            )
            sanity = cojoined.run_sanity_audit(
                station_ids=station_ids,
                trades_df=trades_df,
                pred_maps=pred_maps,
                market_indices=market_indices,
                day_debug=day_debug,
                ev_min=float(ev_min),
                win_min=float(win_min),
                min_market_price=float(args.min_market_price),
                entry_hour_z=int(args.entry_hour_z),
                entry_minute_z=int(args.entry_minute_z),
                min_entry_minutes_after_open=int(args.min_entry_minutes_after_open),
                stake_cap_usd=float(args.stake_cap_usd),
            )

            ev_tag = _tag(ev_min)
            win_tag = _tag(win_min)
            trades_path = run_dir / f"trades_base_ev{ev_tag}_win{win_tag}.csv"
            summary_path = run_dir / f"summary_base_ev{ev_tag}_win{win_tag}.json"
            sanity_path = run_dir / f"sanity_base_ev{ev_tag}_win{win_tag}.json"
            debug_path = run_dir / f"day_debug_base_ev{ev_tag}_win{win_tag}.json"

            trades_df.to_csv(trades_path, index=False)
            summary_payload = dict(summary)
            summary_payload["run_id"] = run_id
            summary_payload["stations"] = station_ids
            summary_payload["entry_rule"] = entry_rule_text
            summary_payload["side_price_rule"] = side_price_rule_text
            summary_payload["stake_rule"] = (
                f"selection-run only: stake=min(balance_before*{float(args.selection_risk_fraction):.6f}, {float(args.stake_cap_usd):.2f})"
            )
            summary_payload["forecast_sqlite_path"] = str(sqlite_path)
            summary_path.write_text(json.dumps(_sanitize_for_json(summary_payload), indent=2), encoding="utf-8")
            sanity_path.write_text(json.dumps(_sanitize_for_json(sanity), indent=2), encoding="utf-8")
            debug_path.write_text(json.dumps(_sanitize_for_json(day_debug), indent=2), encoding="utf-8")

            base_rows.append(
                {
                    "run_id": run_id,
                    "ev_min": float(ev_min),
                    "win_min": float(win_min),
                    "trades": int(summary.get("trades", 0)),
                    "wins": int(summary.get("wins", 0)),
                    "losses": int(summary.get("losses", 0)),
                    "win_rate": float(summary.get("win_rate", 0.0)),
                    "final_balance": float(summary.get("final_balance", args.start_balance)),
                    "max_drawdown": float(summary.get("max_drawdown", 0.0)),
                    "days_without_trade_candidate": int(summary.get("days_without_trade_candidate", 0)),
                    "station_counts_json": _json_text(summary.get("station_counts", {})),
                    "side_counts_json": _json_text(summary.get("side_counts", {})),
                    "trades_csv_path": str(trades_path),
                    "summary_json_path": str(summary_path),
                    "sanity_json_path": str(sanity_path),
                    "day_debug_json_path": str(debug_path),
                    "sanity_passes_all_checks": 1 if bool(sanity.get("passes_all_checks", False)) else 0,
                    "sanity_checked_trades": int(sanity.get("checked_trades", 0)),
                }
            )

            if not bool(sanity.get("passes_all_checks", False)):
                raise RuntimeError(f"Base stream sanity failed for ev_min={ev_min}, win_min={win_min}: {sanity_path}")

            stream_template = trades_df[
                [
                    "target_date_local",
                    "station_id",
                    "entry_timestamp_utc",
                    "market_open_utc",
                    "gate_cutoff_utc",
                    "effective_cutoff_utc",
                    "market_file",
                    "market_file_date_local",
                    "bucket",
                    "bucket_raw",
                    "side",
                    "market_price",
                    "model_win_prob",
                    "ev",
                    "y_tmax",
                    "win",
                ]
            ].copy()

            for risk_fraction in fixed_risk_values:
                _, metrics = _simulate_bankroll(
                    trades_template=stream_template,
                    start_balance=float(args.start_balance),
                    stake_cap_usd=float(args.stake_cap_usd),
                    sizing_mode="fixed_risk",
                    risk_fraction=float(risk_fraction),
                    kelly_fraction=None,
                )
                combo_rows.append(
                    {
                        "run_id": run_id,
                        "sizing_mode": "fixed_risk",
                        "ev_min": float(ev_min),
                        "win_min": float(win_min),
                        "risk_fraction": float(risk_fraction),
                        "kelly_fraction": None,
                        "stake_cap_usd": float(args.stake_cap_usd),
                        "entry_rule": entry_rule_text,
                        "side_price_rule": side_price_rule_text,
                        "stake_rule": fixed_rule_text,
                        "summary_json_path": str(summary_path),
                        "sanity_json_path": str(sanity_path),
                        "trades": int(metrics["trades"]),
                        "wins": int(metrics["wins"]),
                        "losses": int(metrics["losses"]),
                        "win_rate": float(metrics["win_rate"]),
                        "profit_factor": metrics["profit_factor"],
                        "final_balance": float(metrics["final_balance"]),
                        "total_pnl": float(metrics["total_pnl"]),
                        "max_drawdown": float(metrics["max_drawdown"]),
                        "avg_ev_at_trade": float(metrics["avg_ev_at_trade"]),
                        "median_ev_at_trade": float(metrics["median_ev_at_trade"]),
                        "station_counts_json": _json_text(metrics["station_counts"]),
                        "side_counts_json": _json_text(metrics["side_counts"]),
                        "risk_fraction_used_avg": float(metrics["risk_fraction_used_avg"]),
                        "risk_fraction_used_min": float(metrics["risk_fraction_used_min"]),
                        "risk_fraction_used_max": float(metrics["risk_fraction_used_max"]),
                        "stake_cap_breach_count": int(metrics["stake_cap_breach_count"]),
                    }
                )

            for kelly_fraction in kelly_values:
                _, metrics = _simulate_bankroll(
                    trades_template=stream_template,
                    start_balance=float(args.start_balance),
                    stake_cap_usd=float(args.stake_cap_usd),
                    sizing_mode="fractional_kelly",
                    risk_fraction=None,
                    kelly_fraction=float(kelly_fraction),
                )
                combo_rows.append(
                    {
                        "run_id": run_id,
                        "sizing_mode": "fractional_kelly",
                        "ev_min": float(ev_min),
                        "win_min": float(win_min),
                        "risk_fraction": None,
                        "kelly_fraction": float(kelly_fraction),
                        "stake_cap_usd": float(args.stake_cap_usd),
                        "entry_rule": entry_rule_text,
                        "side_price_rule": side_price_rule_text,
                        "stake_rule": kelly_rule_text,
                        "summary_json_path": str(summary_path),
                        "sanity_json_path": str(sanity_path),
                        "trades": int(metrics["trades"]),
                        "wins": int(metrics["wins"]),
                        "losses": int(metrics["losses"]),
                        "win_rate": float(metrics["win_rate"]),
                        "profit_factor": metrics["profit_factor"],
                        "final_balance": float(metrics["final_balance"]),
                        "total_pnl": float(metrics["total_pnl"]),
                        "max_drawdown": float(metrics["max_drawdown"]),
                        "avg_ev_at_trade": float(metrics["avg_ev_at_trade"]),
                        "median_ev_at_trade": float(metrics["median_ev_at_trade"]),
                        "station_counts_json": _json_text(metrics["station_counts"]),
                        "side_counts_json": _json_text(metrics["side_counts"]),
                        "risk_fraction_used_avg": float(metrics["risk_fraction_used_avg"]),
                        "risk_fraction_used_min": float(metrics["risk_fraction_used_min"]),
                        "risk_fraction_used_max": float(metrics["risk_fraction_used_max"]),
                        "stake_cap_breach_count": int(metrics["stake_cap_breach_count"]),
                    }
                )

            print(json.dumps({"phase": "grid", "ev_min": ev_min, "win_min": win_min, "trades": int(summary.get("trades", 0))}))

    day_df = pd.DataFrame(day_rows)
    forecast_df = pd.DataFrame(forecast_rows)
    base_df = pd.DataFrame(base_rows)
    combo_df = pd.DataFrame(combo_rows)

    rank_df = combo_df.copy()
    if not rank_df.empty:
        rank_df = rank_df[rank_df["profit_factor"].notna()].copy()
        if not rank_df.empty:
            max_pf_log = max(float(math.log1p(float(x))) for x in rank_df["profit_factor"])
            if max_pf_log <= 0.0:
                max_pf_log = 1.0
            rank_df["pf_component"] = rank_df["profit_factor"].map(lambda x: float(math.log1p(float(x))) / max_pf_log)
            rank_df["win_component"] = rank_df["win_rate"].clip(lower=0.0, upper=1.0)
            rank_df["drawdown_component"] = (1.0 - rank_df["max_drawdown"]).clip(lower=0.0, upper=1.0)
            rank_df["composite_score_pf_win_lowdd"] = (
                rank_df["pf_component"] * rank_df["win_component"] * rank_df["drawdown_component"]
            ) ** (1.0 / 3.0)

    run_meta_df = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "start_date": str(args.start_date),
                "end_date": str(args.end_date),
                "stations_json": _json_text(station_ids),
                "forecast_live_root": str(live_root),
                "forecast_script_path": str(args.live_script_path),
                "forecast_script_python": str(args.live_script_python),
                "entry_hour_z": int(args.entry_hour_z),
                "entry_minute_z": int(args.entry_minute_z),
                "min_entry_minutes_after_open": int(args.min_entry_minutes_after_open),
                "min_market_price": float(args.min_market_price),
                "start_balance": float(args.start_balance),
                "stake_cap_usd": float(args.stake_cap_usd),
                "selection_risk_fraction": float(args.selection_risk_fraction),
                "ev_values_json": _json_text(ev_values),
                "win_values_json": _json_text(win_values),
                "fixed_risk_values_json": _json_text(fixed_risk_values),
                "kelly_values_json": _json_text(kelly_values),
                "entry_rule": entry_rule_text,
                "side_price_rule": side_price_rule_text,
                "stake_rule_fixed": fixed_rule_text,
                "stake_rule_fractional_kelly": kelly_rule_text,
                "requested_day_count": int(len(requested_days)),
                "forecast_day_count": int(len(day_df)),
                "forecast_row_count": int(len(forecast_df)),
                "base_stream_count": int(len(base_df)),
                "combo_count": int(len(combo_df)),
                "out_dir": str(run_dir),
                "sqlite_path": str(sqlite_path),
            }
        ]
    )

    with sqlite3.connect(sqlite_path) as conn:
        day_df.to_sql("live_replay_target_days", conn, if_exists="replace", index=False)
        forecast_df.to_sql("live_replay_station_forecasts", conn, if_exists="replace", index=False)
        base_df.to_sql("backtest_base_streams", conn, if_exists="replace", index=False)
        combo_df.to_sql("backtest_combo_results", conn, if_exists="replace", index=False)
        rank_df.to_sql("backtest_ranked_scores", conn, if_exists="replace", index=False)
        run_meta_df.to_sql("backtest_run_meta", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_replay_target_days_target_date ON live_replay_target_days(target_date_local)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_replay_station_forecasts_target_station ON live_replay_station_forecasts(target_date_local, station_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_backtest_combo_results_grid ON backtest_combo_results(sizing_mode, ev_min, win_min, risk_fraction, kelly_fraction)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_backtest_combo_results_perf ON backtest_combo_results(final_balance DESC)")
        conn.commit()

    run_sanity = {
        "run_id": run_id,
        "passes_all_checks": (
            int((day_df["status"] == "ok").sum()) > 0
            and int((base_df["sanity_passes_all_checks"] == 0).sum()) == 0
            and int(combo_df["stake_cap_breach_count"].sum()) == 0
        ),
        "forecast_failed_day_count": int((day_df["status"] != "ok").sum()) if not day_df.empty else 0,
        "forecast_success_day_count": int((day_df["status"] == "ok").sum()) if not day_df.empty else 0,
        "base_stream_sanity_fail_count": int((base_df["sanity_passes_all_checks"] == 0).sum()) if not base_df.empty else 0,
        "combo_stake_cap_breach_total": int(combo_df["stake_cap_breach_count"].sum()) if not combo_df.empty else 0,
        "sqlite_path": str(sqlite_path),
    }
    run_sanity_path = run_dir / "run_sanity.json"
    run_sanity_path.write_text(json.dumps(_sanitize_for_json(run_sanity), indent=2), encoding="utf-8")

    top_rows: List[Dict] = []
    if not combo_df.empty:
        for mode in ["fixed_risk", "fractional_kelly"]:
            top = combo_df[combo_df["sizing_mode"] == mode].sort_values(
                ["final_balance", "profit_factor"], ascending=[False, False]
            ).head(5)
            for _, r in top.iterrows():
                top_rows.append(
                    {
                        "sizing_mode": str(r["sizing_mode"]),
                        "ev_min": float(r["ev_min"]),
                        "win_min": float(r["win_min"]),
                        "risk_fraction": None if pd.isna(r["risk_fraction"]) else float(r["risk_fraction"]),
                        "kelly_fraction": None if pd.isna(r["kelly_fraction"]) else float(r["kelly_fraction"]),
                        "trades": int(r["trades"]),
                        "win_rate": float(r["win_rate"]),
                        "profit_factor": None if pd.isna(r["profit_factor"]) else float(r["profit_factor"]),
                        "final_balance": float(r["final_balance"]),
                        "max_drawdown": float(r["max_drawdown"]),
                        "summary_json_path": str(r["summary_json_path"]),
                        "sanity_json_path": str(r["sanity_json_path"]),
                    }
                )

    run_summary = {
        "run_id": run_id,
        "sqlite_path": str(sqlite_path),
        "out_dir": str(run_dir),
        "summary_json_path": str(run_dir / "run_summary.json"),
        "sanity_json_path": str(run_sanity_path),
        "period_start": str(args.start_date),
        "period_end": str(args.end_date),
        "stations": station_ids,
        "forecast_live_root": str(live_root),
        "entry_rule": entry_rule_text,
        "side_price_rule": side_price_rule_text,
        "stake_rules": {
            "fixed_risk": fixed_rule_text,
            "fractional_kelly": kelly_rule_text,
        },
        "requested_day_count": int(len(requested_days)),
        "forecast_success_day_count": int((day_df["status"] == "ok").sum()) if not day_df.empty else 0,
        "forecast_failed_day_count": int((day_df["status"] != "ok").sum()) if not day_df.empty else 0,
        "base_stream_count": int(len(base_df)),
        "combo_count": int(len(combo_df)),
        "top_combos": top_rows,
    }
    run_summary_path = run_dir / "run_summary.json"
    run_summary_path.write_text(json.dumps(_sanitize_for_json(run_summary), indent=2), encoding="utf-8")

    print("WROTE_SQLITE", sqlite_path)
    print("WROTE_RUN_SUMMARY", run_summary_path)
    print("WROTE_RUN_SANITY", run_sanity_path)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "sqlite_path": str(sqlite_path),
                "forecast_success_day_count": run_summary["forecast_success_day_count"],
                "forecast_failed_day_count": run_summary["forecast_failed_day_count"],
                "base_stream_count": run_summary["base_stream_count"],
                "combo_count": run_summary["combo_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
