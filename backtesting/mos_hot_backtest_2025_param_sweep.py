from __future__ import annotations

import argparse
import json
import sqlite3
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import mos_blend12_knyc_kmia_cojoined_audit as cojoined


DEFAULT_WIN_VALUES = [0.65, 0.75, 0.85, 0.95]
DEFAULT_EV_VALUES = [0.15, 0.25, 0.35, 0.45]
DEFAULT_RISK_VALUES = [0.03, 0.045, 0.06, 0.075]
DEFAULT_KELLY_VALUES = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
ALL_SIZING_MODES = ["fixed_risk", "fractional_kelly", "fractional_kelly_with_risk_ceiling"]


def _tag(x: float) -> str:
    return str(x).replace(".", "p")


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


def _simulate_bankroll(
    trades_template: pd.DataFrame,
    start_balance: float,
    stake_cap_usd: float,
    sizing_mode: str,
    risk_fraction: Optional[float],
    kelly_fraction: Optional[float],
) -> Tuple[pd.DataFrame, Dict]:
    t = trades_template.copy()
    t = t.sort_values(["target_date_local", "entry_timestamp_utc"]).reset_index(drop=True)

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
            risk_fraction_used = float(risk_fraction)
            full_kelly = None
        elif sizing_mode == "fractional_kelly":
            if kelly_fraction is None:
                raise ValueError("kelly_fraction is required for fractional_kelly")
            if price <= 0.0 or price >= 1.0:
                full_kelly = 0.0
            else:
                full_kelly = (model_win_prob - price) / (1.0 - price)
                full_kelly = max(0.0, min(1.0, float(full_kelly)))
            risk_fraction_used = float(kelly_fraction) * full_kelly
        elif sizing_mode == "fractional_kelly_with_risk_ceiling":
            if kelly_fraction is None or risk_fraction is None:
                raise ValueError("risk_fraction and kelly_fraction are required for fractional_kelly_with_risk_ceiling")
            if price <= 0.0 or price >= 1.0:
                full_kelly = 0.0
            else:
                full_kelly = (model_win_prob - price) / (1.0 - price)
                full_kelly = max(0.0, min(1.0, float(full_kelly)))
            kelly_risk = float(kelly_fraction) * full_kelly
            risk_fraction_used = min(float(risk_fraction), kelly_risk)
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
        wins = 0
        losses = 0
        gross_profit = 0.0
        gross_loss = 0.0
        avg_ev = 0.0
        median_ev = 0.0
        max_drawdown = 0.0
        risk_fraction_avg = 0.0
        risk_fraction_min = 0.0
        risk_fraction_max = 0.0
    else:
        wins = int((out["win"] == 1).sum())
        losses = int((out["win"] == 0).sum())
        gross_profit = float(out.loc[out["win"] == 1, "pnl"].sum())
        gross_loss = float(-out.loc[out["win"] == 0, "pnl"].sum())
        avg_ev = float(out["ev"].mean())
        median_ev = float(out["ev"].median())
        max_drawdown = float(out["drawdown"].max())
        risk_fraction_avg = float(out["risk_fraction_used"].mean())
        risk_fraction_min = float(out["risk_fraction_used"].min())
        risk_fraction_max = float(out["risk_fraction_used"].max())

    station_counts = out["station_id"].value_counts().to_dict() if not out.empty else {}
    side_counts = out["side"].value_counts().to_dict() if not out.empty else {}
    summary = {
        "sizing_mode": sizing_mode,
        "risk_fraction": None if risk_fraction is None else float(risk_fraction),
        "kelly_fraction": None if kelly_fraction is None else float(kelly_fraction),
        "stake_cap_usd": float(stake_cap_usd),
        "start_balance": float(start_balance),
        "trades": int(len(out)),
        "wins": wins,
        "losses": losses,
        "win_rate": (float(wins) / float(len(out))) if len(out) > 0 else 0.0,
        "profit_factor": (float(gross_profit) / float(gross_loss)) if gross_loss > 0 else None,
        "final_balance": float(balance),
        "total_pnl": float(balance - start_balance),
        "avg_ev_at_trade": avg_ev,
        "median_ev_at_trade": median_ev,
        "max_drawdown": max_drawdown,
        "station_counts": station_counts,
        "side_counts": side_counts,
        "risk_fraction_used_avg": risk_fraction_avg,
        "risk_fraction_used_min": risk_fraction_min,
        "risk_fraction_used_max": risk_fraction_max,
        "stake_cap_breach_count": int((out["stake"] > (float(stake_cap_usd) + 1e-9)).sum()) if not out.empty else 0,
    }
    return out, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hot parameter sweep for co-joined KNYC/KMIA MOS backtesting with SQLite export."
    )
    p.add_argument("--pred-dev-knyc", default=r"D:\Ahmed\data\kalshi\Experiments\MOS\03_blends\blend_12\dev_predictions.parquet")
    p.add_argument("--pred-test-knyc", default=r"D:\Ahmed\data\kalshi\Experiments\MOS\03_blends\blend_12\test_predictions.parquet")
    p.add_argument("--pred-dev-kmia", default=r"D:\Ahmed\data\kalshi\Experiments\MOS_KMIA\03_blends\blend_12\dev_predictions.parquet")
    p.add_argument("--pred-test-kmia", default=r"D:\Ahmed\data\kalshi\Experiments\MOS_KMIA\03_blends\blend_12\test_predictions.parquet")
    p.add_argument("--kalshi-root-knyc", default=r"D:\Ahmed\data\kalshi\kalshi_history\kxhighny_2024_10_01_to_2025_12_31")
    p.add_argument("--kalshi-root-kmia", default=r"D:\Ahmed\data\kalshi\kalshi_history\kxhighmia_2024_10_01_to_2025_12_31")
    p.add_argument("--start-date", default="2024-10-01")
    p.add_argument("--end-date", default="2025-09-30")
    p.add_argument("--entry-hour-z", type=int, default=12)
    p.add_argument("--entry-minute-z", type=int, default=0)
    p.add_argument("--min-entry-minutes-after-open", type=int, default=30)
    p.add_argument("--min-market-price", type=float, default=0.0)
    p.add_argument("--start-balance", type=float, default=2700.0)
    p.add_argument("--stake-cap-usd", type=float, default=700.0)
    p.add_argument("--sqlite-dir", default=r"D:\Ahmed\data\sqlite")
    p.add_argument("--out-dir", default=r"D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest\hot_param_sweep")
    p.add_argument("--run-prefix", default="cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_hot_2025")
    p.add_argument("--selection-risk-fraction", type=float, default=0.03)
    p.add_argument("--ev-start", type=float, default=min(DEFAULT_EV_VALUES))
    p.add_argument("--ev-end", type=float, default=max(DEFAULT_EV_VALUES))
    p.add_argument("--ev-step", type=float, default=0.10)
    p.add_argument("--win-start", type=float, default=min(DEFAULT_WIN_VALUES))
    p.add_argument("--win-end", type=float, default=max(DEFAULT_WIN_VALUES))
    p.add_argument("--win-step", type=float, default=0.10)
    p.add_argument("--risk-start", type=float, default=min(DEFAULT_RISK_VALUES))
    p.add_argument("--risk-end", type=float, default=max(DEFAULT_RISK_VALUES))
    p.add_argument("--risk-step", type=float, default=0.015)
    p.add_argument("--kelly-start", type=float, default=min(DEFAULT_KELLY_VALUES))
    p.add_argument("--kelly-end", type=float, default=max(DEFAULT_KELLY_VALUES))
    p.add_argument("--kelly-step", type=float, default=0.02)
    p.add_argument(
        "--sizing-modes",
        default="fixed_risk,fractional_kelly,fractional_kelly_with_risk_ceiling",
        help="Comma-separated subset of: fixed_risk,fractional_kelly,fractional_kelly_with_risk_ceiling",
    )
    p.add_argument(
        "--ranking-scope",
        choices=["all_modes", "fixed_risk", "fractional_kelly", "fractional_kelly_with_risk_ceiling"],
        default="fractional_kelly_with_risk_ceiling",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ev_values = _build_float_grid(float(args.ev_start), float(args.ev_end), float(args.ev_step))
    win_values = _build_float_grid(float(args.win_start), float(args.win_end), float(args.win_step))
    risk_values = _build_float_grid(float(args.risk_start), float(args.risk_end), float(args.risk_step))
    kelly_values = _build_float_grid(float(args.kelly_start), float(args.kelly_end), float(args.kelly_step))
    sizing_modes = [x.strip() for x in str(args.sizing_modes).split(",") if str(x).strip()]
    invalid_modes = [m for m in sizing_modes if m not in ALL_SIZING_MODES]
    if invalid_modes:
        raise ValueError(f"Invalid sizing modes: {invalid_modes}. valid={ALL_SIZING_MODES}")
    if not sizing_modes:
        raise ValueError("At least one sizing mode must be enabled")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.out_dir) / f"{args.run_prefix}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    sqlite_dir = Path(args.sqlite_dir)
    sqlite_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = sqlite_dir / f"mos_hot_backtest_2025_{run_id}.sqlite"

    pred_maps = {
        "KNYC": cojoined.load_predictions(Path(args.pred_dev_knyc), Path(args.pred_test_knyc)),
        "KMIA": cojoined.load_predictions(Path(args.pred_dev_kmia), Path(args.pred_test_kmia)),
    }
    market_indices = {
        "KNYC": cojoined.build_market_index(Path(args.kalshi_root_knyc), "KNYC"),
        "KMIA": cojoined.build_market_index(Path(args.kalshi_root_kmia), "KMIA"),
    }

    base_rows: List[Dict] = []
    combo_rows: List[Dict] = []

    entry_rule_text = (
        f"Gate: T-1 {int(args.entry_hour_z):02d}:{int(args.entry_minute_z):02d}Z. "
        f"Effective cutoff per station: max(gate, market_open + {int(args.min_entry_minutes_after_open)}m). "
        "Entry timestamp: first global (KNYC/KMIA union) timestamp with an eligible side-aware candidate."
    )
    side_price_text = "YES side uses normalized bucket price; NO side uses 1-YES."
    fixed_rule_text = "stake=min(balance_before*risk_fraction, stake_cap_usd)"
    kelly_rule_text = "full_kelly=(q-p)/(1-p) clamped to [0,1]; risk_used=kelly_fraction*full_kelly; stake=min(balance_before*risk_used, stake_cap_usd)"
    kelly_ceiling_rule_text = "full_kelly=(q-p)/(1-p) clamped to [0,1]; risk_used=min(risk_fraction, kelly_fraction*full_kelly); stake=min(balance_before*risk_used, stake_cap_usd)"

    for ev_min in ev_values:
        for win_min in win_values:
            trades_df, base_summary, day_debug = cojoined.run_backtest(
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
            base_summary_path = run_dir / f"summary_base_ev{ev_tag}_win{win_tag}.json"
            base_sanity_path = run_dir / f"sanity_base_ev{ev_tag}_win{win_tag}.json"
            base_summary_payload = dict(base_summary)
            base_summary_payload["rule_entry"] = entry_rule_text
            base_summary_payload["rule_side_price"] = side_price_text
            base_summary_payload["rule_selection_sizing"] = f"selection-run fixed risk fraction={float(args.selection_risk_fraction)} with cap={float(args.stake_cap_usd)}"
            base_summary_payload["run_id"] = run_id
            base_summary_path.write_text(json.dumps(_sanitize_for_json(base_summary_payload), indent=2), encoding="utf-8")
            base_sanity_path.write_text(json.dumps(_sanitize_for_json(sanity), indent=2), encoding="utf-8")

            base_rows.append(
                {
                    "run_id": run_id,
                    "ev_min": float(ev_min),
                    "win_min": float(win_min),
                    "trades": int(base_summary.get("trades", 0)),
                    "days_without_trade_candidate": int(base_summary.get("days_without_trade_candidate", 0)),
                    "base_summary_json_path": str(base_summary_path),
                    "base_sanity_json_path": str(base_sanity_path),
                    "sanity_passes_all_checks": bool(sanity.get("passes_all_checks", False)),
                    "sanity_checked_trades": int(sanity.get("checked_trades", 0)),
                }
            )

            if not bool(sanity.get("passes_all_checks", False)):
                raise RuntimeError(f"Sanity failed for ev_min={ev_min}, win_min={win_min}. See {base_sanity_path}")

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

            if "fixed_risk" in sizing_modes:
                for risk_fraction in risk_values:
                    _, summary_fixed = _simulate_bankroll(
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
                            "side_price_rule": side_price_text,
                            "stake_rule": fixed_rule_text,
                            "summary_json_path": str(base_summary_path),
                            "sanity_json_path": str(base_sanity_path),
                            "trades": int(summary_fixed["trades"]),
                            "wins": int(summary_fixed["wins"]),
                            "losses": int(summary_fixed["losses"]),
                            "win_rate": float(summary_fixed["win_rate"]),
                            "profit_factor": summary_fixed["profit_factor"],
                            "final_balance": float(summary_fixed["final_balance"]),
                            "total_pnl": float(summary_fixed["total_pnl"]),
                            "max_drawdown": float(summary_fixed["max_drawdown"]),
                            "avg_ev_at_trade": float(summary_fixed["avg_ev_at_trade"]),
                            "median_ev_at_trade": float(summary_fixed["median_ev_at_trade"]),
                            "station_counts_json": json.dumps(summary_fixed["station_counts"], sort_keys=True),
                            "side_counts_json": json.dumps(summary_fixed["side_counts"], sort_keys=True),
                            "risk_fraction_used_avg": float(summary_fixed["risk_fraction_used_avg"]),
                            "risk_fraction_used_min": float(summary_fixed["risk_fraction_used_min"]),
                            "risk_fraction_used_max": float(summary_fixed["risk_fraction_used_max"]),
                            "stake_cap_breach_count": int(summary_fixed["stake_cap_breach_count"]),
                        }
                    )

            if "fractional_kelly" in sizing_modes:
                for kelly_fraction in kelly_values:
                    _, summary_kelly = _simulate_bankroll(
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
                            "side_price_rule": side_price_text,
                            "stake_rule": kelly_rule_text,
                            "summary_json_path": str(base_summary_path),
                            "sanity_json_path": str(base_sanity_path),
                            "trades": int(summary_kelly["trades"]),
                            "wins": int(summary_kelly["wins"]),
                            "losses": int(summary_kelly["losses"]),
                            "win_rate": float(summary_kelly["win_rate"]),
                            "profit_factor": summary_kelly["profit_factor"],
                            "final_balance": float(summary_kelly["final_balance"]),
                            "total_pnl": float(summary_kelly["total_pnl"]),
                            "max_drawdown": float(summary_kelly["max_drawdown"]),
                            "avg_ev_at_trade": float(summary_kelly["avg_ev_at_trade"]),
                            "median_ev_at_trade": float(summary_kelly["median_ev_at_trade"]),
                            "station_counts_json": json.dumps(summary_kelly["station_counts"], sort_keys=True),
                            "side_counts_json": json.dumps(summary_kelly["side_counts"], sort_keys=True),
                            "risk_fraction_used_avg": float(summary_kelly["risk_fraction_used_avg"]),
                            "risk_fraction_used_min": float(summary_kelly["risk_fraction_used_min"]),
                            "risk_fraction_used_max": float(summary_kelly["risk_fraction_used_max"]),
                            "stake_cap_breach_count": int(summary_kelly["stake_cap_breach_count"]),
                        }
                    )

            if "fractional_kelly_with_risk_ceiling" in sizing_modes:
                for risk_fraction in risk_values:
                    for kelly_fraction in kelly_values:
                        _, summary_kelly_ceiling = _simulate_bankroll(
                            trades_template=stream_template,
                            start_balance=float(args.start_balance),
                            stake_cap_usd=float(args.stake_cap_usd),
                            sizing_mode="fractional_kelly_with_risk_ceiling",
                            risk_fraction=float(risk_fraction),
                            kelly_fraction=float(kelly_fraction),
                        )
                        combo_rows.append(
                            {
                                "run_id": run_id,
                                "sizing_mode": "fractional_kelly_with_risk_ceiling",
                                "ev_min": float(ev_min),
                                "win_min": float(win_min),
                                "risk_fraction": float(risk_fraction),
                                "kelly_fraction": float(kelly_fraction),
                                "stake_cap_usd": float(args.stake_cap_usd),
                                "entry_rule": entry_rule_text,
                                "side_price_rule": side_price_text,
                                "stake_rule": kelly_ceiling_rule_text,
                                "summary_json_path": str(base_summary_path),
                                "sanity_json_path": str(base_sanity_path),
                                "trades": int(summary_kelly_ceiling["trades"]),
                                "wins": int(summary_kelly_ceiling["wins"]),
                                "losses": int(summary_kelly_ceiling["losses"]),
                                "win_rate": float(summary_kelly_ceiling["win_rate"]),
                                "profit_factor": summary_kelly_ceiling["profit_factor"],
                                "final_balance": float(summary_kelly_ceiling["final_balance"]),
                                "total_pnl": float(summary_kelly_ceiling["total_pnl"]),
                                "max_drawdown": float(summary_kelly_ceiling["max_drawdown"]),
                                "avg_ev_at_trade": float(summary_kelly_ceiling["avg_ev_at_trade"]),
                                "median_ev_at_trade": float(summary_kelly_ceiling["median_ev_at_trade"]),
                                "station_counts_json": json.dumps(summary_kelly_ceiling["station_counts"], sort_keys=True),
                                "side_counts_json": json.dumps(summary_kelly_ceiling["side_counts"], sort_keys=True),
                                "risk_fraction_used_avg": float(summary_kelly_ceiling["risk_fraction_used_avg"]),
                                "risk_fraction_used_min": float(summary_kelly_ceiling["risk_fraction_used_min"]),
                                "risk_fraction_used_max": float(summary_kelly_ceiling["risk_fraction_used_max"]),
                                "stake_cap_breach_count": int(summary_kelly_ceiling["stake_cap_breach_count"]),
                            }
                        )

    base_df = pd.DataFrame(base_rows)
    combo_df = pd.DataFrame(combo_rows)

    with sqlite3.connect(sqlite_path) as conn:
        combo_df.to_sql("hot_backtest_combo_results", conn, if_exists="replace", index=False)
        base_df.to_sql("hot_backtest_base_streams", conn, if_exists="replace", index=False)
        rank_df = combo_df.copy()
        rank_df = rank_df[rank_df["profit_factor"].notna()].copy()
        if not rank_df.empty:
            if str(args.ranking_scope) != "all_modes":
                rank_df = rank_df[rank_df["sizing_mode"] == str(args.ranking_scope)].copy()
            if not rank_df.empty:
                rank_df["pf_log_component_raw"] = rank_df["profit_factor"].map(lambda x: float(np.log1p(float(x))))
                max_pf_log = float(rank_df["pf_log_component_raw"].max()) if not rank_df.empty else 1.0
                if max_pf_log <= 0:
                    max_pf_log = 1.0
                rank_df["pf_component"] = rank_df["pf_log_component_raw"] / max_pf_log
                rank_df["win_component"] = rank_df["win_rate"].clip(lower=0.0, upper=1.0)
                rank_df["drawdown_component"] = (1.0 - rank_df["max_drawdown"]).clip(lower=0.0, upper=1.0)
                rank_df["composite_score_pf_win_lowdd"] = (
                    rank_df["pf_component"] * rank_df["win_component"] * rank_df["drawdown_component"]
                ) ** (1.0 / 3.0)
        rank_df.to_sql("hot_backtest_ranked_scores", conn, if_exists="replace", index=False)
        meta_df = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "start_date": str(args.start_date),
                    "end_date": str(args.end_date),
                    "entry_hour_z": int(args.entry_hour_z),
                    "entry_minute_z": int(args.entry_minute_z),
                    "min_entry_minutes_after_open": int(args.min_entry_minutes_after_open),
                    "min_market_price": float(args.min_market_price),
                    "start_balance": float(args.start_balance),
                    "stake_cap_usd": float(args.stake_cap_usd),
                    "win_values_json": json.dumps(win_values),
                    "ev_values_json": json.dumps(ev_values),
                    "risk_values_json": json.dumps(risk_values),
                    "kelly_values_json": json.dumps(kelly_values),
                    "sizing_modes_json": json.dumps(sizing_modes),
                    "ranking_scope": str(args.ranking_scope),
                    "entry_rule": entry_rule_text,
                    "side_price_rule": side_price_text,
                    "base_stream_count": int(len(base_df)),
                    "combo_count": int(len(combo_df)),
                    "out_dir": str(run_dir),
                }
            ]
        )
        meta_df.to_sql("hot_backtest_run_meta", conn, if_exists="replace", index=False)

    sanity_fail_count = int((~base_df["sanity_passes_all_checks"]).sum()) if not base_df.empty else 0
    run_sanity = {
        "run_id": run_id,
        "passes_all_checks": sanity_fail_count == 0 and int(combo_df["stake_cap_breach_count"].sum()) == 0,
        "base_stream_sanity_fail_count": sanity_fail_count,
        "combo_stake_cap_breach_count_total": int(combo_df["stake_cap_breach_count"].sum()) if not combo_df.empty else 0,
        "base_stream_count": int(len(base_df)),
        "combo_count": int(len(combo_df)),
        "sqlite_path": str(sqlite_path),
    }
    run_sanity_path = run_dir / "run_sanity.json"
    run_sanity_path.write_text(json.dumps(_sanitize_for_json(run_sanity), indent=2), encoding="utf-8")

    top_rows: List[Dict] = []
    if not combo_df.empty:
        for mode in ["fixed_risk", "fractional_kelly", "fractional_kelly_with_risk_ceiling"]:
            m = combo_df[combo_df["sizing_mode"] == mode]
            if m.empty:
                continue
            top = m.sort_values(["final_balance", "profit_factor"], ascending=[False, False]).head(3)
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
                        "total_pnl": float(r["total_pnl"]),
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
        "start_date": str(args.start_date),
        "end_date": str(args.end_date),
        "entry_rule": entry_rule_text,
        "side_price_rule": side_price_text,
        "stake_rules": {
            "fixed_risk": fixed_rule_text,
            "fractional_kelly": kelly_rule_text,
            "fractional_kelly_with_risk_ceiling": kelly_ceiling_rule_text,
        },
        "base_stream_count": int(len(base_df)),
        "combo_count": int(len(combo_df)),
        "grid": {
            "win_values": win_values,
            "ev_values": ev_values,
            "risk_values": risk_values,
            "kelly_values": kelly_values,
            "sizing_modes": sizing_modes,
            "ranking_scope": str(args.ranking_scope),
        },
        "top_rows_by_mode": top_rows,
    }
    run_summary_path = run_dir / "run_summary.json"
    run_summary_path.write_text(json.dumps(_sanitize_for_json(run_summary), indent=2), encoding="utf-8")

    print("RUN_ID", run_id)
    print("WROTE_SQLITE", sqlite_path)
    print("WROTE_RUN_SUMMARY", run_summary_path)
    print("WROTE_RUN_SANITY", run_sanity_path)
    print("BASE_STREAMS", len(base_df))
    print("COMBO_ROWS", len(combo_df))


if __name__ == "__main__":
    main()
