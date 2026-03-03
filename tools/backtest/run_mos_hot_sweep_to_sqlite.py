from __future__ import annotations

import argparse
import json
import math
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


def _parse_float_list(raw: str) -> List[float]:
    vals: List[float] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError(f"Expected at least one float value in '{raw}'")
    return vals


def _fmt_prob_tag(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def _fmt_pct_tag(x: float) -> str:
    pct = x * 100.0
    s = f"{pct:.6f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_cmd(cmd: List[str], cwd: Path) -> Tuple[int, str, str]:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        f = float(v)
        if math.isnan(f):
            return float(default)
        return f
    except Exception:
        return float(default)


def _safe_int(v: object, default: int = 0) -> int:
    try:
        if v is None:
            return int(default)
        return int(v)
    except Exception:
        return int(default)


def _risk_fraction_for_trade(
    risk_mode: str,
    market_price: float,
    model_win_prob: float,
    fixed_risk_fraction: Optional[float],
    kelly_fraction: Optional[float],
) -> Tuple[float, Optional[float]]:
    if risk_mode == "fixed":
        return max(0.0, _safe_float(fixed_risk_fraction, 0.0)), None
    if risk_mode != "fractional_kelly":
        raise ValueError(f"Unsupported risk_mode: {risk_mode}")
    p = _safe_float(market_price, 0.0)
    q = _safe_float(model_win_prob, 0.0)
    if p <= 0.0 or p >= 1.0:
        full_kelly = 0.0
    else:
        full_kelly = (q - p) / (1.0 - p)
    full_kelly = min(max(full_kelly, 0.0), 1.0)
    frac = _safe_float(kelly_fraction, 0.0) * full_kelly
    return float(frac), float(full_kelly)


def recompute_metrics(
    trades_df: pd.DataFrame,
    *,
    risk_mode: str,
    fixed_risk_fraction: Optional[float],
    kelly_fraction: Optional[float],
    start_balance: float,
    stake_cap_usd: float,
) -> Dict[str, object]:
    if trades_df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": None,
            "start_balance": float(start_balance),
            "final_balance": float(start_balance),
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "avg_ev_at_trade": None,
            "median_ev_at_trade": None,
            "mean_entry_price": None,
            "avg_rr": None,
            "station_counts": {},
            "side_counts": {},
            "gross_profit": 0.0,
            "gross_loss_abs": 0.0,
        }

    bal = float(start_balance)
    peak = bal
    max_dd = 0.0

    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss_abs = 0.0

    station_counts: Dict[str, int] = {}
    side_counts: Dict[str, int] = {}
    rr_vals: List[float] = []

    # Respect chronological trade order.
    t = trades_df.copy()
    if "entry_timestamp_utc" in t.columns:
        t["_entry_ts"] = pd.to_datetime(t["entry_timestamp_utc"], utc=True, errors="coerce")
        t = t.sort_values(["_entry_ts", "target_date_local", "station_id"], kind="stable")
    else:
        t = t.sort_values(["target_date_local", "station_id"], kind="stable")

    for row in t.itertuples(index=False):
        p = _safe_float(getattr(row, "market_price", 0.0), 0.0)
        q = _safe_float(getattr(row, "model_win_prob", 0.0), 0.0)
        win_int = _safe_int(getattr(row, "win", 0), 0)
        is_win = bool(win_int == 1)

        risk_fraction_used, _ = _risk_fraction_for_trade(
            risk_mode=risk_mode,
            market_price=p,
            model_win_prob=q,
            fixed_risk_fraction=fixed_risk_fraction,
            kelly_fraction=kelly_fraction,
        )
        stake = min(max(0.0, bal * risk_fraction_used), float(stake_cap_usd))
        shares = (stake / p) if p > 0.0 else 0.0
        pnl = (shares * (1.0 - p)) if is_win else (-stake)

        bal = bal + pnl
        if bal > peak:
            peak = bal
        dd = 0.0 if peak <= 0.0 else (peak - bal) / peak
        if dd > max_dd:
            max_dd = dd

        if is_win:
            wins += 1
            gross_profit += pnl
        else:
            losses += 1
            gross_loss_abs += (-pnl)

        st = str(getattr(row, "station_id", "")).strip()
        sd = str(getattr(row, "side", "")).strip()
        station_counts[st] = station_counts.get(st, 0) + 1
        side_counts[sd] = side_counts.get(sd, 0) + 1

        if p > 0.0:
            rr_vals.append((1.0 - p) / p)

    trades = int(len(t))
    win_rate = float(wins / trades) if trades > 0 else 0.0
    if gross_loss_abs <= 0.0 and gross_profit > 0.0:
        profit_factor = float("inf")
    elif gross_loss_abs <= 0.0:
        profit_factor = None
    else:
        profit_factor = float(gross_profit / gross_loss_abs)

    avg_ev = float(pd.to_numeric(t["ev"], errors="coerce").mean()) if "ev" in t.columns else None
    median_ev = float(pd.to_numeric(t["ev"], errors="coerce").median()) if "ev" in t.columns else None
    mean_entry_price = float(pd.to_numeric(t["market_price"], errors="coerce").mean()) if "market_price" in t.columns else None
    avg_rr = float(pd.Series(rr_vals, dtype=float).mean()) if rr_vals else None

    return {
        "trades": trades,
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": float(win_rate),
        "profit_factor": None if profit_factor is None else float(profit_factor),
        "start_balance": float(start_balance),
        "final_balance": float(bal),
        "total_pnl": float(bal - start_balance),
        "max_drawdown": float(max_dd),
        "avg_ev_at_trade": avg_ev,
        "median_ev_at_trade": median_ev,
        "mean_entry_price": mean_entry_price,
        "avg_rr": avg_rr,
        "station_counts": station_counts,
        "side_counts": side_counts,
        "gross_profit": float(gross_profit),
        "gross_loss_abs": float(gross_loss_abs),
    }


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sweep_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS hot_backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generated_at_utc TEXT NOT NULL,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            prediction_source TEXT NOT NULL,
            entry_hour_z INTEGER NOT NULL,
            entry_minute_z INTEGER NOT NULL,
            min_entry_minutes_after_open INTEGER NOT NULL,
            min_market_price REAL NOT NULL,
            ev_min REAL NOT NULL,
            win_min REAL NOT NULL,
            risk_mode TEXT NOT NULL,
            fixed_risk_fraction REAL,
            kelly_fraction REAL,
            stake_cap_usd REAL NOT NULL,
            start_balance REAL NOT NULL,
            entry_gate_rule TEXT NOT NULL,
            stake_rule TEXT NOT NULL,
            trades INTEGER NOT NULL,
            wins INTEGER NOT NULL,
            losses INTEGER NOT NULL,
            win_rate REAL NOT NULL,
            profit_factor REAL,
            final_balance REAL NOT NULL,
            total_pnl REAL NOT NULL,
            max_drawdown REAL NOT NULL,
            avg_ev_at_trade REAL,
            median_ev_at_trade REAL,
            mean_entry_price REAL,
            avg_rr REAL,
            gross_profit REAL NOT NULL,
            gross_loss_abs REAL NOT NULL,
            station_counts_json TEXT NOT NULL,
            side_counts_json TEXT NOT NULL,
            base_out_prefix TEXT NOT NULL,
            base_trades_csv_path TEXT NOT NULL,
            base_summary_json_path TEXT NOT NULL,
            base_sanity_json_path TEXT NOT NULL,
            base_sanity_passes_all_checks INTEGER NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hot_backtest_grid ON hot_backtest_runs(ev_min, win_min, risk_mode)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hot_backtest_perf ON hot_backtest_runs(final_balance DESC)")


def insert_meta(conn: sqlite3.Connection, key: str, value: object) -> None:
    conn.execute(
        "INSERT INTO sweep_meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(key), json.dumps(value, ensure_ascii=False)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run hot MOS co-joined EV/Win sweeps and write fixed-risk + fractional-Kelly summaries into SQLite."
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--prediction-source", choices=["parquet", "live-script"], default="parquet")
    parser.add_argument("--start-date", default="2024-10-01")
    parser.add_argument("--end-date", default="2025-09-30")
    parser.add_argument("--entry-hour-z", type=int, default=12)
    parser.add_argument("--entry-minute-z", type=int, default=0)
    parser.add_argument("--min-entry-minutes-after-open", type=int, default=30)
    parser.add_argument("--min-market-price", type=float, default=0.10)
    parser.add_argument("--start-balance", type=float, default=2700.0)
    parser.add_argument("--stake-cap-usd", type=float, default=700.0)
    parser.add_argument("--reference-risk-fraction", type=float, default=0.03)
    parser.add_argument("--win-grid", default="0.65,0.75,0.85,0.95")
    parser.add_argument("--ev-grid", default="0.15,0.25,0.35,0.45")
    parser.add_argument("--fixed-risk-grid", default="0.03,0.045,0.06,0.075")
    parser.add_argument("--kelly-fraction-grid", default="0.10,0.12,0.14,0.16,0.18,0.20")
    parser.add_argument("--out-dir", default=r"D:\Ahmed\data\kalshi\Experiments\MOS\05_backtest")
    parser.add_argument("--plots-dir", default=r"D:\Ahmed\data\kalshi\plots")
    parser.add_argument("--sqlite-path", default=r"D:\Ahmed\data\sqlite\mos_hot_backtest_2024_10_to_2025_09.sqlite")
    parser.add_argument("--sweep-tag", default="hotgrid_2024_10_to_2025_09")
    parser.add_argument("--overwrite-db", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    backtest_script = repo_root / "backtesting" / "mos_blend12_knyc_kmia_cojoined_audit.py"
    if not backtest_script.exists():
        raise FileNotFoundError(f"Backtest script missing: {backtest_script}")

    win_vals = sorted(_parse_float_list(args.win_grid))
    ev_vals = sorted(_parse_float_list(args.ev_grid))
    fixed_risk_vals = sorted(_parse_float_list(args.fixed_risk_grid))
    kelly_vals = sorted(_parse_float_list(args.kelly_fraction_grid))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = Path(args.sqlite_path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite_db and sqlite_path.exists():
        sqlite_path.unlink()

    entry_gate_rule = (
        "entry_timestamp_utc >= max((T-1 12:00Z), market_open_utc + "
        f"{int(args.min_entry_minutes_after_open)}m); first eligible global timestamp across KNYC/KMIA; "
        "tie-break: model_win_prob desc, ev desc, market_price asc, station_id asc."
    )

    rows: List[Tuple] = []
    base_runs = len(win_vals) * len(ev_vals)
    done_base = 0

    generated_at_utc = _utc_now_iso()
    t0_all = time.perf_counter()

    for win_min in win_vals:
        for ev_min in ev_vals:
            done_base += 1
            run_tag = (
                f"{args.sweep_tag}_"
                f"ev{_fmt_prob_tag(ev_min)}_"
                f"win{_fmt_pct_tag(win_min)}_"
                f"openplus{int(args.min_entry_minutes_after_open)}m_"
                f"minprice{_fmt_prob_tag(args.min_market_price)}_"
                f"cap{int(args.stake_cap_usd)}"
            )
            out_prefix = f"cojoined_blend12_{run_tag}_base"
            table_out = plots_dir / f"{out_prefix}_stockholm.csv"
            trades_path = out_dir / f"trades_{out_prefix}.csv"
            summary_path = out_dir / f"summary_{out_prefix}.json"
            sanity_path = out_dir / f"sanity_{out_prefix}.json"

            cmd = [
                str(args.python_exe),
                str(backtest_script),
                "--prediction-source",
                str(args.prediction_source),
                "--start-date",
                str(args.start_date),
                "--end-date",
                str(args.end_date),
                "--entry-hour-z",
                str(int(args.entry_hour_z)),
                "--entry-minute-z",
                str(int(args.entry_minute_z)),
                "--min-entry-minutes-after-open",
                str(int(args.min_entry_minutes_after_open)),
                "--ev-min",
                str(float(ev_min)),
                "--win-min",
                str(float(win_min)),
                "--min-market-price",
                str(float(args.min_market_price)),
                "--start-balance",
                str(float(args.start_balance)),
                "--risk-fraction",
                str(float(args.reference_risk_fraction)),
                "--stake-cap-usd",
                str(float(args.stake_cap_usd)),
                "--out-dir",
                str(out_dir),
                "--out-prefix",
                str(out_prefix),
                "--table-out",
                str(table_out),
            ]

            print(
                f"[{done_base}/{base_runs}] BASE_RUN_START "
                f"win_min={win_min:.2f} ev_min={ev_min:.2f} out_prefix={out_prefix}"
            )
            t0 = time.perf_counter()
            rc, stdout, stderr = _run_cmd(cmd, cwd=repo_root)
            elapsed = time.perf_counter() - t0
            if rc != 0:
                tail_out = (stdout or "")[-3000:]
                tail_err = (stderr or "")[-3000:]
                raise RuntimeError(
                    f"Base backtest failed for win={win_min} ev={ev_min} rc={rc}\n"
                    f"stdout_tail:\n{tail_out}\n\nstderr_tail:\n{tail_err}"
                )
            if not trades_path.exists() or not summary_path.exists() or not sanity_path.exists():
                raise FileNotFoundError(
                    "Expected base artifacts are missing for "
                    f"win={win_min} ev={ev_min}: {trades_path}, {summary_path}, {sanity_path}"
                )
            print(
                f"[{done_base}/{base_runs}] BASE_RUN_DONE "
                f"win_min={win_min:.2f} ev_min={ev_min:.2f} elapsed_sec={elapsed:.1f}"
            )

            trades_df = pd.read_csv(trades_path)
            base_summary = _load_json(summary_path)
            base_sanity = _load_json(sanity_path)
            base_sanity_pass = 1 if bool(base_sanity.get("passes_all_checks", False)) else 0

            for rf in fixed_risk_vals:
                metrics = recompute_metrics(
                    trades_df=trades_df,
                    risk_mode="fixed",
                    fixed_risk_fraction=rf,
                    kelly_fraction=None,
                    start_balance=float(args.start_balance),
                    stake_cap_usd=float(args.stake_cap_usd),
                )
                rows.append(
                    (
                        generated_at_utc,
                        str(args.start_date),
                        str(args.end_date),
                        str(args.prediction_source),
                        int(args.entry_hour_z),
                        int(args.entry_minute_z),
                        int(args.min_entry_minutes_after_open),
                        float(args.min_market_price),
                        float(ev_min),
                        float(win_min),
                        "fixed",
                        float(rf),
                        None,
                        float(args.stake_cap_usd),
                        float(args.start_balance),
                        entry_gate_rule,
                        f"stake=min(balance_before*{rf:.6f}, {float(args.stake_cap_usd):.2f})",
                        int(metrics["trades"]),
                        int(metrics["wins"]),
                        int(metrics["losses"]),
                        float(metrics["win_rate"]),
                        metrics["profit_factor"],
                        float(metrics["final_balance"]),
                        float(metrics["total_pnl"]),
                        float(metrics["max_drawdown"]),
                        metrics["avg_ev_at_trade"],
                        metrics["median_ev_at_trade"],
                        metrics["mean_entry_price"],
                        metrics["avg_rr"],
                        float(metrics["gross_profit"]),
                        float(metrics["gross_loss_abs"]),
                        json.dumps(metrics["station_counts"], ensure_ascii=False, sort_keys=True),
                        json.dumps(metrics["side_counts"], ensure_ascii=False, sort_keys=True),
                        out_prefix,
                        str(trades_path),
                        str(summary_path),
                        str(sanity_path),
                        int(base_sanity_pass),
                    )
                )

            for kf in kelly_vals:
                metrics = recompute_metrics(
                    trades_df=trades_df,
                    risk_mode="fractional_kelly",
                    fixed_risk_fraction=None,
                    kelly_fraction=kf,
                    start_balance=float(args.start_balance),
                    stake_cap_usd=float(args.stake_cap_usd),
                )
                rows.append(
                    (
                        generated_at_utc,
                        str(args.start_date),
                        str(args.end_date),
                        str(args.prediction_source),
                        int(args.entry_hour_z),
                        int(args.entry_minute_z),
                        int(args.min_entry_minutes_after_open),
                        float(args.min_market_price),
                        float(ev_min),
                        float(win_min),
                        "fractional_kelly",
                        None,
                        float(kf),
                        float(args.stake_cap_usd),
                        float(args.start_balance),
                        entry_gate_rule,
                        (
                            "full_kelly=clamp((q-p)/(1-p),0,1); "
                            f"risk_fraction_used={kf:.6f}*full_kelly; "
                            f"stake=min(balance_before*risk_fraction_used, {float(args.stake_cap_usd):.2f})"
                        ),
                        int(metrics["trades"]),
                        int(metrics["wins"]),
                        int(metrics["losses"]),
                        float(metrics["win_rate"]),
                        metrics["profit_factor"],
                        float(metrics["final_balance"]),
                        float(metrics["total_pnl"]),
                        float(metrics["max_drawdown"]),
                        metrics["avg_ev_at_trade"],
                        metrics["median_ev_at_trade"],
                        metrics["mean_entry_price"],
                        metrics["avg_rr"],
                        float(metrics["gross_profit"]),
                        float(metrics["gross_loss_abs"]),
                        json.dumps(metrics["station_counts"], ensure_ascii=False, sort_keys=True),
                        json.dumps(metrics["side_counts"], ensure_ascii=False, sort_keys=True),
                        out_prefix,
                        str(trades_path),
                        str(summary_path),
                        str(sanity_path),
                        int(base_sanity_pass),
                    )
                )

    elapsed_all = time.perf_counter() - t0_all
    print(f"SWEEP_DONE base_runs={base_runs} total_rows={len(rows)} elapsed_sec={elapsed_all:.1f}")

    with sqlite3.connect(str(sqlite_path)) as conn:
        init_db(conn)
        insert_meta(conn, "generated_at_utc", generated_at_utc)
        insert_meta(conn, "script", str(Path(__file__).resolve()))
        insert_meta(conn, "repo_root", str(repo_root))
        insert_meta(conn, "start_date", str(args.start_date))
        insert_meta(conn, "end_date", str(args.end_date))
        insert_meta(conn, "prediction_source", str(args.prediction_source))
        insert_meta(conn, "entry_hour_z", int(args.entry_hour_z))
        insert_meta(conn, "entry_minute_z", int(args.entry_minute_z))
        insert_meta(conn, "min_entry_minutes_after_open", int(args.min_entry_minutes_after_open))
        insert_meta(conn, "min_market_price", float(args.min_market_price))
        insert_meta(conn, "stake_cap_usd", float(args.stake_cap_usd))
        insert_meta(conn, "start_balance", float(args.start_balance))
        insert_meta(conn, "reference_risk_fraction", float(args.reference_risk_fraction))
        insert_meta(conn, "win_grid", win_vals)
        insert_meta(conn, "ev_grid", ev_vals)
        insert_meta(conn, "fixed_risk_grid", fixed_risk_vals)
        insert_meta(conn, "kelly_fraction_grid", kelly_vals)
        insert_meta(conn, "entry_gate_rule", entry_gate_rule)
        insert_meta(conn, "total_result_rows", len(rows))

        conn.execute("DELETE FROM hot_backtest_runs WHERE generated_at_utc = ?", (generated_at_utc,))
        conn.executemany(
            """
            INSERT INTO hot_backtest_runs (
                generated_at_utc,
                period_start,
                period_end,
                prediction_source,
                entry_hour_z,
                entry_minute_z,
                min_entry_minutes_after_open,
                min_market_price,
                ev_min,
                win_min,
                risk_mode,
                fixed_risk_fraction,
                kelly_fraction,
                stake_cap_usd,
                start_balance,
                entry_gate_rule,
                stake_rule,
                trades,
                wins,
                losses,
                win_rate,
                profit_factor,
                final_balance,
                total_pnl,
                max_drawdown,
                avg_ev_at_trade,
                median_ev_at_trade,
                mean_entry_price,
                avg_rr,
                gross_profit,
                gross_loss_abs,
                station_counts_json,
                side_counts_json,
                base_out_prefix,
                base_trades_csv_path,
                base_summary_json_path,
                base_sanity_json_path,
                base_sanity_passes_all_checks
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?
            )
            """,
            rows,
        )
        conn.commit()

        top_df = pd.read_sql_query(
            """
            SELECT
                risk_mode,
                fixed_risk_fraction,
                kelly_fraction,
                win_min,
                ev_min,
                trades,
                win_rate,
                profit_factor,
                final_balance,
                total_pnl,
                max_drawdown
            FROM hot_backtest_runs
            WHERE generated_at_utc = ?
            ORDER BY final_balance DESC
            LIMIT 10
            """,
            conn,
            params=(generated_at_utc,),
        )
        print("TOP_10_BY_FINAL_BALANCE")
        print(top_df.to_string(index=False))

    print(f"WROTE_SQLITE {sqlite_path}")


if __name__ == "__main__":
    main()
