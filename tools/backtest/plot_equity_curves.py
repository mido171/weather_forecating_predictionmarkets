from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    trades_path: Path
    summary_path: Path


def _load_run(run_dir: Path) -> RunArtifacts:
    run_dir = run_dir.resolve()
    trades_path = run_dir / "trades.csv"
    summary_path = run_dir / "summary.json"
    if not trades_path.exists():
        raise FileNotFoundError(f"Missing trades.csv: {trades_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json: {summary_path}")
    return RunArtifacts(run_dir=run_dir, trades_path=trades_path, summary_path=summary_path)


def _build_daily_equity(trades: pd.DataFrame, start: str, end: str, start_balance: float) -> pd.DataFrame:
    df = trades.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["balance_after"] = pd.to_numeric(df["balance_after"], errors="coerce")
    daily = df.groupby("date", as_index=False)["balance_after"].last()

    idx = pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq="D")
    out = pd.DataFrame(index=idx)
    out.index.name = "date"
    out = out.merge(daily.set_index("date"), left_index=True, right_index=True, how="left")
    out["balance_after"] = out["balance_after"].ffill()
    out["balance_after"] = out["balance_after"].fillna(float(start_balance))
    return out


def _plot_single(run_dir: Path, equity: pd.DataFrame, trades: pd.DataFrame, title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(12, 5.5), dpi=160)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(equity.index, equity["balance_after"], lw=2.0, color="#1f77b4")

    # Mark trade entry days.
    entered = trades[trades["entry_time"].notna()].copy()
    if not entered.empty:
        entered["date"] = pd.to_datetime(entered["date"])
        entered["balance_after"] = pd.to_numeric(entered["balance_after"], errors="coerce")
        ax.scatter(
            entered["date"],
            entered["balance_after"],
            s=18,
            marker="o",
            color="#d62728",
            alpha=0.85,
            zorder=5,
            label="trade entered",
        )
        ax.legend(loc="upper left", frameon=False)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Balance ($)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _pretty_name(summary: dict) -> str:
    mode = summary.get("bridge_mode", "unknown")
    trailing = summary.get("bridge_trailing_years")
    if mode == "trailing" and trailing:
        mode = f"trailing {trailing}y"
    start = summary.get("backtest_start")
    end = summary.get("backtest_end")
    gate = summary.get("p_hit_gate")
    win = summary.get("min_win_prob")
    edge = summary.get("edge_prob")
    end_bal = summary.get("end_balance")
    entered = summary.get("entered_trades")
    win_rate = summary.get("win_rate")
    pf = summary.get("profit_factor")
    return (
        f"KMIA  Equity Curve ({mode})  {start}..{end}\n"
        f"p_hit_gate={gate}  min_win={win}  edge={edge}  trades={entered}  "
        f"win%={win_rate:.3f}  PF={pf:.2f}  end=${end_bal:,.2f}"
    )


def _timestamp_tag_utc() -> str:
    # Windows-safe: ':' is not allowed in path names. This is the same information as YYMMDD:HH:MM:SS.
    return datetime.now(timezone.utc).strftime("%y%m%d-%H-%M-%S")


def _sanitize_tag(tag: str) -> str:
    # Be resilient if caller gives YYMMDD:HH:MM:SS. Windows disallows ':' so normalize.
    return tag.strip().replace(":", "-").replace("/", "-").replace("\\", "-")


def _sanitize_component(tag: str) -> str:
    return _sanitize_tag(tag).replace(" ", "_")


def _fmt_float_tag(value: object, decimals: int = 3) -> str:
    try:
        v = float(value)  # type: ignore[arg-type]
    except Exception:
        return "na"
    s = f"{v:.{decimals}f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _run_label(summary: dict, fallback: str) -> str:
    mode = str(summary.get("bridge_mode") or "")
    trailing = summary.get("bridge_trailing_years")
    if mode == "trailing" and trailing:
        mode = f"trailing{int(trailing)}y"

    risk_model = str(summary.get("risk_model") or "")
    if risk_model == "fixed":
        rf = summary.get("fixed_risk_fraction", summary.get("risk_fraction"))
        risk = f"fixed_risk{_fmt_float_tag(rf)}"
    elif risk_model == "kelly":
        kf = summary.get("kelly_fraction")
        risk = f"kelly_frac{_fmt_float_tag(kf)}"
    else:
        risk = fallback

    parts = [p for p in [mode, risk] if p]
    return "_".join(parts) if parts else fallback


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plot equity curves for one or more backtest run directories.")
    parser.add_argument("--run-dir", action="append", required=True, help="Backtest run directory (contains trades.csv + summary.json)")
    parser.add_argument(
        "--out-root",
        default=str(REPO / "backtests" / "kmia_kalshi" / "plots"),
        help="Root output directory for generated images",
    )
    parser.add_argument(
        "--timestamp-tag",
        default=None,
        help="Optional tag to prefix output folders. Format example: YYMMDD:HH:MM:SS (will be sanitized for Windows).",
    )
    parser.add_argument("--compare", action="store_true", default=True, help="Write an overlay comparison plot (default: true)")
    args = parser.parse_args(argv)

    runs = [_load_run(Path(p)) for p in args.run_dir]
    ts_tag = _sanitize_tag(args.timestamp_tag) if args.timestamp_tag else _timestamp_tag_utc()
    out_root = Path(args.out_root).resolve()

    series = []
    labels = []
    for run in runs:
        trades = pd.read_csv(run.trades_path)
        summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
        equity = _build_daily_equity(
            trades,
            start=str(summary["backtest_start"]),
            end=str(summary["backtest_end"]),
            start_balance=float(summary["start_balance"]),
        )

        title = _pretty_name(summary)
        label = _run_label(summary, fallback=run.run_dir.name)
        per_run_dir = out_root / f"{ts_tag}__{_sanitize_component(label)}"
        _plot_single(run.run_dir, equity, trades, title=title, out_path=per_run_dir / "equity_curve.png")

        series.append(equity["balance_after"])
        labels.append(label)

    if args.compare and len(series) >= 2:
        fig = plt.figure(figsize=(12, 5.5), dpi=160)
        ax = fig.add_subplot(1, 1, 1)
        for s, label in zip(series, labels):
            ax.plot(s.index, s.values, lw=2.0, label=label)
        ax.set_title("KMIA Equity Curves (Comparison)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Balance ($)")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper left", frameon=False)
        fig.tight_layout()
        cmp_dir = out_root / f"{ts_tag}__compare"
        cmp_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(cmp_dir / "equity_curve_compare.png")
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
