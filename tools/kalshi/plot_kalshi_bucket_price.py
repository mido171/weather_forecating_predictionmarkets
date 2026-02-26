from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DEFAULT_KALSHI_DIR = REPO / "data" / "kalshi_backtest_data"
DEFAULT_OUT_DIR = REPO / "artifacts" / "kalshi_bucket_plots"


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True)
class BucketSpec:
    kind: str  # "range", "le", "ge"
    a: float
    b: Optional[float] = None

    def key(self) -> str:
        if self.kind == "range":
            return f"range:{_fmt_num(self.a)}-{_fmt_num(self.b)}"
        if self.kind == "le":
            return f"le:{_fmt_num(self.a)}"
        if self.kind == "ge":
            return f"ge:{_fmt_num(self.a)}"
        return f"raw:{self.a}"


def _fmt_num(value: Optional[float]) -> str:
    if value is None:
        return "na"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _parse_bucket_spec(text: str) -> Optional[BucketSpec]:
    s = text.strip().lower()
    s = s.replace("\u00ba", "").replace("\u00b0", "")
    s = s.replace("deg", "")
    s = s.replace("degrees", "").replace("degree", "")
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"(\d)\s*-\s*(\d)", r"\1 to \2", s)
    nums = [float(n) for n in _NUM_RE.findall(s)]
    if ("or below" in s or "below" in s) and nums:
        return BucketSpec(kind="le", a=nums[0])
    if ("or above" in s or "above" in s) and nums:
        return BucketSpec(kind="ge", a=nums[0])
    if len(nums) >= 2:
        if "to" in s or "-" in s:
            return BucketSpec(kind="range", a=nums[0], b=nums[1])
        return BucketSpec(kind="range", a=nums[0], b=nums[1])
    return None


def _bucket_key_map(columns: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for col in columns:
        spec = _parse_bucket_spec(col)
        if spec:
            out.setdefault(spec.key(), col)
    return out


def _normalize_label(label: str) -> str:
    s = label.strip().lower()
    s = s.replace("\u00ba", "").replace("\u00b0", "")
    s = s.replace("deg", "")
    s = re.sub(r"\s+", " ", s)
    return s


def _resolve_bucket_label(bucket_arg: str, columns: List[str]) -> str:
    if bucket_arg in columns:
        return bucket_arg
    for col in columns:
        if col.lower() == bucket_arg.lower():
            return col

    bucket_spec = _parse_bucket_spec(bucket_arg)
    if bucket_spec:
        key_map = _bucket_key_map(columns)
        match = key_map.get(bucket_spec.key())
        if match:
            return match

    normalized_arg = _normalize_label(bucket_arg)
    for col in columns:
        if _normalize_label(col) == normalized_arg:
            return col

    # Fallback: allow substring match if it is unambiguous.
    contains = [c for c in columns if normalized_arg in _normalize_label(c)]
    if len(contains) == 1:
        return contains[0]

    raise ValueError(
        "Bucket not found. "
        f"Requested={bucket_arg!r}. "
        f"Available buckets: {', '.join(columns)}"
    )


def _timestamp_tag_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("_")


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing 'timestamp' column in {path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df


def _plot_series(
    df: pd.DataFrame,
    bucket_label: str,
    mode: str,
    out_path: Path,
    title: str,
    show_markers: bool,
    side: str,
) -> None:
    series = pd.to_numeric(df[bucket_label], errors="coerce")
    if mode == "state":
        series = series.ffill()
        if side == "no":
            series = 100.0 - series
        plot_df = pd.DataFrame({"timestamp": df["timestamp"], "price": series})
    else:
        if side == "no":
            series = 100.0 - series
        plot_df = pd.DataFrame({"timestamp": df["timestamp"], "price": series}).dropna()

    if plot_df.empty:
        raise ValueError(f"No data points found for bucket {bucket_label!r} in mode={mode}")

    fig = plt.figure(figsize=(12, 5.5), dpi=160)
    ax = fig.add_subplot(1, 1, 1)
    x = plot_df["timestamp"].dt.tz_convert(None)
    y = plot_df["price"]

    if mode == "updates":
        ax.plot(x, y, lw=1.6, color="#1f77b4")
        if show_markers:
            ax.scatter(x, y, s=10, color="#1f77b4", alpha=0.8)
    else:
        ax.plot(x, y, lw=1.8, color="#1f77b4")

    ax.set_title(title)
    ax.set_xlabel("UTC Time")
    ax.set_ylabel(f"{side.upper()} price (cents)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plot Kalshi bucket price evolution from a daily CSV.")
    parser.add_argument("--file", required=True, help="CSV filename or path (e.g., KMIA_20251024.csv)")
    parser.add_argument("--bucket", required=True, help="Bucket label or shorthand (e.g., '88-89')")
    parser.add_argument(
        "--kalshi-dir",
        default=str(DEFAULT_KALSHI_DIR),
        help="Base directory for Kalshi CSVs (used if --file is not a path)",
    )
    parser.add_argument(
        "--mode",
        choices=["state", "updates"],
        default="state",
        help="state = forward-filled series; updates = only the sparse updates",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for plot image",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional tag added to output filename (default: UTC timestamp)",
    )
    parser.add_argument(
        "--show-markers",
        action="store_true",
        help="Add markers for update points (useful for sparse updates mode).",
    )
    parser.add_argument(
        "--side",
        choices=["yes", "no"],
        default="yes",
        help="Plot YES or NO side prices (NO is computed as 100 - YES).",
    )
    args = parser.parse_args(argv)

    file_path = Path(args.file)
    if not file_path.exists():
        candidate = Path(args.kalshi_dir) / args.file
        if candidate.exists():
            file_path = candidate
        else:
            raise FileNotFoundError(f"CSV file not found: {file_path} (also tried {candidate})")

    df = _load_csv(file_path)
    bucket_columns = [c for c in df.columns if c != "timestamp"]
    if not bucket_columns:
        raise ValueError(f"No bucket columns found in {file_path}")

    bucket_label = _resolve_bucket_label(args.bucket, bucket_columns)
    mode = args.mode
    tag = _sanitize_filename(args.tag) if args.tag else _timestamp_tag_utc()
    bucket_slug = _sanitize_filename(bucket_label)
    out_name = f"{file_path.stem}__{bucket_slug}__{mode}__{args.side}__{tag}.png"
    out_path = Path(args.out_dir).resolve() / out_name

    title = f"{file_path.name} | {bucket_label} | {mode} | {args.side.upper()}"
    _plot_series(df, bucket_label, mode, out_path, title, show_markers=args.show_markers, side=args.side)

    print(f"Wrote plot: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
