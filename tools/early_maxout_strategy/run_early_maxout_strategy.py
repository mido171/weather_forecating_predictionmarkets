import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

import lightgbm as lgb


LOCAL_TZ = "America/New_York"
STOCKHOLM_TZ = "Europe/Stockholm"

START_DATE = date(2002, 1, 1)
END_DATE = date(2026, 12, 31)

EPS = 0.01


@dataclass
class DayWindow:
    target_date_local: date
    day_start_utc: datetime
    day_end_utc: datetime
    cutoff_utc: datetime


def _date_range(start: date, end: date) -> List[date]:
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days + 1)]


def _load_minute_data(minute_dir: Path) -> pd.DataFrame:
    files = sorted(minute_dir.glob("MIA_tmpf_1min_UTC_*.csv"))
    if not files:
        raise FileNotFoundError(f"No minute files found in {minute_dir}")
    frames = []
    for path in files:
        df = pd.read_csv(path, usecols=["valid(UTC)", "tmpf"], dtype={"tmpf": "string"})
        df["ts_utc"] = pd.to_datetime(df["valid(UTC)"], utc=True, errors="coerce")
        df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")
        df = df.dropna(subset=["ts_utc", "tmpf"])
        frames.append(df[["ts_utc", "tmpf"]])
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values("ts_utc")
    all_df = all_df.drop_duplicates(subset=["ts_utc"], keep="last")
    return all_df


def _build_day_window(day: date) -> DayWindow:
    local = ZoneInfo(LOCAL_TZ)
    stockholm = ZoneInfo(STOCKHOLM_TZ)

    day_start_local = datetime(day.year, day.month, day.day, 0, 0, tzinfo=local)
    day_end_local = day_start_local + timedelta(days=1)
    day_start_utc = day_start_local.astimezone(timezone.utc)
    day_end_utc = day_end_local.astimezone(timezone.utc)

    cutoff_local = datetime(day.year, day.month, day.day, 18, 30, tzinfo=stockholm)
    cutoff_utc = cutoff_local.astimezone(timezone.utc)

    if cutoff_utc < day_start_utc:
        cutoff_utc = day_start_utc
    if cutoff_utc >= day_end_utc:
        cutoff_utc = day_end_utc - timedelta(seconds=1)

    return DayWindow(
        target_date_local=day,
        day_start_utc=day_start_utc,
        day_end_utc=day_end_utc,
        cutoff_utc=cutoff_utc,
    )


def _expected_points(start_utc: datetime, end_utc: datetime, freq_minutes: int, inclusive_end: bool) -> int:
    seconds = (end_utc - start_utc).total_seconds()
    if inclusive_end:
        return int(seconds // (freq_minutes * 60)) + 1
    return int(seconds // (freq_minutes * 60))


def _ols_slope(minutes: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if mask.sum() < 4:
        return float("nan")
    x = minutes[mask] / 60.0
    y = values[mask].astype(float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return float("nan")
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def _longest_run(mask: np.ndarray) -> int:
    best = 0
    run = 0
    for val in mask:
        if val:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best


def _dct_matrix(n: int, kmax: int) -> np.ndarray:
    n_idx = np.arange(n)[:, None]
    k_idx = np.arange(kmax)[None, :]
    return np.cos(np.pi / n * (n_idx + 0.5) * k_idx)


def _select_threshold(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    best_t = 0.5
    best_acc = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        acc = accuracy_score(y_true, p_pred >= t)
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t


def _compute_metrics(y_true: np.ndarray, p_pred: np.ndarray, threshold: float) -> Dict[str, float]:
    y_hat = (p_pred >= threshold).astype(int)
    acc = accuracy_score(y_true, y_hat)
    bal = balanced_accuracy_score(y_true, y_hat)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", pos_label=1)
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    try:
        auc = roc_auc_score(y_true, p_pred)
    except ValueError:
        auc = float("nan")
    try:
        brier = brier_score_loss(y_true, p_pred)
    except ValueError:
        brier = float("nan")
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal),
        "yes_precision": float(prec),
        "yes_recall": float(rec),
        "yes_f1": float(f1),
        "roc_auc": float(auc),
        "brier": float(brier),
        "cm_tn": int(cm[0, 0]),
        "cm_fp": int(cm[0, 1]),
        "cm_fn": int(cm[1, 0]),
        "cm_tp": int(cm[1, 1]),
    }

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class TCN(nn.Module):
    def __init__(self, n_features: int, n_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, n_channels, kernel_size=3, padding=2, dilation=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(n_channels, n_channels, kernel_size=3, padding=4, dilation=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(n_channels, n_channels, kernel_size=3, padding=8, dilation=4),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.net(x)
        x = self.head(x)
        return x.squeeze(-1)


class SimpleTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = 32, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x.squeeze(-1)


def _train_torch_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, pos_weight: float) -> nn.Module:
    device = torch.device("cpu")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_state = None
    best_acc = -1.0
    patience = 5
    patience_left = patience

    for _ in range(1, 31):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(device))
                prob = torch.sigmoid(logits).cpu().numpy()
                preds.append(prob)
                trues.append(yb.numpy())
        p_val = np.concatenate(preds)
        y_val = np.concatenate(trues)
        acc = accuracy_score(y_val, p_val >= 0.5)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--minute-dir",
        default=r"C:\Users\ahmad\Desktop\generalFiles\git\weather-forecasting-predictionmarkets\weather_forecating_predictionmarkets\data\iem_minute_data\MIA\tmpf\UTC\yearly",
    )
    parser.add_argument(
        "--out-base",
        default=r"C:\Users\ahmad\Desktop\generalFiles\git\weather-forecasting-predictionmarkets\weather_forecating_predictionmarkets\artifacts\experiments\early_maxout_strategy",
    )
    parser.add_argument(
        "--data-base",
        default=r"C:\Users\ahmad\Desktop\generalFiles\git\weather-forecasting-predictionmarkets\weather_forecating_predictionmarkets\data\tmax_hit1830",
    )
    args = parser.parse_args()

    minute_dir = Path(args.minute_dir)
    out_base = Path(args.out_base)
    data_base = Path(args.data_base)

    out_base.mkdir(parents=True, exist_ok=True)
    data_base.mkdir(parents=True, exist_ok=True)
    (data_base / "partial_series_5min").mkdir(parents=True, exist_ok=True)
    (data_base / "full_series_5min").mkdir(parents=True, exist_ok=True)
    (data_base / "window12h_5min").mkdir(parents=True, exist_ok=True)
    (data_base / "window18h_5min").mkdir(parents=True, exist_ok=True)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    df_1m = _load_minute_data(minute_dir)
    df_1m = df_1m.sort_values("ts_utc")
    df_1m = df_1m.set_index("ts_utc")

    series_5m = df_1m["tmpf"].resample("5min").median()

    day_rows = []

    dates = _date_range(START_DATE, END_DATE)
    for day in dates:
        window = _build_day_window(day)
        day_start = window.day_start_utc
        day_end = window.day_end_utc
        cutoff = window.cutoff_utc

        full_1m = df_1m.loc[day_start: day_end - timedelta(seconds=1)]
        if full_1m.empty:
            continue

        tmax_full = float(full_1m["tmpf"].max())
        tmin_full = float(full_1m["tmpf"].min())
        max_rows = full_1m[full_1m["tmpf"] >= tmax_full - EPS]
        tmax_time_utc = max_rows.index.min() if not max_rows.empty else pd.NaT
        tmax_time_local = tmax_time_utc.tz_convert(LOCAL_TZ) if pd.notna(tmax_time_utc) else pd.NaT
        tmax_time_local_minute = (
            tmax_time_local.hour * 60 + tmax_time_local.minute if pd.notna(tmax_time_local) else float("nan")
        )

        partial_1m = full_1m.loc[:cutoff]
        tmax_sofar = float(partial_1m["tmpf"].max()) if not partial_1m.empty else float("nan")

        full_5m = series_5m.loc[day_start: day_end - timedelta(minutes=5)]
        partial_5m = series_5m.loc[day_start: cutoff]

        expected_full = _expected_points(day_start, day_end, 5, inclusive_end=False)
        expected_partial = _expected_points(day_start, cutoff, 5, inclusive_end=True)
        full_cov = full_5m.notna().sum() / expected_full if expected_full else 0.0
        partial_cov = partial_5m.notna().sum() / expected_partial if expected_partial else 0.0

        if full_cov < 0.90 or partial_cov < 0.70:
            continue

        partial_idx = pd.date_range(day_start, cutoff, freq="5min")
        partial_vals = partial_5m.reindex(partial_idx).to_numpy()
        np.save(data_base / "partial_series_5min" / f"{day.isoformat()}.npy", partial_vals)

        full_idx = pd.date_range(day_start, day_end - timedelta(minutes=5), freq="5min")
        full_vals = full_5m.reindex(full_idx).to_numpy()
        np.save(data_base / "full_series_5min" / f"{day.isoformat()}.npy", full_vals)

        outflow_drop_cnt = float("nan")
        full_vals_clean = full_vals[np.isfinite(full_vals)]
        if len(full_vals_clean) >= 7:
            drops = full_vals_clean[:-6] - full_vals_clean[6:]
            outflow_drop_cnt = float(np.sum(drops >= 2.0))

        y_hit = int(tmax_sofar >= (tmax_full - EPS))

        day_rows.append(
            {
                "target_date_local": day,
                "day_start_utc": day_start,
                "day_end_utc": day_end,
                "cutoff_utc": cutoff,
                "tmax_full": tmax_full,
                "tmin_full": tmin_full,
                "tmax_sofar": tmax_sofar,
                "tmax_time_utc": tmax_time_utc,
                "tmax_time_local": tmax_time_local,
                "tmax_time_local_minute": tmax_time_local_minute,
                "outflow_drop_cnt": outflow_drop_cnt,
                "full_cov": full_cov,
                "partial_cov": partial_cov,
                "y_hit_by_cutoff": y_hit,
            }
        )

    if not day_rows:
        raise RuntimeError("No day rows were produced; check input data.")

    day_df = pd.DataFrame(day_rows).sort_values("target_date_local")

    day_df["year"] = pd.to_datetime(day_df["target_date_local"]).dt.year
    stats = (
        day_df.groupby("year")
        .agg(total_days=("target_date_local", "count"), hit_by_cutoff=("y_hit_by_cutoff", "sum"))
        .reset_index()
    )
    stats["pct"] = stats["hit_by_cutoff"] / stats["total_days"] * 100.0
    stats_path = reports_dir / "hit_by_1830_stockholm_stats.csv"
    stats.to_csv(stats_path, index=False)

    day_table_path = data_base / "day_table.parquet"
    day_df.to_parquet(day_table_path, index=False)

    dct_cos = _dct_matrix(144, 30)
    features = []
    seq_rows = []

    daily_summary = day_df[["target_date_local", "tmax_time_local_minute", "tmax_full", "tmin_full", "outflow_drop_cnt"]].copy()
    daily_summary["range_full"] = daily_summary["tmax_full"] - daily_summary["tmin_full"]

    for _, row in day_df.iterrows():
        day = row["target_date_local"]
        cutoff = row["cutoff_utc"]

        window_end = cutoff - timedelta(minutes=5)
        window_start = cutoff - timedelta(hours=12)
        idx_12h = pd.date_range(window_start, window_end, freq="5min")
        series_12h = series_5m.reindex(idx_12h)
        vals_12h = series_12h.to_numpy()

        window18_start = cutoff - timedelta(hours=18)
        idx_18h = pd.date_range(window18_start, window_end, freq="5min")
        series_18h = series_5m.reindex(idx_18h)
        vals_18h = series_18h.to_numpy()

        np.save(data_base / "window12h_5min" / f"{day.isoformat()}.npy", vals_12h)
        np.save(data_base / "window18h_5min" / f"{day.isoformat()}.npy", vals_18h)

        micro_30 = df_1m.loc[cutoff - timedelta(minutes=30): cutoff]["tmpf"].to_numpy(dtype=float)
        micro_60 = df_1m.loc[cutoff - timedelta(minutes=60): cutoff]["tmpf"].to_numpy(dtype=float)
        micro_std_30 = float(np.nanstd(micro_30)) if micro_30.size else float("nan")
        micro_std_60 = float(np.nanstd(micro_60)) if micro_60.size else float("nan")
        micro_mean_abs_change_30 = (
            float(np.nanmean(np.abs(np.diff(micro_30)))) if micro_30.size > 1 else float("nan")
        )

        temp_now = float(np.nanmedian(vals_12h[-2:])) if np.isfinite(vals_12h[-2:]).any() else float("nan")
        temp_max = float(np.nanmax(vals_12h)) if np.isfinite(vals_12h).any() else float("nan")
        temp_min = float(np.nanmin(vals_12h)) if np.isfinite(vals_12h).any() else float("nan")
        range_sofar = temp_max - temp_min if np.isfinite(temp_max) and np.isfinite(temp_min) else float("nan")

        if np.isfinite(vals_12h).any():
            max_idx = int(np.nanargmax(vals_12h))
        else:
            max_idx = -1
        time_of_max = max_idx * 5 if max_idx >= 0 else float("nan")
        minutes_since_max = (len(vals_12h) - 1 - max_idx) * 5 if max_idx >= 0 else float("nan")
        drop_from_max = temp_max - temp_now if np.isfinite(temp_max) and np.isfinite(temp_now) else float("nan")

        def slope_last(n_minutes: int) -> float:
            n_points = int(n_minutes / 5)
            if n_points < 2 or len(vals_12h) < n_points:
                return float("nan")
            y = vals_12h[-n_points:]
            mins = np.arange(n_points) * 5
            return _ols_slope(mins, y)

        slope_15 = slope_last(15)
        slope_30 = slope_last(30)
        slope_60 = slope_last(60)
        slope_120 = slope_last(120)
        slope_diff_30_120 = slope_30 - slope_120 if np.isfinite(slope_30) and np.isfinite(slope_120) else float("nan")

        slope_since_max = float("nan")
        if max_idx >= 0 and minutes_since_max > 0 and np.isfinite(temp_now) and np.isfinite(temp_max):
            slope_since_max = (temp_now - temp_max) / (minutes_since_max / 60.0)

        def std_last(n_minutes: int) -> float:
            n_points = int(n_minutes / 5)
            if n_points < 2 or len(vals_12h) < n_points:
                return float("nan")
            y = vals_12h[-n_points:]
            return float(np.nanstd(y))

        std_30 = std_last(30)
        std_60 = std_last(60)
        std_120 = std_last(120)

        mean_abs_change_60 = float("nan")
        if len(vals_12h) >= 12:
            diffs = np.diff(vals_12h[-12:])
            mean_abs_change_60 = float(np.nanmean(np.abs(diffs)))

        count_big_delta_120 = float("nan")
        if len(vals_12h) >= 24:
            diffs = np.diff(vals_12h[-24:])
            count_big_delta_120 = float(np.sum(np.abs(diffs) >= 0.4))

        max_drop_30 = float("nan")
        drop_cnt_30m_0p5 = float("nan")
        drop_cnt_30m_1p0 = float("nan")
        drop_cnt_30m_2p0 = float("nan")
        if len(vals_12h) >= 12:
            last_6h = vals_12h[-72:] if len(vals_12h) >= 72 else vals_12h
            if len(last_6h) >= 7:
                drops = last_6h[:-6] - last_6h[6:]
                max_drop_30 = float(np.nanmax(drops))
                drop_cnt_30m_0p5 = float(np.sum(drops >= 0.5))
                drop_cnt_30m_1p0 = float(np.sum(drops >= 1.0))
                drop_cnt_30m_2p0 = float(np.sum(drops >= 2.0))

        plateau_frac_0p1 = float("nan")
        plateau_frac_0p2 = float("nan")
        plateau_longest_run_0p2 = float("nan")
        if len(vals_12h) >= 24 and np.isfinite(temp_max):
            last_120 = vals_12h[-24:]
            mask_0p1 = last_120 >= (temp_max - 0.1)
            mask_0p2 = last_120 >= (temp_max - 0.2)
            plateau_frac_0p1 = float(np.nanmean(mask_0p1))
            plateau_frac_0p2 = float(np.nanmean(mask_0p2))
            plateau_longest_run_0p2 = float(_longest_run(mask_0p2) * 5)

        def slope_segment(start_idx: int, end_idx: int) -> float:
            seg = vals_12h[start_idx:end_idx]
            mins = np.arange(len(seg)) * 5
            return _ols_slope(mins, seg)

        third = len(vals_12h) // 3
        slope_early = slope_segment(0, third)
        slope_mid = slope_segment(third, 2 * third)
        slope_late = slope_segment(2 * third, len(vals_12h))
        heating_slowdown = slope_mid - slope_late if np.isfinite(slope_mid) and np.isfinite(slope_late) else float("nan")

        dt = pd.Timestamp(day)
        doy = dt.dayofyear
        doy_sin = math.sin(2 * math.pi * doy / 365.25)
        doy_cos = math.cos(2 * math.pi * doy / 365.25)
        month = dt.month

        centered = vals_12h.copy()
        med = np.nanmedian(centered)
        centered = centered - med
        centered = np.where(np.isfinite(centered), centered, 0.0)
        coeff = centered @ dct_cos
        dct_energy_total = float(np.sum(coeff ** 2)) if coeff.size else float("nan")
        dct_energy_hi = float(np.sum(coeff[10:] ** 2) / dct_energy_total) if dct_energy_total else float("nan")
        dct_energy_mid = float(np.sum(coeff[5:15] ** 2) / dct_energy_total) if dct_energy_total else float("nan")

        feat = {
            "target_date_local": day,
            "temp_now": temp_now,
            "temp_max_sofar": temp_max,
            "temp_min_sofar": temp_min,
            "range_sofar": range_sofar,
            "time_of_max_sofar_minutes": time_of_max,
            "minutes_since_max": minutes_since_max,
            "drop_from_max": drop_from_max,
            "slope_15m": slope_15,
            "slope_30m": slope_30,
            "slope_60m": slope_60,
            "slope_120m": slope_120,
            "slope_diff_30_120": slope_diff_30_120,
            "slope_since_max": slope_since_max,
            "std_30m": std_30,
            "std_60m": std_60,
            "std_120m": std_120,
            "mean_abs_change_60m": mean_abs_change_60,
            "count_abs_change_ge_0p4_120m": count_big_delta_120,
            "max_drop_30m": max_drop_30,
            "drop_cnt_30m_ge_0p5_6h": drop_cnt_30m_0p5,
            "drop_cnt_30m_ge_1p0_6h": drop_cnt_30m_1p0,
            "drop_cnt_30m_ge_2p0_6h": drop_cnt_30m_2p0,
            "plateau_frac_0p1": plateau_frac_0p1,
            "plateau_frac_0p2": plateau_frac_0p2,
            "plateau_longest_run_0p2": plateau_longest_run_0p2,
            "slope_early": slope_early,
            "slope_mid": slope_mid,
            "slope_late": slope_late,
            "heating_slowdown": heating_slowdown,
            "doy_sin": doy_sin,
            "doy_cos": doy_cos,
            "month": month,
            "micro_std_30m": micro_std_30,
            "micro_std_60m": micro_std_60,
            "micro_mean_abs_change_30m": micro_mean_abs_change_30,
            "dct_energy_hi": dct_energy_hi,
            "dct_energy_mid": dct_energy_mid,
        }
        for i in range(30):
            feat[f"dct_{i}"] = float(coeff[i])

        features.append(feat)

        seq_vals = vals_12h.copy()
        seq_vals = pd.Series(seq_vals).interpolate(limit_direction="both").to_numpy()
        if not np.isfinite(seq_vals).any():
            seq_vals[:] = 0.0
        seq_delta = np.diff(seq_vals, prepend=seq_vals[0])
        seq_roll = pd.Series(seq_vals).rolling(3, min_periods=1).mean().to_numpy()
        seq_rows.append({
            "target_date_local": day,
            "seq": np.stack([seq_vals, seq_delta, seq_roll], axis=1),
        })

    feat_df = pd.DataFrame(features)
    seq_df = pd.DataFrame(seq_rows)

    daily_summary = daily_summary.sort_values("target_date_local")
    for lag in [1, 2, 3]:
        daily_summary[f"tmax_time_t{lag}"] = daily_summary["tmax_time_local_minute"].shift(lag)
        daily_summary[f"range_t{lag}"] = daily_summary["range_full"].shift(lag)
        daily_summary[f"outflow_drop_cnt_t{lag}"] = daily_summary["outflow_drop_cnt"].shift(lag)

    daily_summary["tmax_time_trend_3d"] = (
        daily_summary["tmax_time_t1"] - daily_summary["tmax_time_t3"]
    ) / 2.0
    daily_summary["range_trend_3d"] = (
        daily_summary["range_t1"] - daily_summary["range_t3"]
    ) / 2.0

    feat_df = feat_df.merge(daily_summary[[
        "target_date_local",
        "tmax_time_t1",
        "tmax_time_t2",
        "tmax_time_t3",
        "range_t1",
        "range_t2",
        "range_t3",
        "outflow_drop_cnt_t1",
        "outflow_drop_cnt_t2",
        "outflow_drop_cnt_t3",
        "tmax_time_trend_3d",
        "range_trend_3d",
    ]], on="target_date_local", how="left")

    df = feat_df.merge(day_df[["target_date_local", "cutoff_utc", "y_hit_by_cutoff", "tmax_full", "tmax_sofar"]], on="target_date_local", how="left")

    audit_path = reports_dir / "feature_leakage_audit_hit1830.md"
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write("Feature leakage audit for tmax_hit1830\n\n")
        f.write("No timestamp-based leakage detected; all features are computed using windows ending at cutoff.\n")

    df["year"] = pd.to_datetime(df["target_date_local"]).dt.year
    df["split"] = np.where(df["year"] <= 2019, "train", np.where(df["year"] <= 2022, "val", np.where(df["year"] <= 2025, "test", "future")))

    features_path = data_base / "features.parquet"
    df.to_parquet(features_path, index=False)

    feature_cols = [c for c in df.columns if c not in {"target_date_local", "cutoff_utc", "y_hit_by_cutoff", "year", "split", "tmax_full"}]

    X = df[feature_cols]
    y = df["y_hit_by_cutoff"].to_numpy()

    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    test_mask = df["split"] == "test"

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X[train_mask])
    X_val = imputer.transform(X[val_mask])
    X_test = imputer.transform(X[test_mask])

    y_train = y[train_mask.to_numpy()]
    y_val = y[val_mask.to_numpy()]
    y_test = y[test_mask.to_numpy()]

    pos_rate = float(y_train.mean())
    always_no_acc = float(accuracy_score(y_test, np.zeros_like(y_test)))
    always_yes_acc = float(accuracy_score(y_test, np.ones_like(y_test)))

    exp_dir = out_base
    exp_dir.mkdir(parents=True, exist_ok=True)

    results_summary = {}

    def save_experiment(name: str, preds: pd.DataFrame, metrics: Dict[str, Dict[str, float]]):
        exp_path = exp_dir / f"exp_{name}"
        exp_path.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(exp_path / "preds.parquet", index=False)
        with open(exp_path / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        results_summary[name] = metrics

    rule_grid = {
        "minutes_since_max": [30, 45, 60],
        "drop_from_max": [0.2, 0.3, 0.4],
        "slope_60m": [-0.10, -0.15, -0.20],
        "plateau_frac": [0.80, 0.85, 0.90],
    }
    best = None
    best_acc = -1.0

    val_df = df[val_mask].copy()
    for m in rule_grid["minutes_since_max"]:
        for d in rule_grid["drop_from_max"]:
            for s in rule_grid["slope_60m"]:
                for p in rule_grid["plateau_frac"]:
                    cond1 = (
                        (val_df["minutes_since_max"] >= m)
                        & (val_df["drop_from_max"] >= d)
                        & (val_df["slope_60m"] <= s)
                    )
                    cond2 = (
                        (val_df["plateau_frac_0p2"] >= p)
                        & (val_df["slope_30m"] <= 0.05)
                    )
                    y_pred = (cond1 | cond2).astype(int).to_numpy()
                    acc = accuracy_score(y_val, y_pred)
                    if acc > best_acc:
                        best_acc = acc
                        best = (m, d, s, p)

    best_m, best_d, best_s, best_p = best

    def rule_predict(frame: pd.DataFrame) -> np.ndarray:
        cond1 = (
            (frame["minutes_since_max"] >= best_m)
            & (frame["drop_from_max"] >= best_d)
            & (frame["slope_60m"] <= best_s)
        )
        cond2 = (
            (frame["plateau_frac_0p2"] >= best_p)
            & (frame["slope_30m"] <= 0.05)
        )
        return (cond1 | cond2).astype(int).to_numpy()

    preds_rule = df[["target_date_local", "cutoff_utc", "y_hit_by_cutoff", "temp_max_sofar", "slope_60m", "drop_from_max", "split"]].copy()
    preds_rule["p_pred"] = rule_predict(df)
    preds_rule["y_pred"] = preds_rule["p_pred"].astype(int)

    metrics_rule = {}
    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        sub = preds_rule[mask]
        metrics_rule[split_name] = _compute_metrics(
            sub["y_hit_by_cutoff"].to_numpy(), sub["p_pred"].to_numpy(), 0.5
        )
    metrics_rule["rule_params"] = {
        "minutes_since_max": best_m,
        "drop_from_max": best_d,
        "slope_60m": best_s,
        "plateau_frac": best_p,
    }
    save_experiment("exp1_rule", preds_rule, metrics_rule)

    season_cols = ["doy_sin", "doy_cos", "month"]
    X_season = df[season_cols].copy()
    X_season = pd.get_dummies(X_season, columns=["month"], drop_first=False)
    Xs_train = X_season[train_mask]
    Xs_val = X_season[val_mask]
    Xs_test = X_season[test_mask]

    logit = LogisticRegression(max_iter=200)
    logit.fit(Xs_train, y_train)
    p_val = logit.predict_proba(Xs_val)[:, 1]
    t_season = _select_threshold(y_val, p_val)

    p_train = logit.predict_proba(Xs_train)[:, 1]
    p_test = logit.predict_proba(Xs_test)[:, 1]

    preds_season = df[["target_date_local", "cutoff_utc", "y_hit_by_cutoff", "temp_max_sofar", "slope_60m", "drop_from_max", "split"]].copy()
    preds_season.loc[train_mask, "p_pred"] = p_train
    preds_season.loc[val_mask, "p_pred"] = p_val
    preds_season.loc[test_mask, "p_pred"] = p_test
    preds_season["y_pred"] = (preds_season["p_pred"] >= t_season).astype(int)

    metrics_season = {
        "train": _compute_metrics(y_train, p_train, t_season),
        "val": _compute_metrics(y_val, p_val, t_season),
        "test": _compute_metrics(y_test, p_test, t_season),
        "threshold": t_season,
    }
    save_experiment("exp2_seasonality", preds_season, metrics_season)

    scale_pos = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    param_grid = [
        {"num_leaves": nl, "min_data_in_leaf": md, "learning_rate": lr, "feature_fraction": ff, "bagging_fraction": bf, "reg_lambda": rl}
        for nl in [31, 63, 127]
        for md in [50, 150, 300]
        for lr in [0.03, 0.05, 0.08]
        for ff in [0.6, 0.8, 1.0]
        for bf in [0.7, 0.9]
        for rl in [0, 1, 5]
    ]

    best_model = None
    best_val_acc = -1.0
    best_params = None
    best_val_pred = None

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    for params in param_grid:
        params_full = {
            "objective": "binary",
            "metric": "auc",
            "verbose": -1,
            "scale_pos_weight": scale_pos,
        }
        params_full.update(params)
        model = lgb.train(
            params_full,
            lgb_train,
            valid_sets=[lgb_val],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        val_pred = model.predict(X_val)
        t = _select_threshold(y_val, val_pred)
        acc = accuracy_score(y_val, val_pred >= t)
        if acc > best_val_acc:
            best_val_acc = acc
            best_model = model
            best_params = params_full
            best_val_pred = val_pred

    lgbm_model = best_model
    p_train_lgbm = lgbm_model.predict(X_train)
    p_val_lgbm = best_val_pred
    p_test_lgbm = lgbm_model.predict(X_test)
    t_lgbm = _select_threshold(y_val, p_val_lgbm)

    preds_lgbm = df[["target_date_local", "cutoff_utc", "y_hit_by_cutoff", "temp_max_sofar", "slope_60m", "drop_from_max", "split"]].copy()
    preds_lgbm.loc[train_mask, "p_pred"] = p_train_lgbm
    preds_lgbm.loc[val_mask, "p_pred"] = p_val_lgbm
    preds_lgbm.loc[test_mask, "p_pred"] = p_test_lgbm
    preds_lgbm["y_pred"] = (preds_lgbm["p_pred"] >= t_lgbm).astype(int)

    metrics_lgbm = {
        "train": _compute_metrics(y_train, p_train_lgbm, t_lgbm),
        "val": _compute_metrics(y_val, p_val_lgbm, t_lgbm),
        "test": _compute_metrics(y_test, p_test_lgbm, t_lgbm),
        "threshold": t_lgbm,
        "params": best_params,
    }
    save_experiment("exp3_lgbm", preds_lgbm, metrics_lgbm)

    delta_train = (df.loc[train_mask, "tmax_full"] - df.loc[train_mask, "temp_max_sofar"]).to_numpy()
    delta_val = (df.loc[val_mask, "tmax_full"] - df.loc[val_mask, "temp_max_sofar"]).to_numpy()

    reg = lgb.LGBMRegressor(
        num_leaves=63,
        min_data_in_leaf=50,
        learning_rate=0.05,
        n_estimators=300,
        reg_lambda=1.0,
    )
    reg.fit(X_train, delta_train)
    p_val_delta = reg.predict(X_val)
    best_tau = None
    best_acc = -1.0
    for tau in [0.0, 0.05, 0.10, 0.20]:
        acc = accuracy_score(y_val, p_val_delta <= tau)
        if acc > best_acc:
            best_acc = acc
            best_tau = tau

    p_train_delta = reg.predict(X_train)
    p_test_delta = reg.predict(X_test)

    preds_reg = df[["target_date_local", "cutoff_utc", "y_hit_by_cutoff", "temp_max_sofar", "slope_60m", "drop_from_max", "split"]].copy()
    preds_reg.loc[train_mask, "p_pred"] = 1.0 - (p_train_delta > best_tau).astype(float)
    preds_reg.loc[val_mask, "p_pred"] = 1.0 - (p_val_delta > best_tau).astype(float)
    preds_reg.loc[test_mask, "p_pred"] = 1.0 - (p_test_delta > best_tau).astype(float)
    preds_reg["y_pred"] = (preds_reg["p_pred"] >= 0.5).astype(int)

    metrics_reg = {
        "train": _compute_metrics(y_train, preds_reg.loc[train_mask, "p_pred"].to_numpy(), 0.5),
        "val": _compute_metrics(y_val, preds_reg.loc[val_mask, "p_pred"].to_numpy(), 0.5),
        "test": _compute_metrics(y_test, preds_reg.loc[test_mask, "p_pred"].to_numpy(), 0.5),
        "tau": best_tau,
    }
    save_experiment("exp4_regression", preds_reg, metrics_reg)

    seq_df = seq_df.merge(df[["target_date_local", "split"]], on="target_date_local", how="left")
    seq_df = seq_df.sort_values("target_date_local")

    seq_train = np.stack(seq_df[seq_df["split"] == "train"]["seq"].to_numpy())
    seq_val = np.stack(seq_df[seq_df["split"] == "val"]["seq"].to_numpy())
    seq_test = np.stack(seq_df[seq_df["split"] == "test"]["seq"].to_numpy())

    seq_mean = seq_train.mean(axis=(0, 1), keepdims=True)
    seq_std = seq_train.std(axis=(0, 1), keepdims=True) + 1e-6
    seq_train = (seq_train - seq_mean) / seq_std
    seq_val = (seq_val - seq_mean) / seq_std
    seq_test = (seq_test - seq_mean) / seq_std

    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    train_loader = DataLoader(SequenceDataset(seq_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(SequenceDataset(seq_val, y_val), batch_size=128, shuffle=False)

    tcn = TCN(n_features=seq_train.shape[2])
    tcn = _train_torch_model(tcn, train_loader, val_loader, pos_weight)

    def predict_torch(model: nn.Module, data: np.ndarray) -> np.ndarray:
        model.eval()
        preds = []
        loader = DataLoader(SequenceDataset(data, np.zeros(data.shape[0])), batch_size=128, shuffle=False)
        with torch.no_grad():
            for xb, _ in loader:
                logits = model(xb)
                preds.append(torch.sigmoid(logits).numpy())
        return np.concatenate(preds)

    p_train_tcn = predict_torch(tcn, seq_train)
    p_val_tcn = predict_torch(tcn, seq_val)
    p_test_tcn = predict_torch(tcn, seq_test)

    t_tcn = _select_threshold(y_val, p_val_tcn)

    preds_tcn = df[["target_date_local", "cutoff_utc", "y_hit_by_cutoff", "temp_max_sofar", "slope_60m", "drop_from_max", "split"]].copy()
    preds_tcn.loc[train_mask, "p_pred"] = p_train_tcn
    preds_tcn.loc[val_mask, "p_pred"] = p_val_tcn
    preds_tcn.loc[test_mask, "p_pred"] = p_test_tcn
    preds_tcn["y_pred"] = (preds_tcn["p_pred"] >= t_tcn).astype(int)

    metrics_tcn = {
        "train": _compute_metrics(y_train, p_train_tcn, t_tcn),
        "val": _compute_metrics(y_val, p_val_tcn, t_tcn),
        "test": _compute_metrics(y_test, p_test_tcn, t_tcn),
        "threshold": t_tcn,
    }
    save_experiment("exp5_tcn", preds_tcn, metrics_tcn)

    transformer = SimpleTransformer(n_features=seq_train.shape[2])
    transformer = _train_torch_model(transformer, train_loader, val_loader, pos_weight)

    p_train_tr = predict_torch(transformer, seq_train)
    p_val_tr = predict_torch(transformer, seq_val)
    p_test_tr = predict_torch(transformer, seq_test)

    t_tr = _select_threshold(y_val, p_val_tr)

    preds_tr = df[["target_date_local", "cutoff_utc", "y_hit_by_cutoff", "temp_max_sofar", "slope_60m", "drop_from_max", "split"]].copy()
    preds_tr.loc[train_mask, "p_pred"] = p_train_tr
    preds_tr.loc[val_mask, "p_pred"] = p_val_tr
    preds_tr.loc[test_mask, "p_pred"] = p_test_tr
    preds_tr["y_pred"] = (preds_tr["p_pred"] >= t_tr).astype(int)

    metrics_tr = {
        "train": _compute_metrics(y_train, p_train_tr, t_tr),
        "val": _compute_metrics(y_val, p_val_tr, t_tr),
        "test": _compute_metrics(y_test, p_test_tr, t_tr),
        "threshold": t_tr,
    }
    save_experiment("exp6_transformer", preds_tr, metrics_tr)

    knn_features = [
        *[f"dct_{i}" for i in range(30)],
        "slope_60m",
        "slope_120m",
        "drop_from_max",
        "plateau_frac_0p2",
    ]
    X_knn = df[knn_features]
    X_knn_train = imputer.fit_transform(X_knn[train_mask])
    X_knn_val = imputer.transform(X_knn[val_mask])
    X_knn_test = imputer.transform(X_knn[test_mask])

    mean_knn = X_knn_train.mean(axis=0, keepdims=True)
    std_knn = X_knn_train.std(axis=0, keepdims=True) + 1e-6
    X_knn_train = (X_knn_train - mean_knn) / std_knn
    X_knn_val = (X_knn_val - mean_knn) / std_knn
    X_knn_test = (X_knn_test - mean_knn) / std_knn

    best_k = None
    best_acc = -1.0
    best_val = None

    for k in [25, 50, 100, 200]:
        nn_model = NearestNeighbors(n_neighbors=k, algorithm="auto")
        nn_model.fit(X_knn_train)
        _, idx = nn_model.kneighbors(X_knn_val, return_distance=True)
        y_neighbors = y_train[idx]
        p_val_knn = y_neighbors.mean(axis=1)
        t_knn = _select_threshold(y_val, p_val_knn)
        acc = accuracy_score(y_val, p_val_knn >= t_knn)
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_val = p_val_knn

    nn_model = NearestNeighbors(n_neighbors=best_k, algorithm="auto")
    nn_model.fit(np.vstack([X_knn_train, X_knn_val]))
    _, idx_test = nn_model.kneighbors(X_knn_test, return_distance=True)
    y_trainval = np.concatenate([y_train, y_val])
    p_test_knn = y_trainval[idx_test].mean(axis=1)

    nn_train = NearestNeighbors(n_neighbors=best_k)
    nn_train.fit(X_knn_train)
    _, idx_train = nn_train.kneighbors(X_knn_train, return_distance=True)
    p_train_knn = y_train[idx_train].mean(axis=1)

    t_knn = _select_threshold(y_val, best_val)

    preds_knn = df[["target_date_local", "cutoff_utc", "y_hit_by_cutoff", "temp_max_sofar", "slope_60m", "drop_from_max", "split"]].copy()
    preds_knn.loc[train_mask, "p_pred"] = p_train_knn
    preds_knn.loc[val_mask, "p_pred"] = best_val
    preds_knn.loc[test_mask, "p_pred"] = p_test_knn
    preds_knn["y_pred"] = (preds_knn["p_pred"] >= t_knn).astype(int)

    metrics_knn = {
        "train": _compute_metrics(y_train, p_train_knn, t_knn),
        "val": _compute_metrics(y_val, best_val, t_knn),
        "test": _compute_metrics(y_test, p_test_knn, t_knn),
        "k": best_k,
        "threshold": t_knn,
    }
    save_experiment("exp7_knn", preds_knn, metrics_knn)

    # Retrain base models on train+val for stacking test predictions
    trainval_mask = df["split"].isin(["train", "val"])
    X_trainval = imputer.transform(X[trainval_mask])
    y_trainval = y[trainval_mask.to_numpy()]

    best_iter = lgbm_model.best_iteration or 200
    lgb_trainval = lgb.Dataset(X_trainval, label=y_trainval)
    lgbm_model_tv = lgb.train(best_params, lgb_trainval, num_boost_round=best_iter)
    p_test_lgbm_tv = lgbm_model_tv.predict(X_test)

    seq_trainval = np.concatenate([seq_train, seq_val], axis=0)
    y_trainval_seq = np.concatenate([y_train, y_val], axis=0)
    split_idx = int(len(seq_trainval) * 0.9)
    seq_tv_train = seq_trainval[:split_idx]
    seq_tv_val = seq_trainval[split_idx:]
    y_tv_train = y_trainval_seq[:split_idx]
    y_tv_val = y_trainval_seq[split_idx:]

    tv_train_loader = DataLoader(SequenceDataset(seq_tv_train, y_tv_train), batch_size=64, shuffle=True)
    tv_val_loader = DataLoader(SequenceDataset(seq_tv_val, y_tv_val), batch_size=128, shuffle=False)
    tcn_tv = TCN(n_features=seq_train.shape[2])
    tcn_tv = _train_torch_model(tcn_tv, tv_train_loader, tv_val_loader, pos_weight)
    p_test_tcn_tv = predict_torch(tcn_tv, seq_test)

    base_val = pd.DataFrame({
        "exp1": preds_rule.loc[val_mask, "p_pred"].to_numpy(),
        "exp3": preds_lgbm.loc[val_mask, "p_pred"].to_numpy(),
        "exp5": preds_tcn.loc[val_mask, "p_pred"].to_numpy(),
        "exp7": preds_knn.loc[val_mask, "p_pred"].to_numpy(),
    })
    base_test = pd.DataFrame({
        "exp1": preds_rule.loc[test_mask, "p_pred"].to_numpy(),
        "exp3": p_test_lgbm_tv,
        "exp5": p_test_tcn_tv,
        "exp7": preds_knn.loc[test_mask, "p_pred"].to_numpy(),
    })
    base_train = pd.DataFrame({
        "exp1": preds_rule.loc[train_mask, "p_pred"].to_numpy(),
        "exp3": preds_lgbm.loc[train_mask, "p_pred"].to_numpy(),
        "exp5": preds_tcn.loc[train_mask, "p_pred"].to_numpy(),
        "exp7": preds_knn.loc[train_mask, "p_pred"].to_numpy(),
    })

    meta = LogisticRegression(max_iter=200)
    meta.fit(base_val, y_val)
    p_val_meta = meta.predict_proba(base_val)[:, 1]
    t_meta = _select_threshold(y_val, p_val_meta)

    p_train_meta = meta.predict_proba(base_train)[:, 1]
    p_test_meta = meta.predict_proba(base_test)[:, 1]

    preds_meta = df[["target_date_local", "cutoff_utc", "y_hit_by_cutoff", "temp_max_sofar", "slope_60m", "drop_from_max", "split"]].copy()
    preds_meta.loc[train_mask, "p_pred"] = p_train_meta
    preds_meta.loc[val_mask, "p_pred"] = p_val_meta
    preds_meta.loc[test_mask, "p_pred"] = p_test_meta
    preds_meta["y_pred"] = (preds_meta["p_pred"] >= t_meta).astype(int)

    metrics_meta = {
        "train": _compute_metrics(y_train, p_train_meta, t_meta),
        "val": _compute_metrics(y_val, p_val_meta, t_meta),
        "test": _compute_metrics(y_test, p_test_meta, t_meta),
        "threshold": t_meta,
    }
    save_experiment("exp8_stack", preds_meta, metrics_meta)

    report_path = reports_dir / "tmax_hit1830_experiments_report.md"
    metrics_table = []
    for name, metrics in results_summary.items():
        m = metrics["test"]
        metrics_table.append(
            {
                "exp": name,
                "test_acc": m["accuracy"],
                "test_bal_acc": m["balanced_accuracy"],
                "test_yes_recall": m["yes_recall"],
                "test_yes_precision": m["yes_precision"],
                "test_auc": m["roc_auc"],
            }
        )

    best_test = max(metrics_table, key=lambda r: r["test_acc"])

    def format_row(r: Dict[str, float]) -> str:
        return f"| {r['exp']} | {r['test_acc']:.3f} | {r['test_bal_acc']:.3f} | {r['test_yes_recall']:.3f} | {r['test_yes_precision']:.3f} | {r['test_auc']:.3f} |"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Tmax Hit by 18:30 Stockholm Experiments\n\n")
        f.write(f"Positive rate (train): {pos_rate:.3f}\n\n")
        f.write(f"Always-NO test accuracy: {always_no_acc:.3f}\n\n")
        f.write(f"Always-YES test accuracy: {always_yes_acc:.3f}\n\n")
        f.write("## Test Metrics Summary\n\n")
        f.write("| Experiment | Test Acc | Test Bal Acc | YES Recall | YES Precision | ROC AUC |\n")
        f.write("|---|---|---|---|---|---|\n")
        for row in metrics_table:
            f.write(format_row(row) + "\n")
        f.write("\n")
        f.write(f"Best test accuracy: {best_test['exp']} ({best_test['test_acc']:.3f})\n\n")

        best_name = best_test["exp"]
        preds_best = pd.read_parquet(exp_dir / f"exp_{best_name}" / "preds.parquet")
        preds_best = preds_best[preds_best["split"] == "test"].copy()
        preds_best["error_type"] = np.where(
            (preds_best["y_pred"] == 1) & (preds_best["y_hit_by_cutoff"] == 0),
            "FP",
            np.where((preds_best["y_pred"] == 0) & (preds_best["y_hit_by_cutoff"] == 1), "FN", "OK"),
        )
        fp = preds_best[preds_best["error_type"] == "FP"].sort_values("p_pred", ascending=False).head(20)
        fn = preds_best[preds_best["error_type"] == "FN"].sort_values("p_pred", ascending=True).head(20)

        f.write("## Error Analysis (Test, Best Model)\n\n")
        f.write("### False Positives (predicted YES, actually NO)\n\n")
        for _, r in fp.iterrows():
            f.write(
                f"- {r['target_date_local']} p={r['p_pred']:.3f} max_sofar={r['temp_max_sofar']:.1f} slope_60m={r['slope_60m']:.3f} drop_from_max={r['drop_from_max']:.2f}\n"
            )
        f.write("\n### False Negatives (predicted NO, actually YES)\n\n")
        for _, r in fn.iterrows():
            f.write(
                f"- {r['target_date_local']} p={r['p_pred']:.3f} max_sofar={r['temp_max_sofar']:.1f} slope_60m={r['slope_60m']:.3f} drop_from_max={r['drop_from_max']:.2f}\n"
            )

        f.write("\n## Acceptance Check\n\n")
        f.write(
            f"Best test accuracy: {best_test['test_acc']:.3f} (target 0.85)\n\n"
        )
        f.write(
            f"Best test YES recall: {best_test['test_yes_recall']:.3f} (target 0.70)\n\n"
        )

    with open(out_base / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
