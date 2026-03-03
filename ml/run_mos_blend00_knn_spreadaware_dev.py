from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_knyc_mos_first_plan as base


QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
STATE = [
    "doy_sin",
    "doy_cos",
    "mos_tmax_raw",
    "mos_tmin_raw",
    "mos_dtr_raw",
    "tmp_15",
    "tmp_18",
    "dep_15",
    "cloud_mean_12_21",
    "p06_max_day",
    "wsp_mean_12_21",
    "wdr_sin_mean_12_21",
    "wdr_cos_mean_12_21",
]


def wquant(values: np.ndarray, qs: list[float], w: np.ndarray) -> np.ndarray:
    idx = np.argsort(values)
    v = values[idx]
    ww = w[idx]
    c = np.cumsum(ww)
    if c[-1] <= 0:
        return np.full(len(qs), np.nan)
    c = c / c[-1]
    return np.array([np.interp(q, c, v) for q in qs], dtype=float)


def pinball(y: np.ndarray, q: np.ndarray, a: float) -> float:
    e = y - q
    return float(np.mean(np.maximum(a * e, (a - 1.0) * e)))


def avg_pin(y: np.ndarray, qmap: dict[float, np.ndarray]) -> float:
    return float(np.mean([pinball(y, qmap[q], q) for q in sorted(qmap)]))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad > 1e-9:
        den = mad * 1.4826
    else:
        std = np.nanstd(x)
        den = std if std > 1e-9 else 1.0
    return (x - med) / den


def main() -> None:
    parser = argparse.ArgumentParser(description="Spread-aware KNN trust gating on blend_00 dev window.")
    parser.add_argument("--mos-csv", default=r"D:\Ahmed\data\kalshi\training_data\KNYC_mos_archive_2000_2025.csv.gz")
    parser.add_argument("--truth-csv", default=r"D:\Ahmed\data\kalshi\training_data\KNYC_settled_tmax.csv")
    parser.add_argument("--blend-dev", default=r"D:\Ahmed\data\kalshi\Experiments\MOS\03_blends\blend_00\dev_predictions.parquet")
    parser.add_argument("--gfs-dev", default=r"D:\Ahmed\data\kalshi\Experiments\MOS\02_phaseB_residual\gfs_00\dev_predictions.parquet")
    parser.add_argument("--nam-dev", default=r"D:\Ahmed\data\kalshi\Experiments\MOS\02_phaseB_residual\nam_00\dev_predictions.parquet")
    parser.add_argument("--out-dir", default=r"D:\Ahmed\data\kalshi\Experiments\MOS\06_dev_knn_sweep")
    parser.add_argument("--dev-start", default="2022-01-01")
    parser.add_argument("--dev-end", default="2023-12-31")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build gfs_00 slice with longer history for analog candidates.
    truth = base.load_truth(Path(args.truth_csv))
    mos = base.load_mos(Path(args.mos_csv))
    sdef = base.SliceDef("gfs_00_2004", "GFS", 0, "2004-01-01")
    gfs = base.build_slice(mos, truth, sdef).sort_values("target_date_local").reset_index(drop=True)

    blend = pd.read_parquet(args.blend_dev).copy()
    gfs_dev = pd.read_parquet(args.gfs_dev).copy()
    nam_dev = pd.read_parquet(args.nam_dev).copy()
    for d in (blend, gfs_dev, nam_dev):
        d["target_date_local"] = pd.to_datetime(d["target_date_local"]).dt.normalize()
        d.sort_values("target_date_local", inplace=True)

    # Build KNN predictions using best prior combo: k32_exp_hl5y
    dev_mask = gfs["target_date_local"].between(pd.Timestamp(args.dev_start), pd.Timestamp(args.dev_end))
    pre_dev = gfs["target_date_local"] < pd.Timestamp(args.dev_start)

    state = gfs[STATE].copy()
    med = state.loc[pre_dev].median(axis=0, skipna=True).fillna(0.0)
    state = state.fillna(med)
    mu = state.loc[pre_dev].mean(axis=0)
    sd = state.loc[pre_dev].std(axis=0).replace(0, 1.0).fillna(1.0)
    z = ((state - mu) / sd).to_numpy(dtype=float)

    g_dates = pd.to_datetime(gfs["target_date_local"]).dt.normalize().to_numpy()
    g_res = (gfs["y_tmax"] - gfs["mos_tmax_raw"]).to_numpy(dtype=float)

    dev = gfs.loc[dev_mask, ["target_date_local", "y_tmax", "mos_tmax_raw"]].copy()
    dev = dev.merge(blend, on=["target_date_local", "y_tmax"], how="inner", suffixes=("", "_blend"))
    dev = dev.merge(gfs_dev[["target_date_local", "pred_point", "q_0.10", "q_0.50", "q_0.90"]], on="target_date_local", how="left", suffixes=("", "_g"))
    dev = dev.merge(nam_dev[["target_date_local", "pred_point", "q_0.10", "q_0.50", "q_0.90"]], on="target_date_local", how="left", suffixes=("", "_n"))
    dev = dev.sort_values("target_date_local").reset_index(drop=True)

    row_map = {pd.Timestamp(d).normalize(): i for i, d in enumerate(g_dates)}
    idx = np.array([row_map[pd.Timestamp(d).normalize()] for d in dev["target_date_local"]], dtype=int)

    n = len(dev)
    knn_point = np.full(n, np.nan)
    knn_q = {q: np.full(n, np.nan) for q in QUANTILES}
    dmin = np.full(n, np.nan)
    dmean = np.full(n, np.nan)
    effk = np.full(n, np.nan)
    iqr = np.full(n, np.nan)

    K = 32
    half_days = 5.0 * 365.25
    for j, gi in enumerate(idx):
        qd = pd.Timestamp(g_dates[gi]).normalize()
        cand = (gfs["target_date_local"] >= pd.Timestamp("2004-01-01")) & (gfs["target_date_local"] < qd)
        cidx = np.where(cand.to_numpy())[0]
        if len(cidx) < 30:
            continue
        xc = z[cidx]
        yc = g_res[cidx]
        ok = np.isfinite(xc).all(axis=1) & np.isfinite(yc)
        xc = xc[ok]
        yc = yc[ok]
        cidx = cidx[ok]
        if len(xc) < 30 or not np.isfinite(z[gi]).all():
            continue
        dist = np.sqrt(np.sum((xc - z[gi]) ** 2, axis=1))
        kk = min(K, len(dist))
        sidx = np.argpartition(dist, kk - 1)[:kk]
        ds = dist[sidx]
        ys = yc[sidx]
        dates = pd.to_datetime(g_dates[cidx[sidx]]).normalize()
        scale = float(np.median(ds)) if np.isfinite(ds).any() else 1.0
        if scale <= 1e-8:
            scale = 1.0
        w = np.exp(-ds / scale)
        age = (qd - dates).days.astype(float)
        w = w * np.exp(-np.log(2.0) * age / half_days)
        sw = float(np.sum(w))
        if sw <= 0:
            continue
        w = w / sw
        qv = wquant(ys, QUANTILES, w)
        qmap = {q: qv[i] for i, q in enumerate(QUANTILES)}
        anchor = float(gfs.iloc[gi]["mos_tmax_raw"])
        knn_point[j] = anchor + qmap[0.50]
        for q in QUANTILES:
            knn_q[q][j] = anchor + qmap[q]
        dmin[j] = float(np.min(ds))
        dmean[j] = float(np.mean(ds))
        effk[j] = float(1.0 / np.sum(w**2))
        iqr[j] = float(qmap[0.75] - qmap[0.25])

    valid = np.isfinite(knn_point)
    d = dev.loc[valid].copy().reset_index(drop=True)
    y = d["y_tmax"].to_numpy(dtype=float)

    blend_point = d["pred_point"].to_numpy(dtype=float)
    blend_q = {q: d[f"q_{q:.2f}"].to_numpy(dtype=float) for q in QUANTILES}
    kp = knn_point[valid]
    kq = {q: knn_q[q][valid] for q in QUANTILES}

    # Spread features from GFS/NAM dev predictions
    spread_center = np.abs(d["q_0.50_g"].to_numpy(dtype=float) - d["q_0.50_n"].to_numpy(dtype=float))
    spread_width = np.abs(
        (d["q_0.90_g"].to_numpy(dtype=float) - d["q_0.10_g"].to_numpy(dtype=float))
        - (d["q_0.90_n"].to_numpy(dtype=float) - d["q_0.10_n"].to_numpy(dtype=float))
    )
    spread_point = np.abs(d["pred_point_g"].to_numpy(dtype=float) - d["pred_point_n"].to_numpy(dtype=float))

    # Base KNN trust score
    score_base_raw = (
        -1.0 * robust_z(dmean[valid])
        -0.5 * robust_z(dmin[valid])
        -0.7 * robust_z(iqr[valid])
        +0.5 * robust_z(effk[valid])
    )

    rows = []
    best = None
    for b_center in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for b_width in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for b_point in [0.0, 0.25, 0.5, 0.75, 1.0]:
                raw = score_base_raw - b_center * robust_z(spread_center) - b_width * robust_z(spread_width) - b_point * robust_z(spread_point)
                sc = sigmoid(raw)
                for a_lo in np.round(np.arange(0.50, 0.96, 0.05), 2):
                    for a_hi in np.round(np.arange(max(a_lo, 0.70), 1.0001, 0.05), 2):
                        for gamma in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
                            s = np.power(sc, gamma)
                            alpha = a_lo + (a_hi - a_lo) * s
                            pt = alpha * blend_point + (1.0 - alpha) * kp
                            qmap = {q: alpha * blend_q[q] + (1.0 - alpha) * kq[q] for q in QUANTILES}
                            mae = float(np.mean(np.abs(y - pt)))
                            ap = avg_pin(y, qmap)
                            row = {
                                "beta_center": b_center,
                                "beta_width": b_width,
                                "beta_point": b_point,
                                "a_lo": float(a_lo),
                                "a_hi": float(a_hi),
                                "gamma": float(gamma),
                                "mae": mae,
                                "avg_pinball": ap,
                                "alpha_mean": float(np.mean(alpha)),
                                "alpha_min": float(np.min(alpha)),
                                "alpha_max": float(np.max(alpha)),
                            }
                            rows.append(row)
                            if best is None or mae < best["mae"] or (abs(mae - best["mae"]) < 1e-12 and ap < best["avg_pinball"]):
                                best = row

    grid = pd.DataFrame(rows).sort_values(["mae", "avg_pinball"]).reset_index(drop=True)
    grid.to_csv(out_dir / "knn_spreadaware_gated_alpha_grid_dev.csv", index=False)

    assert best is not None
    base_mae = float(np.mean(np.abs(y - blend_point)))
    base_pin = avg_pin(y, blend_q)
    const_alpha = 0.90
    const_pt = const_alpha * blend_point + (1.0 - const_alpha) * kp
    const_q = {q: const_alpha * blend_q[q] + (1.0 - const_alpha) * kq[q] for q in QUANTILES}
    const_mae = float(np.mean(np.abs(y - const_pt)))
    const_pin = avg_pin(y, const_q)

    summary = {
        "generated_at_utc": base.utc_now_iso(),
        "dev_rows_used": int(len(d)),
        "holdout_untouched": ["2024-01-01", "2025-12-31"],
        "knn_combo": "k32_exp_hl5y",
        "baseline_blend00_dev_mae": base_mae,
        "baseline_blend00_dev_avg_pinball": base_pin,
        "const_alpha_0p90_dev_mae": const_mae,
        "const_alpha_0p90_dev_avg_pinball": const_pin,
        "best_spreadaware": best,
        "mae_gain_vs_blend": base_mae - best["mae"],
        "mae_gain_vs_const_alpha": const_mae - best["mae"],
        "pinball_gain_vs_blend": base_pin - best["avg_pinball"],
        "pinball_gain_vs_const_alpha": const_pin - best["avg_pinball"],
    }
    (out_dir / "knn_spreadaware_gated_alpha_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("Top 10:")
    print(grid.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

