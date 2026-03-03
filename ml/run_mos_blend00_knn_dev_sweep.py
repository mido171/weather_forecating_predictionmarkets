from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import run_knyc_mos_first_plan as base


QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
STATE_COLS = [
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


@dataclass(frozen=True)
class Combo:
    k: int
    weight_mode: str
    half_life_years: float | None

    def key(self) -> str:
        hl = "none" if self.half_life_years is None else f"{self.half_life_years:g}y"
        return f"k{self.k}_{self.weight_mode}_hl{hl}"


def weighted_quantile(values: np.ndarray, quantiles: list[float], w: np.ndarray) -> np.ndarray:
    idx = np.argsort(values)
    v = values[idx]
    ww = w[idx]
    c = np.cumsum(ww)
    if c[-1] <= 0:
        return np.full(len(quantiles), np.nan)
    c = c / c[-1]
    return np.array([np.interp(q, c, v) for q in quantiles], dtype=float)


def pinball(y: np.ndarray, q: np.ndarray, a: float) -> float:
    e = y - q
    return float(np.mean(np.maximum(a * e, (a - 1.0) * e)))


def avg_pinball(y: np.ndarray, qmap: dict[float, np.ndarray]) -> float:
    vals = [pinball(y, qmap[q], q) for q in sorted(qmap)]
    return float(np.mean(vals))


def build_gfs00_slice_2004(mos_csv: Path, truth_csv: Path) -> pd.DataFrame:
    truth = base.load_truth(truth_csv)
    mos = base.load_mos(mos_csv)
    sdef = base.SliceDef("gfs_00_2004", "GFS", 0, "2004-01-01")
    df = base.build_slice(mos, truth, sdef)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Dev-only KNN sweep to aid blend_00")
    parser.add_argument("--mos-csv", default=r"D:\Ahmed\data\kalshi\training_data\KNYC_mos_archive_2000_2025.csv.gz")
    parser.add_argument("--truth-csv", default=r"D:\Ahmed\data\kalshi\training_data\KNYC_settled_tmax.csv")
    parser.add_argument("--blend-dev", default=r"D:\Ahmed\data\kalshi\Experiments\MOS\03_blends\blend_00\dev_predictions.parquet")
    parser.add_argument("--out-dir", default=r"D:\Ahmed\data\kalshi\Experiments\MOS\06_dev_knn_sweep")
    parser.add_argument("--dev-start", default="2022-01-01")
    parser.add_argument("--dev-end", default="2023-12-31")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build / load data
    gfs = build_gfs00_slice_2004(Path(args.mos_csv), Path(args.truth_csv))
    gfs = gfs.sort_values("target_date_local").reset_index(drop=True)
    blend = pd.read_parquet(args.blend_dev).copy()
    blend["target_date_local"] = pd.to_datetime(blend["target_date_local"]).dt.normalize()
    blend = blend.sort_values("target_date_local").reset_index(drop=True)

    dev_mask = gfs["target_date_local"].between(pd.Timestamp(args.dev_start), pd.Timestamp(args.dev_end))
    pre_dev_mask = gfs["target_date_local"] < pd.Timestamp(args.dev_start)

    # Standardize state with train-history-only stats (<=2021)
    state = gfs[STATE_COLS].copy()
    med = state.loc[pre_dev_mask].median(axis=0, skipna=True).fillna(0.0)
    state = state.fillna(med)
    mu = state.loc[pre_dev_mask].mean(axis=0)
    sd = state.loc[pre_dev_mask].std(axis=0).replace(0, 1.0).fillna(1.0)
    z = (state - mu) / sd

    # Dev query table aligned to blend_00 dev rows
    dev = gfs.loc[dev_mask, ["target_date_local", "y_tmax", "mos_tmax_raw"]].copy()
    dev = dev.merge(
        blend[
            [
                "target_date_local",
                "pred_point",
                "q_0.05",
                "q_0.10",
                "q_0.25",
                "q_0.50",
                "q_0.75",
                "q_0.90",
                "q_0.95",
            ]
        ],
        on="target_date_local",
        how="inner",
        suffixes=("", "_blend"),
    )
    dev = dev.sort_values("target_date_local").reset_index(drop=True)

    # Index mapping from dev rows to gfs rows
    row_map = {pd.Timestamp(d).normalize(): i for i, d in enumerate(gfs["target_date_local"])}
    dev_idx = np.array([row_map[pd.Timestamp(d).normalize()] for d in dev["target_date_local"]], dtype=int)

    y_dev = dev["y_tmax"].to_numpy(dtype=float)
    blend_point = dev["pred_point"].to_numpy(dtype=float)
    blend_q = {q: dev[f"q_{q:.2f}"].to_numpy(dtype=float) for q in QUANTILES}

    combos = [
        Combo(k=k, weight_mode=wm, half_life_years=hl)
        for k in [32, 64, 128, 200, 256]
        for wm in ["exp", "inv"]
        for hl in [None, 20.0, 10.0, 5.0]
    ]
    alpha_grid = np.round(np.arange(0.0, 1.0001, 0.05), 2).tolist()

    results = []
    all_preds = []

    # Precompute candidate pools for each dev query (leakage-safe: candidate_date < query_date)
    g_dates = gfs["target_date_local"].to_numpy()
    g_resid = (gfs["y_tmax"] - gfs["mos_tmax_raw"]).to_numpy(dtype=float)
    z_all = z.to_numpy(dtype=float)

    for combo in combos:
        n = len(dev)
        knn_point = np.full(n, np.nan, dtype=float)
        knn_q = {q: np.full(n, np.nan, dtype=float) for q in QUANTILES}

        for j, gi in enumerate(dev_idx):
            q_date = pd.Timestamp(g_dates[gi]).normalize()
            cand_mask = (gfs["target_date_local"] >= pd.Timestamp("2004-01-01")) & (gfs["target_date_local"] < q_date)
            idx = np.where(cand_mask.to_numpy())[0]
            if len(idx) < 30:
                continue
            xc = z_all[idx]
            yc = g_resid[idx]
            valid = np.isfinite(xc).all(axis=1) & np.isfinite(yc)
            xc = xc[valid]
            yc = yc[valid]
            idx_valid = idx[valid]
            if len(xc) < 30:
                continue
            xq = z_all[gi]
            if not np.isfinite(xq).all():
                continue
            dist = np.sqrt(np.sum((xc - xq) ** 2, axis=1))
            kk = min(combo.k, len(dist))
            sidx = np.argpartition(dist, kk - 1)[:kk]
            dsel = dist[sidx]
            ysel = yc[sidx]
            cand_dates = pd.to_datetime(g_dates[idx_valid[sidx]]).normalize()

            if combo.weight_mode == "exp":
                scale = float(np.median(dsel)) if np.isfinite(dsel).any() else 1.0
                if scale <= 1e-8:
                    scale = 1.0
                w = np.exp(-dsel / scale)
            else:
                w = 1.0 / (dsel + 1e-6)

            if combo.half_life_years is not None:
                half_days = combo.half_life_years * 365.25
                age_days = (q_date - cand_dates).days.astype(float)
                rec = np.exp(-np.log(2.0) * age_days / half_days)
                w = w * rec

            sw = float(np.sum(w))
            if sw <= 0:
                continue
            w = w / sw
            qvals = weighted_quantile(ysel, QUANTILES, w)
            qmap = {q: qvals[i] for i, q in enumerate(QUANTILES)}
            knn_point[j] = float(gfs.iloc[gi]["mos_tmax_raw"]) + qmap[0.50]
            for q in QUANTILES:
                knn_q[q][j] = float(gfs.iloc[gi]["mos_tmax_raw"]) + qmap[q]

        valid = np.isfinite(knn_point)
        if valid.sum() < 100:
            continue

        yv = y_dev[valid]
        bp = blend_point[valid]
        bq = {q: blend_q[q][valid] for q in QUANTILES}
        kp = knn_point[valid]
        kq = {q: knn_q[q][valid] for q in QUANTILES}

        # standalone KNN metrics
        knn_mae = float(np.mean(np.abs(yv - kp)))
        knn_pin = avg_pinball(yv, kq)

        best_by_mae = None
        best_by_pin = None
        for a in alpha_grid:
            pt = a * bp + (1.0 - a) * kp
            qmap = {q: a * bq[q] + (1.0 - a) * kq[q] for q in QUANTILES}
            mae = float(np.mean(np.abs(yv - pt)))
            apin = avg_pinball(yv, qmap)
            row = {
                "combo": combo.key(),
                "k": combo.k,
                "weight_mode": combo.weight_mode,
                "half_life_years": combo.half_life_years,
                "alpha": a,
                "valid_dev_rows": int(valid.sum()),
                "blend_dev_mae_ref": float(np.mean(np.abs(yv - bp))),
                "blend_dev_avg_pinball_ref": avg_pinball(yv, bq),
                "knn_dev_mae": knn_mae,
                "knn_dev_avg_pinball": knn_pin,
                "blend_plus_knn_dev_mae": mae,
                "blend_plus_knn_dev_avg_pinball": apin,
            }
            all_preds.append(row)

            if best_by_mae is None or mae < best_by_mae["blend_plus_knn_dev_mae"]:
                best_by_mae = row
            if best_by_pin is None or apin < best_by_pin["blend_plus_knn_dev_avg_pinball"]:
                best_by_pin = row

        assert best_by_mae is not None and best_by_pin is not None
        results.append(
            {
                **{k: v for k, v in best_by_mae.items() if k not in {"alpha", "blend_plus_knn_dev_mae", "blend_plus_knn_dev_avg_pinball"}},
                "best_alpha_by_mae": best_by_mae["alpha"],
                "best_dev_mae": best_by_mae["blend_plus_knn_dev_mae"],
                "best_dev_avg_pinball_at_best_mae": best_by_mae["blend_plus_knn_dev_avg_pinball"],
                "best_alpha_by_pinball": best_by_pin["alpha"],
                "best_dev_avg_pinball": best_by_pin["blend_plus_knn_dev_avg_pinball"],
                "best_dev_mae_at_best_pinball": best_by_pin["blend_plus_knn_dev_mae"],
            }
        )

    res_df = pd.DataFrame(results).sort_values(["best_dev_mae", "best_dev_avg_pinball"], na_position="last")
    full_df = pd.DataFrame(all_preds)

    res_path = out_dir / "knn_blend00_dev_sweep_summary.csv"
    full_path = out_dir / "knn_blend00_dev_sweep_fullgrid.csv"
    res_df.to_csv(res_path, index=False)
    full_df.to_csv(full_path, index=False)

    summary = {
        "generated_at_utc": base.utc_now_iso(),
        "dev_window": [args.dev_start, args.dev_end],
        "holdout_untouched": ["2024-01-01", "2025-12-31"],
        "combos_tested": int(len(res_df)),
        "full_grid_rows": int(len(full_df)),
        "best_by_dev_mae": (res_df.iloc[0].to_dict() if len(res_df) else {}),
    }
    (out_dir / "knn_blend00_dev_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Wrote:", res_path)
    print("Wrote:", full_path)
    print("Wrote:", out_dir / "knn_blend00_dev_sweep_summary.json")
    if len(res_df):
        print("Best by dev MAE:")
        print(res_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

