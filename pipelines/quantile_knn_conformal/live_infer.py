from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cdf_bucket_mapper import integer_pmf_to_bucket_probs, load_buckets, quantile_rows_to_integer_pmf
from .train_gate import blend_quantiles, build_gate_features, predict_gate_alpha
from .train_quantiles import predict_quantile_models, repair_quantile_crossings


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def run_live_infer(bundle_dir: str, feature_row_json: str) -> dict[str, Any]:
    bdir = Path(bundle_dir)
    quantile_pack = _load_pickle(bdir / "quantile_models.pkl")
    knn_model = _load_pickle(bdir / "knn_model.pkl")
    gate_pack = _load_pickle(bdir / "gate_model.pkl")
    meta = json.loads((bdir / "bundle_manifest.json").read_text(encoding="utf-8"))
    quantiles = [float(x) for x in meta["quantiles"]]
    buckets = load_buckets(str(bdir / "bucket_config_snapshot.yaml"))

    row = pd.DataFrame([json.loads(feature_row_json)])

    missing_critical = [c for c in quantile_pack.feature_cols if c not in row.columns]
    for c in missing_critical:
        row[c] = np.nan

    ml_q = predict_quantile_models(quantile_pack, row)

    from .knn_analog import predict_knn_analog

    knn_q, knn_trust = predict_knn_analog(knn_model, row, quantiles)
    gate_x = build_gate_features(row, ml_q, knn_q, knn_trust)
    alpha = predict_gate_alpha(gate_pack, gate_x)
    blend_q = blend_quantiles(ml_q, knn_q, alpha)
    blend_q, _ = repair_quantile_crossings(blend_q, quantiles)

    pmf = quantile_rows_to_integer_pmf(blend_q, quantiles)
    bprob = integer_pmf_to_bucket_probs(pmf, buckets)

    out = {
        "median_tmax": float(blend_q.iloc[0]["q_0.500"]),
        "quantile_grid": {f"{q:.3f}": float(blend_q.iloc[0][f"q_{q:.3f}"]) for q in quantiles},
        "intervals": {
            "50": [float(blend_q.iloc[0]["q_0.250"]), float(blend_q.iloc[0]["q_0.750"])],
            "80": [float(blend_q.iloc[0]["q_0.100"]), float(blend_q.iloc[0]["q_0.900"])],
            "90": [float(blend_q.iloc[0]["q_0.050"]), float(blend_q.iloc[0]["q_0.950"])],
            "95": [float(blend_q.iloc[0]["q_0.025"]), float(blend_q.iloc[0]["q_0.975"])],
        },
        "top5_integers": pmf.iloc[0]["top5_temps"],
        "alpha": float(alpha[0]),
        "knn_trust": {k: float(v) if np.isfinite(v) else None for k, v in knn_trust.iloc[0].to_dict().items()},
        "bucket_probabilities": {k: float(v) for k, v in bprob.iloc[0].to_dict().items()},
        "warnings": [],
    }
    if missing_critical:
        out["warnings"].append(f"Missing critical features filled as NaN: {missing_critical[:15]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--feature-row-json", required=True)
    args = ap.parse_args()

    out = run_live_infer(args.bundle_dir, args.feature_row_json)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
