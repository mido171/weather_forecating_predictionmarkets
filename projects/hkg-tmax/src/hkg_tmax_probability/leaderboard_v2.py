"""V2 promotion gates for HKG Tmax probability distribution benchmarks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

B4_METHOD = "B4_hierarchical_residual_pmf"


def _metric(frame: pd.DataFrame, method: str, column: str) -> float:
    rows = frame[frame["method"] == method]
    if rows.empty or column not in rows.columns:
        return np.nan
    return float(rows[column].iloc[0])


def _gain_vs_b4(b4_value: float, method_value: float) -> float:
    if np.isnan(b4_value) or np.isnan(method_value) or b4_value == 0.0:
        return np.nan
    return (b4_value - method_value) / b4_value


def apply_v2_champion_gates(
    overall: pd.DataFrame,
    fold14: pd.DataFrame,
    presealed: pd.DataFrame,
    gates: dict[str, Any],
    leakage_pass: bool = True,
    row_identity_pass: bool = True,
) -> pd.DataFrame:
    """Apply the predeclared V2 promotion contract.

    Raw rank is still sorted by normalized RPS ascending.  Champion selection is
    stricter: a challenger must beat B4 by the fold/presealed thresholds and
    must not materially worsen NLL or Brier.
    """
    out = overall.sort_values("rps", ascending=True).reset_index(drop=True).copy()
    if "rank" in out.columns:
        out = out.drop(columns=["rank"])
    out.insert(0, "rank", np.arange(1, len(out) + 1))

    b4_overall_rps = _metric(out, B4_METHOD, "rps")
    b4_overall_nll = _metric(out, B4_METHOD, "nll")
    b4_overall_brier = _metric(out, B4_METHOD, "brier")
    b4_fold_rps = _metric(fold14, B4_METHOD, "rps")
    b4_presealed_rps = _metric(presealed, B4_METHOD, "rps")

    fold_min = float(gates.get("complex_vs_b4_fold14_min_rps_gain", 0.015))
    presealed_min = float(gates.get("complex_vs_b4_presealed_min_rps_gain", 0.010))
    nll_margin = float(gates.get("nll_worse_than_b4_max", 0.005))
    brier_margin = float(gates.get("brier_worse_than_b4_max", 0.002))

    fold_scores = []
    presealed_scores = []
    fold_gains = []
    presealed_gains = []
    gate_labels = []
    pass_flags = []

    for _, row in out.iterrows():
        method = str(row["method"])
        fold_rps = _metric(fold14, method, "rps")
        pre_rps = _metric(presealed, method, "rps")
        fold_gain = _gain_vs_b4(b4_fold_rps, fold_rps)
        pre_gain = _gain_vs_b4(b4_presealed_rps, pre_rps)
        fold_scores.append(fold_rps)
        presealed_scores.append(pre_rps)
        fold_gains.append(fold_gain)
        presealed_gains.append(pre_gain)

        if method == B4_METHOD:
            failures = []
            label_prefix = "reference"
        else:
            label_prefix = "pass"
            failures = []
            if np.isnan(fold_gain) or fold_gain < fold_min:
                failures.append("fold14_rps_gain")
            if np.isnan(pre_gain) or pre_gain < presealed_min:
                failures.append("presealed_rps_gain")
            if not np.isnan(b4_overall_nll) and float(row["nll"]) > b4_overall_nll + nll_margin:
                failures.append("nll")
            if not np.isnan(b4_overall_brier) and float(row["brier"]) > b4_overall_brier + brier_margin:
                failures.append("brier")
            if not leakage_pass:
                failures.append("leakage_audit")
            if not row_identity_pass:
                failures.append("row_identity")
            if not np.isnan(b4_overall_rps) and float(row["rps"]) >= b4_overall_rps:
                failures.append("overall_rps_not_better")
        pass_flags.append(method != B4_METHOD and not failures)
        gate_labels.append(label_prefix if not failures else "fail:" + ",".join(failures))

    out["fold14_rps"] = fold_scores
    out["fold14_relative_rps_gain_vs_b4"] = fold_gains
    out["presealed_rps"] = presealed_scores
    out["presealed_relative_rps_gain_vs_b4"] = presealed_gains
    out["v2_promotion_pass"] = pass_flags
    out["gates"] = gate_labels

    promoted = out[out["v2_promotion_pass"]].sort_values("rps", ascending=True)
    champion = str(promoted["method"].iloc[0]) if not promoted.empty else B4_METHOD
    out["champion_flag"] = out["method"].eq(champion)
    return out
