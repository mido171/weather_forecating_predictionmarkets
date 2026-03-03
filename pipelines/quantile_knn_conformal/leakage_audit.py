from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class LeakageAuditResult:
    passed: bool
    checks: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": self.checks}


def _check(name: str, passed: bool, detail: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {"check": name, "passed": bool(passed), "detail": detail}
    if extra:
        out["extra"] = extra
    return out


def run_leakage_audit(
    all_rows: pd.DataFrame,
    dev_oof_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    tuning_rows: pd.DataFrame,
    knn_neighbor_diag: pd.DataFrame,
    gate_train_rows: pd.DataFrame,
    conformal_diag: pd.DataFrame,
    fold_boundaries: list[dict[str, Any]],
) -> LeakageAuditResult:
    checks: list[dict[str, Any]] = []

    # 1. Fold chronology
    fold_ok = True
    fold_details = []
    for f in fold_boundaries:
        tr_max = pd.Timestamp(f["train_max_date"]).date()
        pr_min = pd.Timestamp(f["pred_min_date"]).date()
        ok = tr_max < pr_min
        fold_ok = fold_ok and ok
        fold_details.append({"fold": f.get("fold"), "train_max_date": str(tr_max), "pred_min_date": str(pr_min), "ok": ok})
    checks.append(_check("fold_train_date_before_prediction", fold_ok, "Every fold must satisfy max(train_date) < min(pred_date)", {"folds": fold_details}))

    # 2. Neighbor as-of timestamp safety
    src_cols = [c for c in all_rows.columns if c.endswith("_source_valid_time_utc")]
    if src_cols:
        violations = 0
        for c in src_cols:
            m = pd.to_datetime(all_rows[c], errors="coerce", utc=True) > pd.to_datetime(all_rows["valid_time_utc"], utc=True)
            violations += int(m.fillna(False).sum())
        checks.append(_check("neighbor_asof_no_future", violations == 0, "Neighbor source timestamps must not be after cutoff", {"violations": violations}))
    else:
        checks.append(_check("neighbor_asof_no_future", False, "No neighbor source timestamp columns found"))

    # 3. KNN candidate date safety
    if knn_neighbor_diag is not None and not knn_neighbor_diag.empty:
        qd = pd.to_datetime(knn_neighbor_diag["query_date"], errors="coerce")
        nd = pd.to_datetime(knn_neighbor_diag["neighbor_date"], errors="coerce")
        bad_future = int((nd >= qd).sum())
        bad_same = int((nd == qd).sum())
        checks.append(_check("knn_no_future_or_same_day", bad_future == 0 and bad_same == 0, "KNN neighbor dates must be strictly before query date", {"violations_future_or_same": bad_future, "same_day": bad_same}))
    else:
        checks.append(_check("knn_no_future_or_same_day", False, "KNN diagnostics are missing"))

    # 4. Gate train rows are dev only
    gate_dates = pd.to_datetime(gate_train_rows["target_date_local"], errors="coerce")
    gate_ok = bool((gate_dates >= pd.Timestamp("2022-01-01")).all() and (gate_dates <= pd.Timestamp("2023-12-31")).all()) if len(gate_dates) > 0 else False
    checks.append(_check("gate_training_window", gate_ok, "Gate training rows must be from 2022-2023 OOF only", {"rows": int(len(gate_train_rows))}))

    # 5. Conformal chronology sanity (non-decreasing history length)
    conf_ok = True
    if conformal_diag is not None and not conformal_diag.empty and "hist_len" in conformal_diag.columns:
        hist = conformal_diag.sort_values("valid_time_utc")["hist_len"].to_numpy(dtype=float)
        conf_ok = bool(np.all(np.diff(hist) >= -1e-9))
    else:
        conf_ok = False
    checks.append(_check("conformal_buffer_chronology", conf_ok, "Conformal history length must be non-decreasing chronologically"))

    # 6. Tuning rows excluded from 2024-2025
    tune_dates = pd.to_datetime(tuning_rows["target_date_local"], errors="coerce")
    tune_ok = bool((tune_dates <= pd.Timestamp("2021-12-31")).all()) if len(tune_dates) > 0 else False
    checks.append(_check("tuning_window_pre2022", tune_ok, "Hyperparameter tuning must not use 2024-2025 rows", {"max_tuning_date": str(tune_dates.max().date()) if len(tune_dates) else None}))

    # 7. Timezone/date consistency
    ny_date = pd.to_datetime(all_rows["valid_time_ny"], errors="coerce").dt.date
    target_date = pd.to_datetime(all_rows["target_date_local"], errors="coerce").dt.date
    tz_ok = bool((ny_date == target_date).all())
    checks.append(_check("timezone_target_date_alignment", tz_ok, "target_date_local must equal date(valid_time_ny)"))

    # 8. Dev/test disjointness
    dev_dates = set(pd.to_datetime(dev_oof_rows["target_date_local"], errors="coerce").dt.date.tolist())
    test_dates = set(pd.to_datetime(test_rows["target_date_local"], errors="coerce").dt.date.tolist())
    overlap = len(dev_dates.intersection(test_dates))
    checks.append(_check("dev_test_date_disjoint", overlap == 0, "Dev and test date ranges must not overlap", {"overlap_dates": overlap}))

    passed = all(c["passed"] for c in checks)
    return LeakageAuditResult(passed=passed, checks=checks)


def leakage_audit_markdown(result: LeakageAuditResult) -> str:
    lines = ["# Leakage Audit", "", f"Overall pass: **{result.passed}**", ""]
    for c in result.checks:
        lines.append(f"- {c['check']}: {'PASS' if c['passed'] else 'FAIL'}")
        lines.append(f"  - {c['detail']}")
        if c.get("extra") is not None:
            lines.append(f"  - extra: {c['extra']}")
    lines.append("")
    return "\n".join(lines)
