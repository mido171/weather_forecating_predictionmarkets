from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


STAGE_DIRS = [
    "00_config_snapshot",
    "01_data_contract",
    "02_row_dataset",
    "03_baselines",
    "04_ml_quantiles",
    "05_knn_analog",
    "06_gate",
    "07_conformal",
    "08_predictions",
    "09_reports",
    "10_bundle",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_stage_tree(root: str | Path) -> dict[str, Path]:
    root_p = Path(root)
    root_p.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for s in STAGE_DIRS:
        p = root_p / s
        p.mkdir(parents=True, exist_ok=True)
        out[s] = p
    return out


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def write_manifest(stage_dir: str | Path, payload: dict[str, Any]) -> None:
    d = dict(payload)
    d.setdefault("created_at_utc", utc_now_iso())
    write_json(Path(stage_dir) / "manifest.json", d)


def write_df(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".csv":
        df.to_csv(p, index=index)
    elif p.suffix.lower() == ".parquet":
        df.to_parquet(p, index=index)
    else:
        raise ValueError(f"Unsupported dataframe output extension: {p}")


def model_comparison_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    cols = [
        "model",
        "decision_row_mae",
        "all_rows_mae",
        "avg_pinball",
        "80_cov",
        "90_cov",
        "95_cov",
        "avg_90_width",
        "bucket_brier",
        "bucket_logloss",
        "bucket_ece10",
        "pit_ks_pvalue",
        "deployable_flag",
    ]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def executive_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Quantile + KNN + Gate + Conformal Results",
        "",
        f"- Leakage audit pass: **{summary.get('leakage_audit_pass')}**",
        f"- Deployable verdict: **{summary.get('deploy_recommendation', {}).get('deployable')}**",
        f"- Winner by decision-row MAE: **{summary.get('deploy_recommendation', {}).get('winner_by_mae')}**",
        f"- Winner by decision-row bucket ECE: **{summary.get('deploy_recommendation', {}).get('winner_by_bucket_ece')}**",
        "",
        "## What Helped",
        f"- KNN helped vs ML-only: **{summary.get('deploy_recommendation', {}).get('knn_helped')}**",
        f"- Gate helped vs best standalone: **{summary.get('deploy_recommendation', {}).get('gate_helped')}**",
        f"- Conformal improved coverage: **{summary.get('deploy_recommendation', {}).get('conformal_helped')}**",
        "",
        "## Top Failure Modes",
    ]
    failures = summary.get("deploy_recommendation", {}).get("top_failure_modes", [])
    if not failures:
        lines.append("- none")
    else:
        for f in failures[:3]:
            lines.append(f"- {f}")
    lines.append("")
    return "\n".join(lines)
