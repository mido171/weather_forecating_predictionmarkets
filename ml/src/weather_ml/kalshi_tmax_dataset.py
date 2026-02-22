"""Kalshi Tmax dataset builder (MOS daily + optional time-step MOS + extra features)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .kalshi_tmax_features import add_kalshi_extra_features
from .mos_config import MosDatasetConfig
from .mos_dataset_builder import build_dataset as build_mos_dataset
from .mos_dataset_builder import compute_feature_order as compute_mos_feature_order
from .mos_time_step_features import add_mos_time_step_features
from .mos_utils import utc_now


def build_dataset(cfg: MosDatasetConfig, engine) -> tuple[pd.DataFrame, dict[str, Any]]:
    df, metadata = build_mos_dataset(cfg, engine)
    df = add_kalshi_extra_features(df, cfg)

    time_step_meta = {}
    if cfg.use_time_step_mos:
        start_target = df["target_date_local"].min()
        end_target = df["target_date_local"].max()
        df, time_step_meta = add_mos_time_step_features(
            df, engine, cfg, start_target=start_target, end_target=end_target
        )

    metadata = metadata.copy()
    metadata["kalshi_extra_features_added_at_utc"] = utc_now().isoformat()
    metadata["kalshi_time_step_meta"] = time_step_meta
    metadata["feature_columns"] = list(df.columns)
    return df, metadata


def write_outputs(df: pd.DataFrame, metadata: dict[str, Any], cfg: MosDatasetConfig) -> Path:
    output_root = Path(cfg.output_root)
    output_dir = output_root / "kmia_kalshi_tmax" / cfg.feature_version / metadata["config_hash"]
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_list_path = output_dir / "feature_list.json"
    feature_list_path.write_text(json.dumps(list(df.columns), indent=2), encoding="utf-8")

    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    ordered = compute_mos_feature_order(df)
    extra_cols = sorted([c for c in df.columns if c not in ordered])
    ordered = ordered + extra_cols
    df = df[ordered]

    out_path = output_dir / "features.csv"
    tmp_path = out_path.with_suffix(".csv.tmp")
    df.to_csv(tmp_path, index=False, na_rep="")
    tmp_path.replace(out_path)
    return output_dir
