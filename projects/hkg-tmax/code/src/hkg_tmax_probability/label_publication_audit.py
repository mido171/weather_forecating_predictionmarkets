"""First-publication target-label audit against raw Daily Extract archive rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from hkg_tmax_probability.bucket_rules import bucket_key
from hkg_tmax_probability.data_build import DEFAULT_DATABASE_URL, query_frame

RAW_AUDIT_TABLE = "raw_audit.codex_audit_ds_01_hko_daily_tmax_target_hko_daily_extr_23cb54da"


def run_label_publication_audit(
    canonical: pd.DataFrame,
    database_url: str = DEFAULT_DATABASE_URL,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sql = f"""
    select local_date,
           absolute_daily_max_c::float8 as first_publication_tmax_c,
           ingested_at_utc,
           raw_retrieved_at_utc,
           ingest_source_file,
           ingest_source_file_id,
           ingest_source_row_number,
           content_sha256
    from {RAW_AUDIT_TABLE}
    where local_date is not null
      and absolute_daily_max_c is not null
    """
    try:
        raw = query_frame(database_url, sql)
    except Exception as exc:  # pragma: no cover - exercised only when DB table is absent
        details = {
            "status": "unavailable",
            "raw_audit_table": RAW_AUDIT_TABLE,
            "error": str(exc),
            "canonical_rows": int(len(canonical)),
            "bucket_changes": None,
            "scoreboard_required": False,
        }
        empty = pd.DataFrame()
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "label_publication_audit.json").write_text(json.dumps(details, indent=2, default=str), encoding="utf-8")
            empty.to_csv(output_dir / "label_publication_audit.csv", index=False)
        return empty, details

    if raw.empty:
        details = {
            "status": "empty",
            "raw_audit_table": RAW_AUDIT_TABLE,
            "canonical_rows": int(len(canonical)),
            "raw_rows": 0,
            "bucket_changes": 0,
            "scoreboard_required": False,
        }
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "label_publication_audit.json").write_text(json.dumps(details, indent=2, default=str), encoding="utf-8")
            raw.to_csv(output_dir / "label_publication_audit.csv", index=False)
        return raw, details

    raw["target_date"] = pd.to_datetime(raw["local_date"], errors="coerce")
    raw["ingested_at_utc"] = pd.to_datetime(raw["ingested_at_utc"], utc=True, errors="coerce")
    raw["first_publication_bucket_key"] = raw["first_publication_tmax_c"].map(bucket_key)
    first = raw.sort_values(
        ["target_date", "ingested_at_utc", "ingest_source_file_id", "ingest_source_row_number"],
        kind="mergesort",
    ).groupby("target_date", as_index=False).first()
    base = canonical[["target_date", "target_tmax_c", "bucket_key"]].drop_duplicates("target_date").copy()
    base["target_date"] = pd.to_datetime(base["target_date"])
    audit = base.merge(
        first[
            [
                "target_date",
                "first_publication_tmax_c",
                "first_publication_bucket_key",
                "ingested_at_utc",
                "raw_retrieved_at_utc",
                "ingest_source_file",
                "content_sha256",
            ]
        ],
        on="target_date",
        how="left",
    )
    audit["has_first_publication"] = audit["first_publication_tmax_c"].notna()
    audit["canonical_bucket_key"] = audit["bucket_key"]
    audit["bucket_changed_first_vs_canonical"] = audit["has_first_publication"] & (
        audit["canonical_bucket_key"] != audit["first_publication_bucket_key"]
    )
    details = {
        "status": "ok",
        "raw_audit_table": RAW_AUDIT_TABLE,
        "canonical_rows": int(len(base)),
        "raw_rows": int(len(raw)),
        "first_publication_rows": int(audit["has_first_publication"].sum()),
        "bucket_changes": int(audit["bucket_changed_first_vs_canonical"].sum()),
        "scoreboard_required": bool(audit["bucket_changed_first_vs_canonical"].any()),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        audit.to_csv(output_dir / "label_publication_audit.csv", index=False)
        (output_dir / "label_publication_audit.json").write_text(json.dumps(details, indent=2, default=str), encoding="utf-8")
    return audit, details


def apply_first_publication_labels(modeling: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty or "first_publication_bucket_key" not in audit.columns:
        out = modeling.copy()
        out["first_publication_bucket_key"] = out["bucket_key"]
        out["first_publication_bucket_index"] = out["bucket_index"]
        return out
    merged = modeling.merge(
        audit[["target_date", "first_publication_tmax_c", "first_publication_bucket_key"]],
        on="target_date",
        how="left",
    )
    merged["first_publication_bucket_key"] = merged["first_publication_bucket_key"].fillna(merged["bucket_key"])
    bucket_to_idx = {
        "24_or_below": 0,
        "25": 1,
        "26": 2,
        "27": 3,
        "28": 4,
        "29": 5,
        "30": 6,
        "31": 7,
        "32": 8,
        "33": 9,
        "34_or_higher": 10,
    }
    merged["first_publication_bucket_index"] = merged["first_publication_bucket_key"].map(bucket_to_idx).astype(int)
    return merged
