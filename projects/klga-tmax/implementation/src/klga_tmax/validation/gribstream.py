from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from klga_tmax.db.migrations_check import ContractInspection
from klga_tmax.providers.gribstream.plan import MODEL_SPECS, supported_cutoff_ids


def validate_gribstream(connection: Connection) -> ContractInspection:
    result = ContractInspection()
    cutoff_ids = supported_cutoff_ids()
    cutoff_rows = connection.execute(
        text("SELECT cutoff_id FROM registry.cutoffs"),
    ).scalars().all()
    missing_cutoffs = sorted(set(cutoff_ids) - set(cutoff_rows))
    for cutoff_id in missing_cutoffs:
        result.failures.append(f"missing cutoff registry row {cutoff_id}")

    required_tables = (
        "audit.gribstream_catalog_snapshots",
        "audit.gribstream_backfill_jobs",
        "audit.gribstream_backfill_chunks",
        "audit.gribstream_source_gaps",
        "silver.grib_forecast_values",
    )
    for qualified in required_tables:
        schema, table = qualified.split(".", 1)
        count = connection.execute(
            text(
                """
                SELECT count(*)
                FROM information_schema.tables
                WHERE table_schema = :schema AND table_name = :table
                """
            ),
            {"schema": schema, "table": table},
        ).scalar_one()
        if int(count) != 1:
            result.failures.append(f"missing table {qualified}")

    missing_identity = connection.execute(
        text(
            """
            SELECT count(*)
            FROM silver.grib_forecast_values
            WHERE model_id IS NULL
               OR member IS NULL
               OR grid_point_id IS NULL
               OR forecasted_at_utc IS NULL
               OR forecasted_time_utc IS NULL
               OR variable_alias IS NULL
               OR variable_name IS NULL
               OR source_request_id IS NULL
               OR source_record_id IS NULL
               OR request_sha256 IS NULL
               OR availability_method IS NULL
            """
        )
    ).scalar_one()
    if int(missing_identity) > 0:
        result.failures.append(f"{missing_identity} GribStream rows are missing lineage/identity fields")

    values_count = connection.execute(text("SELECT count(*) FROM silver.grib_forecast_values")).scalar_one()
    availability_count = connection.execute(
        text("SELECT count(*) FROM silver.availability_ledger WHERE source_name = 'gribstream'")
    ).scalar_one()
    chunks_count = connection.execute(text("SELECT count(*) FROM audit.gribstream_backfill_chunks")).scalar_one()
    completed_chunks = connection.execute(
        text(
            """
            SELECT count(*)
            FROM audit.gribstream_backfill_chunks
            WHERE status IN ('completed','completed_empty')
            """
        )
    ).scalar_one()
    if int(values_count) > 0 and int(availability_count) == 0:
        result.failures.append("GribStream values exist but no availability ledger rows exist")

    model_count = connection.execute(
        text("SELECT count(DISTINCT model_id) FROM audit.gribstream_backfill_chunks")
    ).scalar_one()
    if int(chunks_count) > 0 and int(model_count) < len(MODEL_SPECS):
        result.warnings.append(
            f"planner has chunks for {model_count} models; full action plan has {len(MODEL_SPECS)} models"
        )

    result.details.update(
        {
            "cutoff_rows": len(cutoff_rows),
            "cutoff_ids_required": list(cutoff_ids),
            "cutoff_ids_present": sorted(cutoff_rows),
            "grib_forecast_values": int(values_count),
            "grib_availability_rows": int(availability_count),
            "grib_chunks": int(chunks_count),
            "grib_completed_chunks": int(completed_chunks),
            "grib_models_planned": int(model_count),
        }
    )
    return result
