from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import Connection

from klga_tmax.utils.git import current_git_sha

AUDIT_BOOTSTRAP_SQL = """
CREATE SCHEMA IF NOT EXISTS audit;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE TABLE IF NOT EXISTS audit.pipeline_runs (
    pipeline_run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    command_name text NOT NULL,
    command_args jsonb NOT NULL DEFAULT '{}'::jsonb,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    status text NOT NULL,
    exit_code integer,
    source_code_git_sha text NOT NULL,
    row_counts jsonb NOT NULL DEFAULT '{}'::jsonb,
    error_message text,
    log_uri text,
    CONSTRAINT ck_pipeline_status CHECK (
        status IN ('started','success','failed','skipped')
    )
);
"""


def ensure_audit_table(connection: Connection) -> None:
    connection.execute(text(AUDIT_BOOTSTRAP_SQL))


def start_pipeline_run(
    connection: Connection,
    *,
    command_name: str,
    command_args: dict[str, Any] | None = None,
) -> UUID:
    ensure_audit_table(connection)
    row = connection.execute(
        text(
            """
            INSERT INTO audit.pipeline_runs (
                command_name,
                command_args,
                status,
                source_code_git_sha
            )
            VALUES (
                :command_name,
                CAST(:command_args_json AS jsonb),
                'started',
                :source_code_git_sha
            )
            RETURNING pipeline_run_id
            """
        ),
        {
            "command_name": command_name,
            "command_args_json": json.dumps(command_args or {}, sort_keys=True),
            "source_code_git_sha": current_git_sha(),
        },
    ).one()
    return row.pipeline_run_id


def finish_pipeline_run(
    connection: Connection,
    *,
    pipeline_run_id: UUID,
    status: str,
    exit_code: int,
    row_counts: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    connection.execute(
        text(
            """
            UPDATE audit.pipeline_runs
            SET
                finished_at = now(),
                status = :status,
                exit_code = :exit_code,
                row_counts = CAST(:row_counts_json AS jsonb),
                error_message = :error_message
            WHERE pipeline_run_id = :pipeline_run_id
            """
        ),
        {
            "pipeline_run_id": pipeline_run_id,
            "status": status,
            "exit_code": exit_code,
            "row_counts_json": json.dumps(row_counts or {}, sort_keys=True),
            "error_message": error_message,
        },
    )
