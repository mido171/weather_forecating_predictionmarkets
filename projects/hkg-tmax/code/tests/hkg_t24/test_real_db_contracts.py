from __future__ import annotations

import os

import pytest
from hkg_t24.db.connection import connect, get_database_url
from hkg_t24.features.source_contracts import run_source_contract_checks


def test_real_db_source_contracts_when_dsn_is_available() -> None:
    if not os.environ.get("HKG_TMAX_DATABASE_URL") and not os.environ.get("HKG_TMAX_DB_DSN"):
        pytest.skip("SKIPPED_REAL_DB_NO_DATABASE_URL")
    database_url = get_database_url()
    with connect(database_url) as connection:
        result = run_source_contract_checks(connection)
    assert result.passed, result.failures
