from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from klga_tmax.constants import (
    FEATURE_SET_NAME,
    FEATURE_VERSION,
    FORMULA_CONTRACT_HASH,
)
from klga_tmax.registry.seed_cutoffs import seed_cutoffs
from klga_tmax.registry.seed_stations import seed_station_registry, seed_stations
from klga_tmax.utils.git import current_git_sha


def seed_feature_version(connection: Connection) -> int:
    result = connection.execute(
        text(
            """
            INSERT INTO registry.feature_versions (
                feature_set_name,
                feature_version,
                source_code_git_sha,
                formula_contract_hash,
                feature_names
            )
            VALUES (
                :feature_set_name,
                :feature_version,
                :source_code_git_sha,
                :formula_contract_hash,
                :feature_names
            )
            ON CONFLICT (feature_set_name, feature_version) DO UPDATE SET
                source_code_git_sha = EXCLUDED.source_code_git_sha,
                formula_contract_hash = EXCLUDED.formula_contract_hash,
                feature_names = EXCLUDED.feature_names
            """
        ),
        {
            "feature_set_name": FEATURE_SET_NAME,
            "feature_version": FEATURE_VERSION,
            "source_code_git_sha": current_git_sha(),
            "formula_contract_hash": FORMULA_CONTRACT_HASH,
            "feature_names": [],
        },
    )
    return result.rowcount or 0


def seed_all(connection: Connection) -> dict[str, int]:
    return {
        "registry.cutoffs": seed_cutoffs(connection),
        "registry.station_registry": seed_station_registry(connection),
        "registry.stations": seed_stations(connection),
        "registry.feature_versions": seed_feature_version(connection),
    }
