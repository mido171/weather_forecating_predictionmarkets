"""Feature-family and lineage helpers for the residual-ML pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FeatureLineage:
    feature_name: str
    family: str
    source_table: str
    source_time_column: str
    eligibility_rule: str
    uses_target_label_boolean: bool = False
    minimum_lag_days: int | None = None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


class FeatureRegistry:
    def __init__(self) -> None:
        self._records: dict[str, FeatureLineage] = {}
        self._families: dict[str, list[str]] = {}

    def add(
        self,
        names: list[str],
        *,
        family: str,
        source_table: str,
        source_time_column: str,
        eligibility_rule: str,
        uses_target_label_boolean: bool = False,
        minimum_lag_days: int | None = None,
    ) -> None:
        for name in names:
            self._records[name] = FeatureLineage(
                feature_name=name,
                family=family,
                source_table=source_table,
                source_time_column=source_time_column,
                eligibility_rule=eligibility_rule,
                uses_target_label_boolean=uses_target_label_boolean,
                minimum_lag_days=minimum_lag_days,
            )
            self._families.setdefault(family, [])
            if name not in self._families[family]:
                self._families[family].append(name)

    @property
    def families(self) -> dict[str, list[str]]:
        return {key: list(value) for key, value in self._families.items()}

    def lineage_frame(self) -> pd.DataFrame:
        return pd.DataFrame([record.to_record() for record in self._records.values()])

    def schema_frame(self, matrix: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for record in self._records.values():
            if record.feature_name not in matrix.columns:
                continue
            series = matrix[record.feature_name]
            rows.append(
                {
                    **record.to_record(),
                    "dtype": str(series.dtype),
                    "missing_pct": float(series.isna().mean() * 100.0),
                    "non_null_count": int(series.notna().sum()),
                }
            )
        return pd.DataFrame(rows).sort_values(["family", "feature_name"]).reset_index(drop=True)

