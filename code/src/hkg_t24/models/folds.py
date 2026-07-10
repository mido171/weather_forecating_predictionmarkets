"""Chronological fold helpers for genuine OOF predictions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    train_start_date: date
    train_end_date: date
    test_start_date: date
    test_end_date: date

    def validate(self) -> None:
        if self.train_end_date >= self.test_start_date:
            raise ValueError(
                f"OOF fold {self.fold_id} violates chronology: "
                f"train_end_date={self.train_end_date}, test_start_date={self.test_start_date}"
            )
        if self.train_start_date > self.train_end_date:
            raise ValueError(f"OOF fold {self.fold_id} has empty train range")
        if self.test_start_date > self.test_end_date:
            raise ValueError(f"OOF fold {self.fold_id} has empty test range")


def validate_folds(folds: list[FoldSpec]) -> None:
    if not folds:
        raise ValueError("At least one fold is required")
    for fold in folds:
        fold.validate()


def default_pre2024_folds() -> list[FoldSpec]:
    """Return conservative expanding-window pre-2024 development folds."""
    folds = [
        FoldSpec("pre2024_y2018", date(2005, 1, 1), date(2017, 12, 31), date(2018, 1, 1), date(2018, 12, 31)),
        FoldSpec("pre2024_y2019", date(2005, 1, 1), date(2018, 12, 31), date(2019, 1, 1), date(2019, 12, 31)),
        FoldSpec("pre2024_y2020", date(2005, 1, 1), date(2019, 12, 31), date(2020, 1, 1), date(2020, 12, 31)),
        FoldSpec("pre2024_y2021", date(2005, 1, 1), date(2020, 12, 31), date(2021, 1, 1), date(2021, 12, 31)),
        FoldSpec("pre2024_y2022", date(2005, 1, 1), date(2021, 12, 31), date(2022, 1, 1), date(2022, 12, 31)),
        FoldSpec("pre2024_y2023", date(2005, 1, 1), date(2022, 12, 31), date(2023, 1, 1), date(2023, 12, 31)),
    ]
    validate_folds(folds)
    return folds
