from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping
import pandas as pd


def write_parquet_rows(path: str | Path, rows: Iterable[Mapping[str, object]]) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_parquet(out, index=False)
    return str(out)
