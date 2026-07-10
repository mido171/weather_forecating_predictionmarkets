from __future__ import annotations

import argparse
import json
from pathlib import Path

from hkg_tmax.config import find_repo_root
from hkg_tmax.hkg_t24.governance import DEFAULT_DATA_ROOT, write_hkg_t24_governance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HKG T24 governance contracts and reports.")
    parser.add_argument("--root", default=None)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve() if args.root else find_repo_root()
    outputs = write_hkg_t24_governance(root, data_root=Path(args.data_root))
    print(
        json.dumps(
            {
                "station_registry": str(outputs.station_registry_parquet),
                "research_ledger": str(outputs.research_ledger_parquet),
                "source_contracts": str(outputs.source_contracts_yaml),
                "feature_catalog": str(outputs.feature_catalog_yaml),
                "reports": [str(path) for path in outputs.reports],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
