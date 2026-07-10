# Data Folder

Use `datasets/` for normal browsing.

The old pipeline folders are now hidden compatibility junctions that point to
`_pipeline_internal/`. They remain in place so existing scripts, tests, and
reports can keep using paths like `data/silver/...` and `data/gold/...`.

## Human-Facing Layout

- `datasets/` - clean dataset-by-dataset folders.
- `datasets/MANIFEST.csv` - every organized file, source path, size, and link/copy method.

## Internal Layout

- `_pipeline_internal/` - hidden physical storage for pipeline-owned folders.
- hidden junctions: `raw/`, `bronze/`, `silver/`, `gold/`, `metadata/`, `logs/`, `state/`, `cache/`, and `quarantine/`.

Do not delete the hidden junctions unless the code has first been migrated away
from the old internal paths.
