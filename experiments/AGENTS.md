# Experiment Directory Rules

These instructions apply to every child experiment.

- Never edit another experiment to accommodate a new result.
- Complete hypothesis/protocol/as-of/data/run config before holdout inspection.
- Keep source manifests and hashes.
- Store row-level predictions.
- Record every attempted specification in the declared family.
- Report failures and nulls.
- `ACCEPTED` requires independent leakage and reproducibility PASS.
- A result may enter `MILESTONES.md` only after all gates pass.
- Large files may be stored externally, but the manifest must include immutable URI/path, checksum, format, rows, and schema.
- No notebook-only accepted result.
