# External storage layout

The repository contains code, configuration, contracts, and compact evidence.
It does not own large mutable data.

```text
HKG_TMAX_DATA_ROOT/
  raw/ bronze/ silver/ gold/
  metadata/ manifests/ state/ quarantine/ cache/
  datasets/
  _pipeline_internal/
  imports/repo-20260710/
HKG_TMAX_RUN_ROOT/
  inputs/ ledger/ logs/ reports/ models/ predictions/ experiments/ tmp/
```

Operator-supplied research specifications belong under `inputs/` or at the
file-specific paths configured by `HKG_T24_*_PATH`. A home-directory location
is accepted only as a relocation transition fallback for two historical
roadmaps; new runs should set the external path explicitly.

`HKG_TMAX_DATA_ROOT` is currently expected to be `C:/hkg_tmax_data` on this
workstation. Configure it in the ignored `.env`; never encode credentials there
in tracked files. New artifacts must record a run ID and portable path relative
to the appropriate root. Legacy absolute paths remain provenance only.

`datasets/` and `_pipeline_internal/` are the active compatibility locations for
the preserved research corpus. Their verified source snapshots remain under
`imports/repo-20260710/`; the migration copied rather than moved them. Do not edit
the dated import snapshot or replace either path with a junction.
