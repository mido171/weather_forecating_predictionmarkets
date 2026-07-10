# HKG Tmax experiment workspace

Experiments are grouped by research campaign. Start at
`campaigns/README.md`; do not recursively browse the evidence tree. Every new
governed experiment is created from `templates/standard`, recorded in
`registry/registry.yaml`, and indexed in the project-level
`EXPERIMENT_INDEX.md`.

- `campaigns/hkg-tmax`: core HKG Tmax data and forecast experiments
- `campaigns/hkg-t24`: H24N/T-24 strategy research
- `campaigns/residual-modeling`: residual and routing model research
- `campaigns/probability`: probability and bucket calibration research
- `campaigns/market-edges`: market-comparison research; never order execution
- `campaigns/general`: genuinely cross-cutting research; never an implicit default
- `campaigns/DOCUMENT_PROVENANCE.csv`: hash and Git-recovery ledger for the
  pre-consolidation documentation set
- `campaigns/DOCUMENT_ARCHIVE_PRE_CONSOLIDATION.zip`: byte-exact compact
  archive of the retired Windows working-tree documents
- `registry`: machine-readable allocation registry
- `templates/standard`: mandatory experiment contract

Create a governed record from the project root with an allowlisted campaign:

```powershell
.\.venv\Scripts\python.exe -m hkg_tmax experiments create `
  --campaign hkg-tmax `
  --title "Falsifiable hypothesis title"
```

The campaign argument is mandatory. The creator validates the existing ledger,
allocates the ID under a crash-released OS lock, creates a campaign README only
when an allowlisted empty campaign is first used, validates the rendered
standard files, atomically updates the registry and index, and rolls back
task-created state on failure. An ignored transaction journal lets the next
create roll back a pre-commit interruption or finish index repair after a
committed registry update. Never copy a prior experiment, hand-number a
directory, move the generated folder, delete transaction state, or edit the
registry manually.

Each campaign and top-level experiment has one human entry point named
`README.md`. That README contains the question, as-of contract, method, result,
decision, limitations, reproduction guidance, and evidence map. Shards, runs,
results, and compatibility copies contain machine evidence only; they do not
carry their own Markdown summaries.

Large outputs, models, raw data, logs, and temporary files belong under the
configured external run/data roots. Git keeps compact README dossiers plus
machine-readable protocols, manifests, metrics, audits, and hashes.
