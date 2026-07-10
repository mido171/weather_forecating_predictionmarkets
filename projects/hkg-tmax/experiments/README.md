# HKG Tmax experiment workspace

Experiments are grouped by research campaign. Start at
`campaigns/README.md`; do not recursively browse the evidence tree. Every new
governed experiment is created from `templates/standard`, recorded in
`registry/registry.yaml`, and indexed in the project-level
`EXPERIMENT_INDEX.md`.

- `campaigns/hkg-tmax`: core HKG Tmax data and forecast experiments
- `campaigns/hkg-t24`: T-24 implementation and acquisition work
- `campaigns/residual-modeling`: residual and routing model research
- `campaigns/probability`: probability and bucket calibration research
- `campaigns/market-edges`: market-comparison research; never order execution
- `campaigns/general`: default destination for newly created experiments
- `campaigns/DOCUMENT_PROVENANCE.csv`: hash and Git-recovery ledger for the
  pre-consolidation documentation set
- `campaigns/DOCUMENT_ARCHIVE_PRE_CONSOLIDATION.zip`: byte-exact compact
  archive of the retired Windows working-tree documents
- `registry`: machine-readable allocation registry
- `templates/standard`: mandatory experiment contract

Each campaign and top-level experiment has one human entry point named
`README.md`. That README contains the question, as-of contract, method, result,
decision, limitations, reproduction guidance, and evidence map. Shards, runs,
results, and compatibility copies contain machine evidence only; they do not
carry their own Markdown summaries.

Large outputs, models, raw data, logs, and temporary files belong under the
configured external run/data roots. Git keeps compact README dossiers plus
machine-readable protocols, manifests, metrics, audits, and hashes.
