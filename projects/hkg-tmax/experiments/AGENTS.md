# Experiment workspace rules

The parent `AGENTS.md` applies. Before creating or changing an experiment, read
`experiments/README.md`, the relevant governing contract, and prior conclusions
for the campaign.

- Predeclare hypothesis, split, baseline, metrics, leakage checks, resource
  budget, and stop conditions before viewing holdout outcomes.
- Use the standard template and registry for new governed IDs.
- Never overwrite completed results or tune repeatedly on a locked test.
- Keep only compact protocols, manifests, metrics, and conclusions in Git.
- Put raw payloads, Parquet, models, predictions, plots, and logs under the
  external run/data roots and reference them by manifest/hash.
- Record rejected, null, inconclusive, and blocked results.
- Historical campaign folders preserve their original naming and provenance;
  do not bulk-normalize their internal evidence paths.
- Keep human narrative for each campaign and top-level experiment in its
  canonical `README.md`. Fold hypothesis, as-of rules, protocol, results,
  conclusion, limitations, and reproduction guidance into that file instead
  of creating sibling or run-level Markdown files.
- Keep machine evidence (`STATUS.yaml`, manifests, metrics, audits, tables,
  and hashes) separate from the README. A frozen source input may remain as a
  non-Markdown artifact when its exact bytes matter.
- Experiment runners that refresh human documentation must update a bounded,
  idempotent section of the canonical README. They must not create Markdown in
  shard, run, result, handoff, or compatibility-copy directories.
- `campaigns/DOCUMENT_PROVENANCE.csv` is the recovery ledger for documentation
  retired by the 2026-07-10 consolidation. Do not edit historical hashes or
  claim an old path is current. Use the adjacent pre-consolidation ZIP for
  byte-exact recovery; the recorded Git commit is the normalized fallback.
