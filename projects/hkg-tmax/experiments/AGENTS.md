# Experiment workspace rules

The parent `AGENTS.md` applies. Before creating or changing an experiment, read
`experiments/README.md`, the relevant governing contract, and prior conclusions
for the campaign.

## Create; never copy or hand-number

Select one allowlisted campaign and allocate the experiment from the project
root:

```powershell
.\.venv\Scripts\python.exe -m hkg_tmax experiments create `
  --campaign hkg-t24 `
  --title "Falsifiable hypothesis title"
```

Valid campaign values are `hkg-tmax`, `hkg-t24`, `residual-modeling`,
`probability`, `market-edges`, and `general`.

The creator owns registry locking, ID allocation, campaign placement, the
campaign README, validated standard scaffold, atomic registry update, and
`EXPERIMENT_INDEX.md` refresh. `--campaign` is mandatory; there is no implicit
dumping-ground destination. Never move the created directory or edit
`registry/registry.yaml` manually. The tracked record is exactly:

```text
campaigns/<campaign>/EXP-####-<slug>/
  README.md
  STATUS.yaml
  DATA_MANIFEST.yaml
  RUN_CONFIG.yaml
  results/metrics.json
```

The creator also owns ignored OS-lock, transaction-journal, and staging state
under `var/`. A new create repairs an interrupted transaction while holding the
lock. Never delete those files to bypass a failure; run the creator again or
report its fail-closed recovery error.

Reusable code belongs in the parent contract's `src/` owner, tests in the
matching `tests/` owner, parameters in governed `config/`, optional orchestration
in a thin `scripts/` entry point, and large outputs under
`${HKG_TMAX_RUN_ROOT}/experiments/<experiment-id>/<run-id>/`.

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
- Finish with `python -m hkg_tmax validate registry`, inspect the automatically
  refreshed `EXPERIMENT_INDEX.md`, and run
  `scripts/manage_campaign_documentation.py check`. A completed experiment is
  immutable; changed hypotheses, frozen splits, or candidate sets require a
  new governed ID.
