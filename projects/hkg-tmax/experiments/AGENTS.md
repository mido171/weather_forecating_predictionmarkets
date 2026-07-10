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
