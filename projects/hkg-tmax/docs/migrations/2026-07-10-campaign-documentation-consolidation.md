# Campaign documentation consolidation

Date: 2026-07-10

Scope: `experiments/campaigns/`

Recovery source commit: `b2da67aca12cdc2c2da69a691a18793ef3c35d88`

## Outcome

The campaign archive now has one human entry point, one README per campaign,
and one README per top-level experiment. Detailed run, shard, handoff, and
result evidence remains machine-readable. The migration replaced a recursive
documentation maze with a three-level reading path:

1. `experiments/campaigns/README.md`;
2. the selected campaign `README.md`;
3. the selected experiment `README.md`.

The original working-tree bytes remain recoverable from a checked-in ZIP and
an entry-by-entry provenance ledger.

| Measure | Before | After | Change |
|---|---:|---:|---:|
| Markdown files | 813 | 26 | 787 fewer, 96.8% reduction |
| Markdown bytes | 1,238,411 | 42,039 | 96.6% reduction |
| Text documents in the original census | 8 | 4 retained in place | 4 noisy snapshots retired |
| Canonical human records | 15 existing READMEs | 26 complete READMEs | root, campaign, and experiment coverage |

One 96,940-byte source specification that was previously buried inside a
handoff snapshot was retained as
`residual-modeling/strategy/inputs/hkg_tmax_ml_strategy_codex_implementation_20260705.txt`.
It is an intentional new source copy, not one of the four original retained
text files.

## Canonical layout

The 26 permitted Markdown files are:

- one root campaign index;
- five campaign indexes: HKG Tmax, HKG T-24, probability, residual modeling,
  and market edges;
- twelve HKG Tmax experiment dossiers (`0001` through `0012`);
- two HKG T-24 experiment dossiers (`0214` and `0215`);
- two probability experiment dossiers;
- three residual-modeling experiment dossiers;
- one live market-edge snapshot dossier.

Nested directories may contain JSON, YAML, CSV, Parquet, logs, models, and
other machine evidence. They must not contain additional Markdown reports.
The standard experiment template now expresses hypothesis, as-of contract,
method, results, decision, reproduction, and evidence map as sections of its
single `README.md`.

## Lossless provenance

`experiments/campaigns/DOCUMENT_PROVENANCE.csv` contains all 821 source
documents. Each row records:

- source commit;
- original relative path;
- SHA-256;
- byte count;
- canonical destination README;
- disposition.

The dispositions are:

- `retain_canonical_readme`: 15 original READMEs updated in place;
- `merge_then_prune`: 798 Markdown files merged and retired;
- `retain_non_markdown_source`: 4 original text sources retained;
- `merged_then_pruned_text`: 4 redundant Git-status/command snapshots merged
  and retired.

`experiments/campaigns/DOCUMENT_ARCHIVE_PRE_CONSOLIDATION.zip` contains all
821 original working-tree documents. Archive properties:

- entries: 821;
- bytes: 834,756;
- SHA-256:
  `3f82cf4abd5b3c5983aaf24f2b4fb6b8309f57eb7be38e7779f588200a515e22`.

The ZIP is the byte-exact source because the retired Windows files used CRLF
line endings. Git commit `b2da67a` remains a content fallback, but Git stores
normalized blobs. The project `.gitignore` explicitly allows this one bounded
archive; the repository `.gitattributes` treats ZIP files as binary.

Verify the archive without extracting it:

```powershell
.\.venv\Scripts\python.exe scripts\manage_campaign_documentation.py verify-archive `
  --archive experiments\campaigns\DOCUMENT_ARCHIVE_PRE_CONSOLIDATION.zip `
  --snapshot experiments\campaigns\DOCUMENT_PROVENANCE.csv
```

Recover into a separate directory:

```powershell
Expand-Archive `
  .\experiments\campaigns\DOCUMENT_ARCHIVE_PRE_CONSOLIDATION.zip `
  -DestinationPath C:\temp\hkg-campaign-docs-recovery
```

Do not extract over the live campaign tree.

## Consolidated records

The canonical READMEs preserve the decision-bearing information from the
retired documents: question, cutoff/as-of contract, method, frozen result,
promotion decision, reproduction command, caveats, and machine-evidence map.
Important negative results remain explicit; consolidation does not convert a
failed gate into a success.

The large `0010` and `0012` shard/run trees were the main source of document
fan-out. Their per-shard templates and per-run result pages were reduced to
the top-level dossiers while their JSON/YAML/CSV evidence remained in place.
Historical machine statuses were not rewritten merely to make prose agree;
the dossier explains any campaign-level qualification.

Probability model cards and residual-model result cards were converted to
machine-readable selection summaries or folded into canonical READMEs. The
market-edge snapshot remains analysis-only and does not authorize order
execution.

## Producer and validation changes

The migration also prevents normal reruns from recreating the clutter:

- the public-weather backfill updates bounded generated sections in the
  experiment README, while the shard driver keeps its aggregate report as
  JSON-only machine evidence;
- HKG T-24 `0215` writes one README plus JSON status/configuration, and the
  `0214` deep audit emits JSON;
- experiment `0004` updates a marker-bounded README section;
- probability V1/V2 emit `model_selection_summary.json` instead of result
  Markdown;
- residual-model compatibility copies exclude Markdown and use the canonical
  campaign paths;
- public-weather smoke, benchmark, normalization, decode, tactical, and
  rehearsal defaults now point at the canonical campaign directories;
- the experiment index accepts both current and historical status schemas;
- the template validator requires the consolidated README headings;
- producer regression tests reject nested Markdown creation.

`scripts/manage_campaign_documentation.py` provides four bounded operations:

- `inventory`: enumerate Markdown/text documents without following links;
- `snapshot`: create a hash/size/source-commit ledger;
- `check`: enforce the exact canonical README layout;
- `prune`: dry-run by default and delete only unchanged, snapshotted legacy
  Markdown when `--execute` is explicit;
- `verify-archive`: reject missing, extra, duplicate, unsafe, size-mismatched,
  or hash-mismatched ZIP entries.

The guard rejects path traversal, links/reparse points, unexpected files,
invalid dispositions, changed post-snapshot files, and missing destination
READMEs before deletion.

## Files and ownership

The change is intentionally grouped by responsibility:

- `experiments/campaigns/`: 26 canonical READMEs, provenance CSV, exact ZIP,
  retained machine evidence, and the removals enumerated by the CSV;
- `experiments/templates/standard/`: one-document experiment template;
- `AGENTS.md`, `experiments/AGENTS.md`, `experiments/README.md`, and operations
  docs: durable read order and anti-sprawl rules;
- `scripts/`: canonical output paths and one-document report producers;
- `src/hkg_tmax/experiments.py` and `src/hkg_tmax/validation.py`: generated
  index and template contract;
- `src/hkg_tmax_probability/reporting.py`: machine-readable probability
  summaries;
- `tests/`: archive/layout safety, producer behavior, template/index behavior,
  and focused backfill/T-24 regression coverage;
- `EXPERIMENT_INDEX.md` and `scripts/REGISTRY.csv`: regenerated navigation and
  script metadata.

## Verification record

The migration was checked without database or network execution. Evidence
includes:

- exact verification of all 821 ZIP entries against the provenance CSV;
- layout guard: 26 expected READMEs, zero missing, zero unexpected;
- local Markdown-link scan: 37 local targets, zero broken;
- focused archive guard suite: 12 passed;
- focused producer suite: 23 passed;
- focused public-weather backfill suite: 13 passed;
- focused HKG T-24 producer suite: 11 passed;
- integrated documentation, producer, probability, residual-model, backfill,
  and T-24 suite: 85 passed;
- project fast regression suite: 32 passed;
- project validator and doctor: all executable checks passed, with the two
  pre-existing research gates still reported as warnings (canonical target
  verification and primary-horizon selection);
- targeted Ruff lint, Python compilation, and repository diff checks passed.
- a static producer-policy test covers 21 campaign writers and rejects literal
  non-README Markdown output targets.

## Compatibility and limitations

- Machine artifacts are historical evidence. Absolute paths embedded inside
  them are provenance, not current execution instructions.
- Some immutable reproducibility manifests list retired Markdown artifacts.
  Those bytes remain available in the exact archive; the manifests were not
  rewritten after the fact.
- No historical experiment was rerun, no database was changed, no network
  source was queried, and no model decision was recomputed.
- The consolidation governs `experiments/campaigns/`; reports written to
  external runtime/report roots follow their own contracts.

## Rollback

For a single retired document, read its row in `DOCUMENT_PROVENANCE.csv` and
extract that member from the ZIP into a separate recovery directory. For a
complete historical view, extract the full ZIP. For a repository rollback,
revert the consolidation commit; do not mix an archive extraction into the
live tree with a partial Git revert.
