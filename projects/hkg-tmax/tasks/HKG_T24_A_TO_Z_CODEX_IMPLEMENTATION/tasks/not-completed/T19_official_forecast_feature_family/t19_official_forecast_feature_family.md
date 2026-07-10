# T19 — Official Forecast, Revision, Text, and Trust Feature Family

## Assignment

**Phase:** C Feature Platform  
**Required dependencies:** T14, T18  
**Bookkeeping folder suffix:** `official_forecast_feature_family`

## Mission

Create the complete official-anchor feature family from all eligible forecast vintages and text, without outcomes.

## Why this task exists

The archive provides decades of local operational intelligence and revision behavior.

## Non-negotiable controls for this task

- Target date T is forecast at 15:00 HKT on T−1 under cutoff contract `hkg_t24_1500hkt_v1`, unless T01 formally versions an existing different contract.
- No value enters strict scoring unless availability before cutoff is proven.
- GribStream `asOf` alone is not proof of historical API availability.
- Store UTC as timezone-aware canonical time; derive HKT explicitly.
- Preserve raw data and lineage; clean into normalized tables and quarantine invalid rows.
- Keep 2024+ outcomes sealed unless this task is T36 and the frozen protocol authorizes access.
- Never use target T, same-row residuals, realized error flags, post-cutoff revisions, full-history preprocessing, or in-sample expert predictions.
- Candidate and baseline are compared on identical rows.


## Required inputs and prerequisites

1. Official vintage store
2. target snapshots
3. full source text

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Generate max/min/range/midpoint and issue-age features.
2. Generate first-to-latest revisions, count, velocity, sign, volatility and staleness.
3. Generate source/product/parser era and issue-hour features.
4. Tokenize/normalize weather, wind, RH and rain-probability text; preserve raw.
5. Define fold-local text models/embeddings; no full-history target-informed vocabulary selection.
6. Generate official-versus-causal-target-memory and official-versus-NWP disagreement only when counterpart features exist.
7. Generate missing/null/parser confidence indicators.

## Database/code objects that must exist or be updated

1. feature_store official feature tables/definitions

## Required task-folder artifacts

In addition to the global folder contract, create:

1. feature_definitions_official.csv
2. coverage_profile.csv
3. text_vocabulary_contract.md
4. feature_lineage_samples.csv

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Feature recomputation
2. post-cutoff vintage exclusion
3. no target/residual tokens
4. fold-local NLP tests

## Acceptance criteria

1. All official features have eligibility and lineage
2. No outcome-derived field admitted

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Unparseable text preserved raw and flagged; never discarded silently

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T19",
  "status": "passed|rejected|blocked|partial",
  "git_commit": "...",
  "database_migration_version": "...",
  "input_manifest_sha256": "...",
  "output_manifest_sha256": "...",
  "created_tables_or_views": [],
  "created_files": [],
  "open_blockers": [],
  "downstream_ready": true
}
```

Every path in the handoff must be repository-relative and every listed artifact must exist.
