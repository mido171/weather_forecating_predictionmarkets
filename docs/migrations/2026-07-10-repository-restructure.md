# Repository Restructure — 2026-07-10

Status: implementation and pre-cutover verification complete; canonical-path cutover pending

This is the canonical engineering record for consolidating the weather-data workspace into
one understandable, standalone repository. It records the source state, decisions, path
accounting, safety work, storage migration, verification, and rollback mechanism. The exact
per-file inventory is generated at
`docs/migrations/2026-07-10-changed-files.csv`; the compact semantic path map is
`docs/migrations/2026-07-10-path-map.csv`.

## Executive Summary

The source workspace had three overlapping ownership models: an outer linked Git worktree,
an independently versioned HKG repository nested several bootstrap levels below it, and a
KLGA project whose active implementation was wrapped in an additional `implementation`
directory. Source, specifications, experiments, generated outputs, bulk data, credentials,
and live demo processes were mixed into those trees.

The target is a single standalone Git repository with deployable applications under `apps`,
shared code under `packages`, city research products under `projects`, bounded operator tools
under `tools`, canonical documentation under `docs`, and non-authoritative retained code under
`legacy`. HKG is now `projects/hkg-tmax`; KLGA is `projects/klga-tmax`. Bulk HKG state lives
under `C:\hkg_tmax_data`; bulk KLGA state lives under `C:\klga_tmax_data`. No junction is used
to disguise external data as repository content.

This migration intentionally does not delete the original worktree, destructively mirror a
data directory, rewrite Git history, place an order, contact a weather provider, mutate a
production database, or install a scheduled task.

## Reader Orientation

Start with root `AGENTS.md`, then `docs/START_HERE.md`, the repository map, the safe-command
guide, and the closest scoped instructions. The semantic move map is
`docs/migrations/2026-07-10-path-map.csv`. The generated
`docs/migrations/2026-07-10-changed-files.csv` is the complete file-level accounting record;
it exists so this narrative can explain behavior and ownership without repeating thousands
of mechanical move rows. HKG details are owned by `projects/hkg-tmax/START_HERE.md`; KLGA
details are owned by `projects/klga-tmax/START_HERE.md`.

## Scope Boundaries

The following invariants governed every phase:

- Preserve user source, tests, schemas, configuration, documentation, experiment conclusions,
  and dirty working states before changing topology.
- Preserve both outer and HKG Git lineages in verified bundles and in the consolidated commit
  graph.
- Copy data first, reconcile it, switch configuration, observe, and only consider deletion in
  a separate future retention decision.
- Treat generated artifacts, caches, logs, model output, and credentials as external state.
- Keep provider collection, backfills, schedulers, authentication, trading, reconciliation,
  and production mutation disabled unless a human explicitly enables them.
- Use one worker by default, cap ordinary agent concurrency at two, and cap numerical library
  threads at one during verification.
- Stop only process trees whose executable, command line, parentage, path, and listening port
  were re-queried immediately before the action.
- Make the filesystem cutover reversible: archive and lock the old registered worktree, then
  promote the already-tested standalone repository into the canonical path.

## Source-of-Truth Inputs

The original canonical path was a linked worktree at
`C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction`. Its Git
administrative entry was a `.git` file owned by the sibling primary repository, its branch was
`extraction-cleanup`, and its captured head was `adf7e3b`.

The active HKG mini-project was an independent nested repository at
`bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex`. Its captured branch was
`master` and its head was `206089f`. The migration first committed the exact tracked dirty
states to preservation commits `e512a4d` for the outer tree and `3b04d8f` for HKG. It then
merged current canonical remote ancestry and imported both preserved lineages before any
structural move. The key topology commits are:

| Commit | Purpose |
|---|---|
| `e512a4d` | Preserve outer pre-consolidation tracked state |
| `3b04d8f` | Preserve HKG pre-consolidation tracked state |
| `7d52768` | Merge current canonical remote master ancestry |
| `c0c91ea` | Import preserved HKG history under `projects/hkg-tmax` |
| `c22e7d2` | Import KLGA source and strategy corpus |
| `fdbd56e` | Establish the canonical monorepo component layout |

The staging branch is `migration/restructure-20260710`. Its `origin` is
`https://github.com/mido171/weather_forecating_predictionmarkets.git`. Remote state is fetched
and rechecked immediately before publication; no force push is permitted.

## 4. Recovery artifacts

The restricted recovery directory is:

`C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\.migration-recovery\weather_data_extraction_20260710T021137`

It contains complete outer and HKG Git bundles, tracked working-tree patches, index patches,
captured refs/status/config evidence, worktree registration evidence, storage copy/verification
logs, and representative source/destination hashes. `git bundle verify` passed for both
bundles. The principal immutable artifact hashes recorded before migration are:

| Artifact | Bytes | SHA-256 prefix |
|---|---:|---|
| `outer/outer-all.bundle` | 112,799,176 | `BA2A1D9AED97F235` |
| `outer/outer-working.patch` | 32,028 | `944BF4FDC73760CE` |
| `hkg/hkg-all.bundle` | 113,470,707 | `28207BB9297E1417` |
| `hkg/hkg-working.patch` | 2,325,692 | `41BF304C11F6FBDA` |

`SHA256SUMS.csv` is the machine-readable manifest. Recovery ACLs were restricted because the
artifacts preserve history and local operational context. Secret values are never copied into
this ledger or console evidence.

## Requirements-to-Implementation Traceability

| Requirement | Implementation evidence |
|---|---|
| One structured root | Standalone `.git`; explicit `apps`, `packages`, `projects`, `tools`, `docs`, `legacy`, and `var` owners |
| Logical grouping | `docs/architecture/REPOSITORY_MAP.md`, scoped `AGENTS.md`, HKG/KLGA wrapper removal, incomplete-tool quarantine |
| Documentation, code, and experiments | Root/project indexes, installable `src` packages, governed `experiments/{campaigns,registry,templates}` |
| Fast, safe agent startup | Root constitution, deterministic read order, task classification, source authority, stop conditions, two-thread cap |
| CPU/Git/runtime protection | One-worker defaults, numerical thread caps, disabled collectors/trading, `--execute`, process ownership, explicit-path staging |
| Preserve history and user work | Two verified bundles, two dirty-state commits, ancestry merge, locked rollback worktree plan |
| Remove bulk state from Git | External HKG/KLGA roots, dated imports, active dataset promotion, ignored artifacts/logs, no junctions |
| Prove the result | Strict doctor, Java/HKG/KLGA suites, smoke/build checks, hashes, data reconciliation, remote equality, reversible cutover |

## Change Inventory

The final staged change set contains 4,420 name-status records after rename detection with
`diff.renameLimit=10000`. The generated CSV records status, old path, new path, owning area,
and purpose for every record. The compact path map describes ownership-level moves and
dispositions. Cached review found 0 unstaged paths, 0 untracked paths, 0 staged files over
10 MiB, and only the two non-secret `.env.example` files matching the environment filename
pattern. The complete pre-cutover staged name-status output is retained with the restricted
recovery evidence.

## Architecture and Control Flow

The durable ownership model is:

```text
weather_data_extraction/
├── apps/                 deployable ingestion and market services
├── packages/             shared Java and Python libraries
├── projects/             HKG and KLGA research products
├── tools/                bounded backfill, discovery, live, and repo utilities
├── tests/smoke/          cross-component offline smoke checks
├── config/examples/      non-secret examples
├── docs/                 repository architecture, operations, security, migrations
├── legacy/               retained non-authoritative implementation history
└── var/                  ignored lightweight local runtime state
```

`docs/architecture/REPOSITORY_MAP.md` is the authority for ownership. Root `AGENTS.md` defines
startup, task classification, CPU, network, process, Git, storage, secret, verification, and
stop rules. Scoped `AGENTS.md` files exist for both applications, both research projects,
experiment areas, live/backfill/discovery tools, and legacy code. Root `.codex/config.toml`
caps agent threads at two. This structure makes a new conversation read a short deterministic
chain instead of recursively consuming the historical corpus.

## 6. HKG normalization

The HKG project is a first-class product rather than a bootstrap-inside-bootstrap checkout.
Its Python packages and tests moved from `code/src` and `code/tests` to `src` and `tests`.
Configuration is grouped under `config/{acquisition,experiments,project,sources}`; database
assets under `db/{migrations,schemas,sql}`; work packages under `planning`; and experiments
under `experiments/{campaigns,registry,templates}`. Canonical documentation is organized by
architecture, contracts, data, decisions, operations, research, security, specifications,
status, and templates. Historical reports and analysis remain evidence, not startup authority.

The path layer in `src/hkg_tmax/paths.py` centralizes project, data, run, configuration, and
database roots. It preserves relocation-aware legacy provenance without hard-coding a user
directory. Experiment indexing uses a bounded top-level status scan rather than recursive
generated-tree traversal. `scripts/REGISTRY.csv` inventories 179 top-level scripts by category,
risk, lifecycle, execute guard, main guard, and size. Its lifecycle census identifies 26
retained reproduction runners, nine active operator scripts, and one active research runner.
All 36 scripts that previously constructed mutable repo-local roots now use `ProjectPaths`;
an AST regression test prevents those path anchors from returning.

All 35 collector sources are disabled by default. Collector execution requires both policy
enablement and an explicit environment acknowledgement. Acquisition and backfill entrypoints
require `--execute`, finite date/request/source limits, bounded retries, and one worker by
default. The Windows collector installer is dry-run by default, does not force replacement,
and registers a disabled task. Docker uses an allowlisted context, a non-root user, loopback
database defaults, manual profiles, and CPU/memory caps.

## 7. KLGA normalization

KLGA no longer hides active code beneath an `implementation` wrapper. Its package, tests,
configuration, Alembic migrations, scripts, experiments, and `pyproject.toml` live directly
under `projects/klga-tmax`. Strategy specifications and implementation context are routed
under `docs`; current status and start documents distinguish active authority from historical
planning evidence.

Network-facing CLI commands now fail closed behind `--execute`. Dates, stations, and models
must be explicit; broad `all` and `latest` scopes are rejected. Defaults are one request,
chunk, model, and worker; workers are capped at two. Wunderground windows default to at most
31 days unless an explicit long-run acknowledgement and finite budget are present.
GribStream attempts are spaced by at least 12 seconds and bounded. Polymarket attempts and
resume-gap worst-case requests have hard budgets. Java truth resolution points to the
canonical `apps/ingestion-service` module and uses a bounded Maven timeout.

## 8. Application and runtime hardening

The ingestion service defaults to loopback, local H2, disabled jobs/backfills, one worker, and
bounded retries. Administrative controllers require explicit enablement and an
`X-Local-Control-Token`. Provider clients reject calls without credentials, enforce request
budgets where applicable, and redact keys from errors. Generated training/runtime resources
were removed from source control and route beneath ignored runtime roots.

The Kalshi service defaults to DEMO with authentication, trading, WebSockets, live inference,
smoke execution, backtests, and reconciliation disabled. Production requires an explicit
acknowledgement. Trading requires authentication and guardrails. Sensitive controllers require
explicit enablement and a local control token. CORS is loopback-only. Startup reconciliation
failure creates a global fail-closed buy halt.

One previously tracked proxy credential was removed from the current tree and replaced by
environment-based configuration. Because the value remains in pre-existing Git history,
provider-side rotation remains mandatory. History purge is deliberately excluded from this
migration because it would rewrite shared history and require separate authorization and
coordination.

The shared live Python code is now a conventional installable package at
`packages/python/weather-live/src/ml_live`; both the canonical and compatibility import styles
are verified. Root operator scripts whose required source modules never existed in tracked
history were not cosmetically patched. They are retained under `legacy/incomplete-live-tools`
or `legacy/incomplete-backfill-tools` with the exact missing dependency documented. The active
`tools/live` promotion boundary is intentionally empty until a complete, tested runner exists.

## File-by-File Deep Dive

### `AGENTS.md`

Defines startup, authority, artifact routing, resource limits, prohibited scans, external-action
budgets, process ownership, Git recovery/staging, storage, credentials, verification, and stop
conditions. Scoped constitutions may tighten but cannot weaken it.

### `tools/repo/doctor.py`

Implements the read-only local/CI policy gate. It checks topology, tracked runtime state, large
files, credentials, stale paths, unsafe defaults, and fail-closed startup behavior.
`tools/repo/tests/test_doctor.py` covers policy detections, monorepo Git discovery, and Windows
symlink privilege handling.

### `tools/repo/build_migration_inventory.py`

Builds the exact CSV from Git evidence. It disables filesystem monitoring and automatic
line-ending rewriting for the probe and raises rename detection to 10,000 so the HKG moves are
not misrepresented as a delete/add explosion.

### `projects/hkg-tmax/src/hkg_tmax/paths.py`

Owns project, data, run, configuration, database, and operator-input roots. It resolves archive
references after relocation, loads only path-related values from the ignored `.env`, and never
imports provider/database credentials as a side effect. Path and AST tests cover relocation,
containment, legacy provenance, and the ban on repo-local mutable roots.

### `projects/klga-tmax/src/klga_tmax/cli.py`

Centralizes external-effect gates. Provider/database commands require `--execute`, explicit
scope, finite budgets, bounded retries, one worker by default, and provider-specific spacing.
Dry-run tests prove zero provider/database calls.

### `packages/python/weather-live/pyproject.toml`

Makes `ml_live` a conventional installable `src` package. Canonical and compatibility imports
build in isolation; runtime output defaults to ignored `var/weather-live` or an explicit root.

### `apps/ingestion-service/src/test/java/com/predictionmarkets/weather/RuntimeSafetyDefaultsTest.java`

Locks ingestion defaults to local/inert behavior. Companion security tests prove administrative
endpoints require explicit enablement and a local control token.

### `docs/migrations/2026-07-10-changed-files.csv`

Provides complete file-level traceability. It is generated from Git, not hand-edited, and is
regenerated whenever the final staged state changes.

## Public Interfaces and Contracts

The human/agent interface is the root and scoped `AGENTS.md` chain plus the start documents.
HKG exposes `hkg-t24`, `hkg-tmax`, `hkg-tmax-db`, and `hkg-tmax-demo-trading`; path settings use
`HKG_TMAX_DATA_ROOT`, `HKG_TMAX_RUN_ROOT`, and `HKG_TMAX_INPUT_ROOT`. KLGA exposes `klga-tmax`
and uses `KLGA_DB_URL` plus external artifact/run roots. Shared Python consumers install the
`weather-live` distribution and import `ml_live`. Java applications bind to loopback/inert
profiles by default. Sensitive controllers require explicit properties and local tokens.
Network, backfill, and trading CLIs use `--execute` as the visible acknowledgement boundary.

## 9. External data migration

Repository-local generated state was copied, never moved or mirrored, into dated preservation
imports. HKG uses `C:\hkg_tmax_data\imports\repo-20260710`; KLGA uses
`C:\klga_tmax_data\imports\repo-20260710`.

| Area | Files | Size | Copy/verification result |
|---|---:|---:|---|
| HKG pipeline internal | 2,527 | 321.37 MiB | 0 mismatch, 0 failure |
| HKG datasets | 558,459 | 4.737 GiB | 0 mismatch, 0 failure |
| HKG experiments | 3,543 | 1.690 GiB | 0 mismatch, 0 failure |
| HKG deliveries | 469 | 1.230 GiB | 0 mismatch, 0 failure |
| HKG reports | 177 | 16.98 MiB | 0 mismatch, 0 failure |
| HKG data analysis | 9 | 20.32 MiB | 0 mismatch, 0 failure |
| HKG temporary state | 216 | 1.56 MiB | 0 mismatch, 0 failure |
| KLGA artifacts | 4,658 | 764.68 MiB | 0 mismatch, 0 failure |
| KLGA run logs | 24 | 2.47 MiB | 0 mismatch, 0 failure |

Thirty-four representative SHA-256 source/destination pairs matched. Static `/L` reconciliation
reported zero mismatches and failures. One HKG log was expectedly newer after the initial copy
because an owned demo process was still writing; that log directory is recopied and reconciled
after the exact process tree is stopped at cutover. Free space after the large copy was
211.50 GiB, above the declared 180 GiB abort threshold.

The dated imports remain immutable preservation snapshots. To make the configured
`C:\hkg_tmax_data` root immediately usable without a junction, the verified dataset and hidden
pipeline snapshots were copied a second time into active `datasets/` and `_pipeline_internal/`
locations. The active dataset copy reconciled 558,459 files and 4.737 GiB with zero delta; the
pipeline copy reconciled 2,527 files and 321.37 MiB with zero delta. Free space after promotion
was 210.63 GiB. No import snapshot or source directory was removed.

## Testing and Verification Evidence

Verification is serial and offline unless explicitly stated. The current pre-cutover results
are:

| Gate | Result |
|---|---|
| Root repository doctor, strict | 0 errors, 0 warnings |
| Repository doctor unit tests | 13 passed, 1 Windows privilege skip |
| Root Maven reactor | BUILD SUCCESS; shared 6, ingestion 49, Kalshi 23 tests |
| Root offline extractor smoke | Passed |
| HKG full offline suite | 742 passed, 14 external-artifact skips |
| HKG script-path regression | 18 path tests and 60 affected-script tests passed; 1 optional skip |
| KLGA full offline suite | 126 passed |
| KLGA CLI/script safety | 28 focused tests passed; dry runs made 0 provider/DB calls |
| HKG configuration validation | 14 checks passed; 61 sources, 21 pair families, 11 gridded families, 11 buckets |
| Shared `weather-live` package | Isolated editable build/import and compatibility imports passed |
| `git diff --check` | Passed |
| External sample hashes | 34 of 34 matched |
| Static data reconciliation | 0 mismatch, 0 failure except the documented pre-cutover live-log delta |

The HKG validation warnings that target canonical status is not yet verified and a primary
horizon is not yet selected are research-state facts, not migration failures. No provider
request, live order, scheduler installation, production database mutation, or production
environment activation was used as a test.

Representative root commands were:

```powershell
python tools/repo/doctor.py --strict
python -m pytest -q -p no:cacheprovider tools/repo/tests
mvn -B -T 1 test
python tests/smoke/smoke_extractors.py
```

HKG used the staging source, cleared provider/database variables, disabled collectors, one
numerical thread, and these serial gates:

```powershell
python -m pytest -o addopts='' -q -p no:cacheprovider tests
python -m hkg_tmax validate all
python -m hkg_tmax doctor
```

KLGA used an explicit staging `PYTHONPATH`, one-thread limits, disabled bytecode/cache output,
the 125-test serial suite, compile, Ruff, and dry-run CLI probes. Storage proof used
non-mirror Robocopy followed by `/L` reconciliation and 34 representative SHA-256 pairs. Git
proof used `git diff --cached --check`, staged path/status accounting, ancestry and
connectivity checks, plus current-tree credential and large-file scans.

A 4.10 MiB JSON file under the governed HKG implementation evidence pack is intentionally
retained: it is a compact, immutable attribute-value profile used by the planning contract,
not runtime output. No other HKG Git-visible file exceeds 1 MiB, and no repository file exceeds
the 10 MiB release threshold. After tests, 61 exact cache/build roots containing 1,496 files
and 65.55 MiB were removed with path containment and reparse-point guards.

## 11. Git staging, publication, and cutover

The migration never uses `git add .` or `git add -A`. The exact diff and untracked set are
generated into the changed-file inventory, reviewed by area/status, then staged through
explicit pathspecs. Before each commit, cached name-status, statistics, whitespace, ignored
secrets, large files, and nested Git metadata are checked. Immediately before push, the Git
root, branch, remote, upstream, status, and fetched remote heads are revalidated. Publication
uses a normal branch push; force push is prohibited.

The filesystem cutover sequence is:

1. Re-query ports 6000 and 6001 and the full owned parent/child process chains.
2. Stop only the verified old HKG demo children first; confirm both ports are free.
3. Recopy and statically reconcile the formerly live HKG logs.
4. Use the primary repository's `git worktree move` to archive the old linked worktree.
5. Rename the tested standalone staging directory to the canonical
   `weather_data_extraction` path.
6. Confirm `.git` is a directory, exactly one Git boundary exists, and root/branch/remote are
   correct.
7. Lock the archived linked worktree with an explicit migration reason.
8. Run post-cutover doctor, tests, import/help smoke checks, and storage/path verification.

If promotion fails before step 6, the staging rename is reversed and `git worktree move`
restores the old canonical path. There is no recursive deletion in this sequence.

## 12. Rollback and retention

The original linked worktree is retained in
`.migration-archive\weather_data_extraction-pre-restructure-20260710` and locked after a
successful cutover. The independent HKG state remains recoverable from its complete bundle,
working patch, preservation commit, and archived old worktree. External import snapshots are
retained; source data is not deleted by this migration.

To roll back during the observation period: stop only a verified new owned process tree,
rename the new canonical repository back to its staging name, unlock the archived worktree,
move it to the canonical path through the primary repository, and rerun the original health
checks. Bundle restoration is a secondary disaster-recovery path, not the first rollback.

## Known Limitations and Follow-Up Work

Provider rotation of the historically exposed proxy credential is the only external action
that this repository migration cannot perform. Optional Git history purging must be planned
separately because it affects every clone and branch. Old worktree and import-snapshot deletion
is also a separate, explicit retention decision after an observation period.

The migration is complete when all of the following are recorded: canonical root promotion;
one `.git` directory and no nested Git/reparse point; locked rollback worktree; final live-log
reconciliation; strict doctor and project gates; changed-file inventory regeneration; normal
remote push; and a clean final worktree. The status line and this section are updated only
after those facts are observed.

Full-repository HKG Ruff and strict mypy remain inherited debt rather than migration gates:
the pre-cutover baseline audit recorded 195 Ruff diagnostics across 86 files and 117 mypy
errors across 41 files. Changed path/governance modules and affected scripts passed scoped
static checks. Four HKG roadmap tests skip when external generated run artifacts have not been
materialized; generated payloads are no longer required inside Git. The two HKG scientific
research gates remain explicitly open. Provider rotation and any coordinated history purge
remain external follow-up work.

## Reviewer Checklist

- [x] Recovery bundles verify and dirty states are represented by preservation commits.
- [x] Outer and HKG source commits are ancestors of the consolidated branch.
- [x] Every staged path is represented by the generated changed-file inventory.
- [x] Current tracked tree has no credential, nested Git root, reparse point, or file over 10 MiB.
- [x] Root doctor, Java, HKG, KLGA, and cross-component smoke gates pass offline.
- [x] HKG active datasets and pipeline-internal data reconcile to immutable import snapshots.
- [ ] Pre-cutover branch push equals the remote branch head.
- [ ] Original owned demo processes are stopped and final live logs reconcile.
- [ ] Canonical path contains the standalone repository and the rollback worktree is locked.
- [ ] Fresh canonical HKG/KLGA environments resolve imports beneath the new root.
- [ ] Post-cutover doctor, health smoke, final documentation commit, and remote equality pass.
