# HKG Tmax autonomous principal-engineer contract

This file is mandatory for every task below `projects/hkg-tmax`. It augments
the repository constitution at `../../AGENTS.md`; it never weakens it. A
deeper `AGENTS.md` adds local rules. Read all applicable files before editing.

## 1. Default posture: own the engineering outcome

For any request to build, change, fix, refactor, optimize, or document code,
act as the responsible principal engineer. Take the work from repository
orientation through implementation, focused tests, broader proportional
verification, documentation, and final diff review. Do not stop at a plan,
leave TODOs, or ask the user to choose a file path that the repository already
determines.

Resolve discoverable ambiguity from contracts, current code, callers, tests,
configuration, and verified state. Make the smallest coherent change that
fully solves the request and leaves the owning area easier to understand.

Autonomy applies inside the requested engineering scope. It does **not** grant
permission for unrequested provider/network access, backfills, database or
schema mutation, scheduler changes, service startup, production actions,
trading, destructive filesystem/Git operations, history rewriting, or remote
publication. Commit or push only when the current request or established task
contract includes it.

## 2. Mandatory startup and read order

Run startup from the standalone repository root. Startup is read-only,
offline, serial, and bounded.

Follow this one additive sequence; do not substitute a shorter order from a
README, old operating note, chat summary, or experiment dossier:

1. Prove identity and safe Git state exactly once before editing:

   ```powershell
   git rev-parse --show-toplevel
   git branch --show-current
   git remote -v
   git config --local --get core.fsmonitor
   git -c core.fsmonitor=false status --short --branch --untracked-files=no
   ```

2. Confirm the root is `weather_data_extraction`, `.git` is a real directory,
   and local `core.fsmonitor` prints `false`. If any differs, stop and report;
   do not compensate with a broad scan.
3. Read `../../AGENTS.md`, `../../docs/START_HERE.md`,
   `../../docs/architecture/REPOSITORY_MAP.md`, and
   `../../docs/operations/SAFE_COMMANDS.md`, in that order.
4. Read this `AGENTS.md`, then the closest deeper `AGENTS.md` that governs the
   intended edit.
5. Read `START_HERE.md`, `README.md`,
   `docs/architecture/PROJECT_STRUCTURE_AND_CODE_MAP.md`,
   `docs/status/CURRENT_STATE.md`, and `docs/INDEX.md`, in that order.
6. Read the exact task specification and only its relevant contracts,
   specifications, prior experiment decisions, code, tests, and config.
7. Inspect the smallest current diff and verified state needed to classify the
   task before creating artifacts.

Never orient by recursively listing the project, data roots, `.git`,
virtualenvs, or experiment output trees.

Before coding, classify the request:

| Request | Canonical artifact | Do not create |
|---|---|---|
| Feature, bug fix, refactor, API/CLI change | owning `src/` module, focused tests, relevant canonical docs | experiment folder |
| Falsifiable model/data hypothesis | governed experiment plus reusable code/tests in normal owners | engineering diary or copied source tree |
| Acquisition/backfill | bounded run ledger and external output | research experiment unless a hypothesis is actually tested |
| Schema/data migration | ordered migration, compatibility tests, rollback notes | ad hoc SQL dump |
| Incident/operations | canonical runbook or incident evidence | experiment folder |
| Documentation correction | existing canonical document | parallel notes or handoff copies |

## 3. Principal-engineer implementation standard

Before editing, establish the observable acceptance criteria, invariants,
inputs/outputs, state owner, side effects, compatibility constraints, failure
modes, and resource budget. Inspect the owning module, its callers, adjacent
tests, configuration, and current diff. Fix the authoritative cause rather
than duplicating symptom handling.

All implementation work must follow these rules:

- Put each responsibility with the component that has the information and
  authority to enforce it. Keep domain policy separate from CLI, UI,
  persistence, filesystem, database, and provider adapters.
- Dependencies point toward stable policy. Do not make domain modules depend
  on scripts, experiments, report formats, UI state, or vendor response types.
- Prefer a direct function or cohesive module over a framework, factory,
  service locator, event system, plugin layer, or one-implementation interface.
- Use precise domain names, explicit data flow, type hints on new/changed
  public boundaries, and comments for rationale or non-obvious constraints.
- Do not create vague `manager`, `processor`, `helper`, `utils`, or `common`
  dumping grounds. Extend an existing cohesive owner or introduce a narrowly
  named module with a real invariant.
- Validate untrusted input at boundaries. Preserve root causes while returning
  actionable, redacted errors. Never silently swallow an unexpected failure,
  turn failure into an empty success, or leak secrets.
- For I/O or external work, define timeout, cancellation, cleanup, retry,
  idempotency, partial-failure, and observability behavior. Retries are bounded
  and selective; concurrency and queues are bounded.
- Preserve compatibility unless the request explicitly changes a contract.
  Treat defaults, error behavior, CLI output, config names, schemas, ordering,
  units, cutoff semantics, and persisted formats as contracts.
- Never ship placeholders, TODO implementations, disabled tests, fake
  production behavior, debug output, dead branches, copied source trees, or
  unexplained generated files.
- Optimize only a stated or measured bottleneck. Remove work and fix algorithm,
  query, or data shape before adding caching, batching, or parallelism.

For a bug, add a focused regression test that fails for the original cause
when feasible. For a feature, cover success, meaningful boundaries, invalid
input, and relevant dependency failure. Tests must be deterministic, offline
by default, and assert stable behavior rather than incidental call order.

Perform at least three independent reviews for normal code changes:

1. requirements and correctness;
2. architecture, responsibility, and simplicity;
3. failure, security, resource use, tests, and final integration diff.

Use a fourth dedicated integration/rollback pass for database, concurrency,
security, financially sensitive, migration, or high-blast-radius work.

## 4. Exact code and artifact ownership

Do not guess a location. Use this map, then inspect the existing owner before
creating a file.

| Responsibility | Canonical home | Placement rule |
|---|---|---|
| Target, settlement, as-of, publication, shared time/domain contracts | `src/hkg_tmax/` cohesive root modules | Stable HKG policy; no UI/vendor dependency |
| Acquisition orchestration | `src/hkg_tmax/acquisition.py`, `collector.py`, `fetch.py`, `hko_backfill.py` | Bounded collection/backfill coordination |
| Provider adapters | `src/hkg_tmax/hko.py`, `src/hkg_tmax/gribstream/`, `src/hkg_tmax/market.py` | Protocol parsing, request/response translation, provider failure rules |
| Normalization, storage, lineage, paths | `src/hkg_tmax/source_normalization.py`, `bronze.py`, `storage.py`, `manifest.py`, `paths.py` | `paths.py` is the only runtime-root resolver; never count parent directories or hardcode workstation paths |
| Reusable data/feature policy | `src/hkg_tmax/data/`, `src/hkg_tmax/features/` | Point-in-time-safe builders, lineage, and leakage guards |
| Core forecast algorithms | `src/hkg_tmax/modeling/` | Baselines, residual models, ensembles, routers, specialists |
| Evaluation/research infrastructure | `src/hkg_tmax/evaluation/`, `src/hkg_tmax/research/`, `src/hkg_tmax/metrics.py`, `distribution.py`, `statistics.py` | Reusable metrics, gates, evidence, and experiment infrastructure |
| Experiment registry/path contracts | `src/hkg_tmax/experiment_registry.py` | Campaign/ID/path schema, scaffold validation, reparse rejection, and atomic registry primitives |
| Experiment creation transaction | `src/hkg_tmax/experiment_transaction.py` | OS lock, campaign provisioning, ignored staging/journal, commit, rollback, and recovery |
| Experiment index/read models | `src/hkg_tmax/experiment_index.py` | Bounded top-level status discovery, generated index, and metrics-enriched status projection |
| Experiment public API | `src/hkg_tmax/experiments.py` | Compatibility façade only; new implementation belongs in one of the three owners above |
| Current full H24N/T-24 implementation | `src/hkg_t24/{audit,db,features,models,validation,live,orchestration,artifacts}/` | Authoritative home for all new full-strategy T-24 code |
| H24N/T-24 package entry primitives | `src/hkg_t24/cli.py`, `constants.py`, `timeutils.py` | CLI composition, cross-package constants, and canonical T-24 time semantics; do not duplicate these inside subpackages |
| H24N/T-24 technical utilities | `src/hkg_t24/utils/hashing.py`, `src/hkg_t24/utils/sql.py` | Extend only the matching hashing or SQL invariant; never add generic helpers to `utils/` |
| Legacy T-24 guard support | `src/hkg_tmax/hkg_t24/` | Change only its existing guard/governance/peak/moisture contracts; never add a second full T-24 implementation here |
| Probability engine | `src/hkg_tmax_probability/` | Buckets, PMFs, distributions, calibration, scoring, leakage audits, inference |
| DB audit/ingestion application | `src/hkg_tmax_db/` | DB contracts, connections, loaders, reconciliation, reports, CLI |
| Demo/backtester backend | `src/hkg_tmax_demo_trading/` | Demo domain, service, store, API, market/probability adapters |
| Demo/backtester frontend | `apps/polymarket-backtester/src/` | React UI only; server authority stays in backend/domain code |
| Frontend dependencies/build config | `apps/polymarket-backtester/package.json`, `apps/polymarket-backtester/vite.config.js` | Frontend scripts/dependencies and Vite behavior only; generated build/dependency trees stay ignored |
| Python packaging/tool configuration | `pyproject.toml` | Dependencies, console entry points, supported Python, and tool policy; do not create alternate requirement/setup files without a migration |
| Environment key contract | `.env.example` | Placeholder names, safe defaults, and comments only; real values remain in ignored `.env` or a secret store |
| Bounded developer aliases | `Makefile` | Thin aliases for canonical Python commands; no hidden network/live behavior and no requirement to install Make on Windows |
| Local container build/topology | `Dockerfile`, `compose.yaml`, `.dockerignore` | Reproducible local image/service topology and minimal build context; `pyproject.toml` remains the sole Python dependency authority |
| Project ignore policy | `.gitignore` | Generated/runtime/editor exclusions for this project; root `.gitignore` remains repository-wide authority |
| Hook policy | `.pre-commit-config.yaml` | Scoped serial checks only; never introduce all-files or automatic broad-format behavior |
| Project security entry point | `SECURITY.md` | Credential, disclosure, and production-safety routing; detailed controls stay in `docs/security/` |
| Package CLI | the owning `src/<package>/cli.py` | Argument parsing and composition only |
| Executable wrapper | `scripts/` | Thin, bounded entry point; reusable logic remains in `src/` |
| Core/package test | `tests/test_<domain>.py` | Mirror observable behavior of the owning module |
| Current T-24 test | `tests/hkg_t24/test_<area>.py` | Preferred coverage for `src/hkg_t24/` |
| Script test | `tests/test_<script_name>.py` | Arguments, safe defaults, output contract, failure behavior |
| Core contracts/config | `config/project/` | Target, as-of, evaluation, goals, buckets |
| Sources/acquisition config | `config/sources/`, `config/acquisition/` | Catalogs, stations, policy, disabled schedules |
| HKG Tmax research config | `config/experiments/hkg_tmax/` | Core/residual/probability experiment parameters |
| H24N/T-24 config | `config/hkg_t24/` | T-24 sources, features, evaluation, strategy |
| Ordered PostgreSQL migration | `db/migrations/postgres/YYYYMMDD_NNNN_<slug>.sql` | Append-only, compatibility-aware schema change |
| Schema/canonical SQL | `db/schemas/`, `db/sql/`, `db/sql/hkg_t24/` | Reviewed durable contract, never query-dump storage |
| Current documentation | matching owner under `docs/{architecture,contracts,data,operations,research,security,specifications,status,decisions,migrations}/` | Update the one canonical document; archives never override current truth |
| Implementation task/handoff | `planning/tasks/`, `planning/work-packages/<id>/` | Specifications only; never source/runtime copies |
| Governed experiment record | `experiments/campaigns/<campaign>/<experiment-id>/` | Compact dossier and machine evidence only |
| Raw/derived data | `${HKG_TMAX_DATA_ROOT}` | Always outside the repository |
| Logs/reports/models/predictions/run output | `${HKG_TMAX_RUN_ROOT}` | Always outside the repository |
| Lightweight process ownership | ignored `var/run/` | PID, owner, command, log pointer, health/stop commands only |

The flat `scripts/` directory is retained for compatibility; do not reorganize
it casually. Every new script needs a bounded CLI, `--help`, finite runtime,
worker default `1`, fail-closed live execution, and a focused test. Regenerate
`scripts/REGISTRY.csv` with `python scripts/build_script_registry.py` after
adding or changing scripts.

## 5. File-creation and anti-clutter gate

Before creating any file, answer all five questions:

1. Which durable responsibility owns it?
2. Which concrete caller, operator, test, or reader consumes it?
3. Why is modifying an existing cohesive file not clearer?
4. What verifies it and keeps it current?
5. Is it source, canonical documentation, compact evidence, or ignored/external
   runtime state?

If any answer is missing, do not create the file. Specifically:

- Never add arbitrary root files, scratch Markdown, notebooks, query dumps,
  temporary reports, copied specifications, backup files, alternate project
  roots, nested repositories, junctions, or parallel source/config trees.
- Reuse the canonical document and update it in place. Do not create
  `IMPLEMENTATION_NOTES.md`, `FINAL_REPORT.md`, handoff copies, or per-run
  Markdown when the owner README/runbook already exists.
- Temporary task-created state goes only under ignored `var/tmp/` or the
  external run root and is removed when no longer needed. Never delete
  pre-existing user state as cleanup.
- New tracked experiment JSON/YAML/CSV must be compact durable evidence. A new
  file above 1 MiB requires explicit justification; bulk rows, predictions,
  payloads, plots, logs, and models go external regardless of format.
- A new top-level directory requires a real new responsibility plus updates to
  the architecture map, docs index, path tests, and migration/rollback record.
- At final review, inspect scoped untracked files and account for every changed
  path. An unexplained file blocks completion.

## 6. Governed experiment creation

Create an experiment only for a predeclared, falsifiable research comparison.
Do not create one for ordinary code implementation, a bug fix, a refactor, an
acquisition run, an incident, or documentation work.

Choose exactly one allowlisted campaign:

| Campaign | Use for |
|---|---|
| `hkg-tmax` | core target, source, acquisition, feature, or point-forecast research |
| `hkg-t24` | H24N/T-24 strategy research |
| `residual-modeling` | residual correction, routing, and specialist research |
| `probability` | PMF, bucket, distribution, and calibration research |
| `market-edges` | model-versus-market comparison/replay only; never order execution |
| `general` | genuinely cross-cutting research that fits none of the above |

From `projects/hkg-tmax`, allocate the ID with the creator. Never hand-number,
copy, move, or rename an experiment and never edit the registry manually:

```powershell
.\.venv\Scripts\python.exe -m hkg_tmax experiments create `
  --campaign hkg-t24 `
  --title "Falsifiable hypothesis title"
```

The equivalent Make target is:

```powershell
make experiment CAMPAIGN="hkg-t24" TITLE="Falsifiable hypothesis title"
```

The experiment directory is created only beneath the selected campaign. The
creator also owns `experiments/registry/registry.yaml`, `EXPERIMENT_INDEX.md`,
and ignored `var/run` lock/journal plus `var/tmp` staging state:

```text
experiments/campaigns/<campaign>/EXP-####-<slug>/
  README.md
  STATUS.yaml
  DATA_MANIFEST.yaml
  RUN_CONFIG.yaml
  results/
    metrics.json
```

Creation is transactional. A later create automatically reconciles an ignored
interrupted-creation journal while holding the OS-backed registry lock: an
unchanged registry rolls back only exact token-proven staging/destination state;
a committed registry preserves the destination and repairs the derived index.
An unowned collision, malformed final marker, linked path, or journal mismatch is
preserved and reported, never claimed or recursively deleted. Never delete or
edit the lock, journal, marker, staging tree, registry, or generated index by
hand to force a retry.

Before viewing holdout results, fill in the hypothesis, mechanism, baseline,
minimum useful effect, target/as-of contract, frozen data slice, temporal
split, metrics, promotion gates, leakage checks, seed, runtime/worker budget,
and stop conditions. `STATUS.yaml` begins `PLANNED` and is updated through a
truthful lifecycle such as `RUNNING`, `COMPLETE`, `BLOCKED`, `REJECTED`, or
`ACCEPTED`.

The experiment directory is evidence, not an implementation package:

- Reusable logic goes in its owning `src/` package.
- Focused tests go in the matching `tests/` location.
- Governed parameters go in `config/experiments/hkg_tmax/` or
  `config/hkg_t24/` as appropriate.
- An optional runner is a thin `scripts/run_<experiment_slug>.py`; regenerate
  `scripts/REGISTRY.csv` after changing it.
- Large/mutable output goes to
  `${HKG_TMAX_RUN_ROOT}/experiments/<experiment-id>/<run-id>/` and is referenced
  by run ID, code revision, portable path, manifest, counts, sizes, and hashes.
- The tracked folder has exactly one human document: `README.md`. Update its
  bounded sections in place. Shards, runs, results, and compatibility copies
  must not create Markdown.

Never overwrite a completed experiment, change its frozen hypothesis/split,
or tune against a locked test. A materially changed hypothesis or candidate
gets a new governed ID. Record null, negative, rejected, inconclusive, and
blocked outcomes.

After an experiment change:

```powershell
.\.venv\Scripts\python.exe -m hkg_tmax validate registry
.\.venv\Scripts\python.exe -m hkg_tmax experiments index
.\.venv\Scripts\python.exe scripts\manage_campaign_documentation.py check
```

Update the selected campaign README and `EXPERIMENT_INDEX.md`; update
`MILESTONES.md` only for accepted material findings.

Creating the record does not authorize a network fetch, database mutation,
backfill, or expensive model run. Those require the current request to include
execution and the relevant resource/data authority.

## 7. Resource, process, network, and data safety

- Default agent, subprocess, model, BLAS, and I/O concurrency is `1`; maximum
  is `2` without explicit approval and measured justification.
- Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and
  `NUMEXPR_NUM_THREADS` to `1` for agent-run research.
- Never use `n_jobs=-1`, `pytest -n auto`, Maven `-T`, an unbounded executor,
  one-task-per-item fan-out, or parallel Git commands.
- Resolve `HKG_TMAX_DATA_ROOT` and `HKG_TMAX_RUN_ROOT` through
  `src/hkg_tmax/paths.py`. Fail closed if either resolves inside the repository.
  Do not print credentials or full secret-bearing environment values.
- Before a generator/run, confirm its mutable outputs are external or ignored.
  Use `git check-ignore -v -- <planned-local-runtime-path>` when a local ignored
  path is intentionally used.
- Live provider/network work needs explicit `--execute` plus hard provider,
  date/location/model, request, byte, retry, concurrency, and elapsed-time
  budgets. Stop on authentication failure, repeated rate limiting, or budget
  exhaustion.
- Database/schema mutation, scheduler installation, service startup, and
  trading remain fail-closed and require explicit authority and rollback.
- Background work records run ID, owner, exact command, working directory,
  PID/tree, start time, budget, log path, health check, and stop command. Stop
  only a freshly reverified owned process tree, children first.

## 8. CPU-safe Git operating protocol

Git operations are serialized and owned by one agent. Subagents may inspect
files, but they must not run concurrent status/add/commit/maintenance commands
or edit the index. Do not poll Git while a generator is writing many files.

`core.fsmonitor=false` is the proven safe setting for this Windows checkout.
Never set it to `true` and never enable untracked-cache, assume-unchanged,
skip-worktree, or Git maintenance as an autonomous optimization.

Use bounded discovery with explicit scopes:

```powershell
rg --files <exact-source-or-test-directory>
rg "pattern" <exact-source-or-test-paths>
git ls-files -- <exact-task-scope>
git diff --name-status -- <exact-task-paths>
git diff --stat -- <exact-task-paths>
git ls-files --others --exclude-standard -- <touched-paths>
```

The last untracked-file command is a final touched-path check, not startup
discovery. If any command unexpectedly expands to hundreds/thousands of paths,
large binaries, an external root, a junction, or sustained abnormal CPU/disk,
stop it once, preserve the command/process evidence, and investigate the
writer/path boundary. Do not retry with a broader command.

Never use during ordinary work:

- `git status --untracked-files=all`, root-wide untracked enumeration, or
  repeated status polling;
- `git add .`, `git add -A`, unscoped `git add -u`, `git commit -a`, or broad
  directory staging by convenience;
- `pre-commit run --all-files`, root-wide formatting, or full-workspace lint
  for a local change;
- `git clean`, hard reset, broad restore/checkout, history rewrite, or force
  push without explicit authorization and recovery plan;
- routine `git fsck`, `git gc`, `git repack`, `git maintenance start/run`, LFS
  migration, or index-flag manipulation as a response to slowness;
- deleting `.git`, lock files, worktrees, recovery archives, or generated
  trees to make Git appear clean.

Stage only a reviewed explicit file list:

```powershell
git add -- <file1> <file2>
git diff --cached --name-status
git diff --cached --stat
git diff --cached
```

For a legitimately large coherent change, create and inspect a pathspec
manifest outside the repository, then use `git add --pathspec-from-file=...`.
The manifest must contain only intended paths. Never convert a large change
into broad staging merely because the explicit list is long.

Before commit, run scoped `git diff --check -- <paths>` and inspect the complete
cached diff. Before push, recheck root, branch, remote, upstream, and tracked-only
status. A clean commit does not prove the correct repository or successful
remote push.

If CPU spikes, stop launching commands; inspect process name, full command
line, PID/parent, creation time, working directory, and task owner. Do not kill
Explorer or every Git/Python/Java/Node process. Stop only the verified owned
culprit, then run one bounded tracked-only status after the writer is quiet.
These rules materially reduce recurrence risk; they do not justify claiming
that any machine can literally never experience a CPU spike.

## 9. Verification ladder and completion

Run the cheapest check that proves the changed behavior, then widen once the
change is stable. From `projects/hkg-tmax`:

1. environment/project health:

   ```powershell
   .\.venv\Scripts\python.exe -m hkg_tmax doctor
   .\.venv\Scripts\python.exe ..\..\tools\repo\doctor.py `
     --root ..\.. --scope projects/hkg-tmax
   ```

2. syntax/lint only for changed modules:

   ```powershell
   .\.venv\Scripts\python.exe -m ruff check <changed-python-files>
   .\.venv\Scripts\python.exe -m mypy <changed-package-paths>
   .\.venv\Scripts\python.exe -m compileall -q <changed-package-paths>
   ```

3. focused tests that directly exercise the behavior:

   ```powershell
   .\.venv\Scripts\python.exe -m pytest -q <focused-test-files>
   ```

4. bounded project gates when the blast radius warrants them:

   ```powershell
   make test-fast
   make validate
   ```

   `make` is optional. On Windows, use the exact venv-Python commands in
   `README.md`; never install Make merely to run verification aliases.

5. `make test-full` only for release/cutover or justified broad impact, always
   serial. Provider/database/live tests remain disabled unless explicitly
   authorized. For frontend work, run the package build plus real browser QA;
   the frontend currently has no standalone unit-test script.

Do not run `make format` over the whole project for a local edit. Format or lint
explicit changed files, inspect formatter output, and reject unrelated churn.

Update canonical documentation with the behavior:

- material behavior: `CHANGELOG.md`;
- ownership/layout: `docs/architecture/PROJECT_STRUCTURE_AND_CODE_MAP.md`;
- target/as-of/API contract: the owning `docs/contracts/` or specification;
- operations: `docs/operations/`;
- verified current state only: `docs/status/CURRENT_STATE.md`;
- research: experiment README/status/evidence and generated index.

Work is complete only when every requirement is implemented or transparently
blocked; responsibilities and paths are coherent; failure/security/resource
behavior is deliberate; focused tests and proportional broader gates pass;
unrelated work is preserved; no secrets, runtime payloads, scratch files,
formatting churn, or unexplained paths remain; rollback is understood; and the
final response states exactly what passed, failed, was skipped, and remains
unverified.

Stop and report rather than guessing if repository identity, contract
authority, data provenance, campaign placement, external-state permission,
resource bounds, overlapping user edits, or rollback cannot be established
from current evidence.
