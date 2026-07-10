# Weather Markets Agent Constitution

This file governs every task in this repository. A closer `AGENTS.md` may add stricter,
component-specific rules but may not weaken this file. Open Codex from this repository root
so these rules are loaded before any scoped instructions.

## 1. Mission and boundaries

Maintain one safe, understandable workspace for weather acquisition, forecast research, and
weather-market integrations. Protect user work, point-in-time correctness, credentials,
provider quotas, databases, local machine resources, and real trading state.

Never infer authorization for network collection, database mutation, schema migration,
scheduler installation, service startup, backfill execution, order placement, Git history
rewriting, or deletion from a request to inspect, explain, organize, or test.

## 2. Mandatory startup protocol

Before editing or running project code:

1. Run bounded Git identity checks:

   ```powershell
   git rev-parse --show-toplevel
   git branch --show-current
   git remote -v
   git -c core.fsmonitor=false status --short --branch --untracked-files=no
   ```

2. Confirm the reported root is this standalone repository and `.git` is a directory.
3. Read, in order:
   - this file;
   - `docs/START_HERE.md`;
   - `docs/architecture/REPOSITORY_MAP.md`;
   - `docs/operations/SAFE_COMMANDS.md`;
   - the closest scoped `AGENTS.md`;
   - the component/project `START_HERE.md`, `README.md`, and current-state document;
   - the exact task specification and only its relevant referenced contracts.
4. Inspect the smallest relevant code, tests, configuration, and current state.
5. Classify the task using section 4 before creating artifacts.

Startup is read-only and offline. Do not fetch, install, backfill, schedule, start a service,
connect to a provider, contact a market, migrate a database, or place an order at startup.

## 3. Source authority

When sources disagree, use this order:

1. Platform safety rules and the user's explicit current request.
2. This root constitution.
3. The closest scoped `AGENTS.md`.
4. Canonical specifications identified by `docs/START_HERE.md` or a scoped start document.
5. Executed code, tests, schemas, and verified live state.
6. Accepted experiment conclusions and generated registries.
7. Historical reports, handoff bundles, copied contracts, planning archives, and legacy code.

Archives, handoffs, generated outputs, and old path snapshots are evidence, not governing
authority. Do not silently choose a convenient copy when two documents conflict; identify
the conflict and resolve it against the authority chain.

## 4. Task classification and artifact routing

Classify work before acting:

| Task | Required record | Do not create automatically |
|---|---|---|
| Research hypothesis or model comparison | Experiment manifest and conclusion | Engineering change log |
| Engineering feature, fix, or refactor | Code, focused tests, relevant docs | Experiment folder |
| Data acquisition or backfill | Bounded acquisition run ledger | Research experiment |
| Operational incident | Incident record and verified remediation | Experiment folder |
| Repository migration or maintenance | Migration ledger and path map | Forecast experiment |
| Documentation-only correction | Canonical documentation change | Code or run artifacts |

An experiment is not a universal task log. Failed research is retained and concluded; failed
engineering attempts belong in Git history or the task report, not new experiment IDs.

## 5. CPU, memory, disk, and scan safety

Default agent concurrency is one worker. Two workers are the repository maximum without
explicit user approval and a measured reason. Set numerical-library thread counts to one for
agent-run research unless a task authorizes otherwise:

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
```

Do not use these as discovery or startup commands:

- root-wide `Get-ChildItem -Recurse`;
- `tree /F`;
- root-wide `rg -uu`, unrestricted hidden scans, or junction-following scans;
- `git status --untracked-files=all`;
- full-workspace hashing or loading all Parquet/CSV files;
- `python -m compileall .`;
- root-wide `pytest`, `pytest -n auto`, or automatic test parallelism;
- full `make test`, `make validate`, or Maven parallel builds at startup;
- `n_jobs=-1`, unbounded process pools, or provider concurrency above two.

Use scoped `rg`, `rg --files <scope>`, explicit directories, representative samples, and
bounded timeouts. Never follow a junction/reparse point during routine discovery. Stop and
report if a command unexpectedly expands into generated data, consumes sustained resources,
or needs a broader scope than the user authorized.

## 6. Network, provider, backfill, and scheduler safety

All live ingestion, collectors, backfills, polling, WebSockets, schedulers, and trading are
disabled by default. A live run requires an explicit execution request and a written budget:

- provider and endpoint;
- station/location and model scope;
- start/end time and cutoff semantics;
- maximum requests, bytes, rows, retries, concurrency, and elapsed time;
- destination and idempotency/resume key;
- expected cost or quota effect;
- dry-run result and exact stop condition.

Use an explicit `--execute` or equivalent opt-in. Retries must be bounded, selective,
idempotent, and observable. Never install or modify a scheduled task as an incidental step.
Never start a service merely to discover its configuration.

## 7. Process and service ownership

Never terminate all Python, Java, Node, browser, or shell processes. Before starting or
stopping anything, verify executable path, complete command line, parent/child chain, working
directory, listening port, creation time, and task ownership.

Background processes must record under ignored `var/run/`:

- PID and child PIDs;
- exact command and working directory;
- owner/task and start time;
- port and log path;
- health check;
- exact stop command.

Re-query ownership immediately before stopping; PIDs are reusable. Stop only the verified
tree, children first, and confirm ports/resources are released. Do not reuse stale PIDs from
reports or earlier conversations.

## 8. Git safety

Preserve all existing changes and unrelated work. The following require explicit user
authorization and a recovery plan:

- `git clean`, `git reset --hard`, broad restore/checkout, or recursive deletion;
- amend, rebase, filter-repo, force-push, history rewriting, or branch deletion;
- deleting `.git`, a worktree, a nested repository, recovery bundle, or rollback archive;
- staging the whole worktree with `git add .` or `git add -A`.

Stage explicit paths or a reviewed pathspec manifest. Before committing:

```powershell
git diff --check
git diff --cached --name-status
git diff --cached --stat
```

Inspect the complete cached diff. Immediately before pushing, re-run Git root, branch,
remote, upstream, and status checks. A successful local commit is not a successful push.
Never assume the deepest repository or current directory is the intended publish target.

## 9. Data, artifacts, and storage

Git contains source, tests, schemas, small fixtures, catalogs, manifests, and durable
conclusions. Large datasets, models, predictions, caches, logs, temporary tables, exports,
and run artifacts live outside Git. `var/` is ignored lightweight local state only.

Use environment-configured data roots. The established HKG root is `C:\hkg_tmax_data`.
Do not create junctions from the repository to an external data tree; common scanners follow
them and can duplicate work. For data moves use copy, manifest verification, configuration
switch, observation, and delete-later. Never use move/mirror semantics before verification.

Every material run records inputs, configuration, code revision, output location, counts,
sizes, schema/version, checksums where proportionate, and retention class. A path alone is
not provenance.

## 10. Secrets, production, and trading

Never commit, print, quote, or copy credentials. Use environment variables or an approved
secret store. Example files contain placeholders only. Redact command lines and errors before
reporting them. If a tracked secret is found:

1. do not repeat its value;
2. stop publishing;
3. neutralize the current file;
4. request/provider-rotate the credential;
5. assess historical exposure separately;
6. keep recovery bundles access-restricted.

Production environment, authentication, trading, reconciliation, startup collectors, and
database mutations must be fail-closed. No order placement or production mutation is an
implicit test. Financially sensitive behavior requires explicit authorization and a narrow,
observable verification plan.

## 11. Verification ladder

Use the least expensive check that proves the changed behavior, then widen according to risk:

1. `python tools/repo/doctor.py` and focused formatting/syntax checks.
2. Focused unit tests for the changed module.
3. Relevant integration/contract tests with network and mutations disabled.
4. Scoped project test/lint/type/build checks.
5. Broader serial validation in a controlled maintenance window.
6. Live smoke only when explicitly authorized and budgeted.

For migrations and security-sensitive changes, perform separate correctness, design,
reliability/security, test/resource, and integration/rollback reviews. Report exactly what
passed, failed, was skipped, or remains unverified. Do not weaken assertions, hide warnings,
or call reasoning a test.

## 12. Documentation and completion

Update the smallest canonical document that owns the changed contract:

- repository shape or ownership: `docs/architecture/`;
- operational behavior or runbook: `docs/operations/`;
- credentials or production safety: `docs/security/`;
- structural/history/data move: `docs/migrations/`;
- project-specific behavior: that project's canonical docs.

Do not copy canonical specifications into task folders. Link to them and record a revision or
checksum when a frozen contract is required.

Work is complete only when scope is accounted for, unrelated work is preserved, relevant
checks pass or failures are explicit, generated/runtime material is not accidentally tracked,
documentation is current, rollback is viable, and Git root/branch/remote state is known.

## 13. Stop conditions

Stop and report rather than guessing when:

- repository identity differs from the expected standalone root;
- a nested `.git`, junction, tracked credential, or unexplained large payload is found;
- a migration path lacks a verified source/destination mapping or rollback;
- live execution lacks budgets, ownership, credentials, or user authorization;
- current changes overlap in a way that cannot preserve both owners;
- disk, memory, CPU, provider quota, or financial exposure exceeds the declared limit;
- tests require unapproved network, database mutation, scheduler, service, or trading access.
