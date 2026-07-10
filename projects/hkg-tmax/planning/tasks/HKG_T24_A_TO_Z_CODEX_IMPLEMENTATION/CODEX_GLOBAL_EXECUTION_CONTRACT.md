# Codex Global Execution Contract

## Operating mode

Each task is an implementation assignment, not a request for a plan. Codex must inspect the repository, implement the required code and migrations, run the required commands against the configured environment, execute tests, and create the required bookkeeping folder. When a dependency or credential is unavailable, implement everything that can be completed, create a precise blocker artifact, and never fabricate successful execution.

## Experiment/infrastructure folder rule

Every task creates one folder under:

```text
C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\experiments
```

Use the next unused four-digit numeric prefix and the task suffix specified in the task document. Never overwrite an existing folder. Each folder must include at minimum:

```text
README.md
RESULTS.md
CONCLUSION.md
task_spec.json
run_manifest.json
data_manifest.csv
leakage_audit.md
quality_report.csv
commands_executed.txt
tests/
logs/
src_or_migration_manifest.txt
```

When a task is rejected because data is unavailable, timestamp proof fails, geographic coverage is absent, or a source is contractually blocked, still create the full folder and record the rejection.

## Universal implementation requirements

- Read every applicable repository `AGENTS.md` and installed HKG T+24 skill/constitution.
- Reuse existing migration, configuration, logging, and test conventions.
- Keep credentials in environment variables only.
- Make every loader idempotent and resumable.
- Hash raw requests, responses, source files, configs, and model artifacts.
- Preserve raw source truth; clean by producing normalized tables and quarantine records, never by destructive overwrite.
- All timestamps are canonical `timestamptz` UTC in storage. HKT is derived explicitly using `Asia/Hong_Kong`.
- Every model feature records model code, run time, valid time, lead, member, selector version, and eligibility proof.
- Every model score compares candidate and baseline on identical target dates.
- All preprocessing and feature selection are fitted inside each training fold.
- Never allow target T, target residuals, target-derived flags, or future labels into inputs.
- Do not automatically open 2024+ outcomes.

## Completion standard

A task passes only when its machine-readable acceptance criteria pass. A narrative claim is insufficient. Each task must leave a handoff manifest that the next task can consume without guessing paths, table names, versions, or unresolved decisions.
