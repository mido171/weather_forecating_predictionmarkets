# Codex Operating Manual

## Orientation after context compaction

Restart the single mandatory sequence in `AGENTS.md` section 2. It includes the
bounded Git proof; root constitution/start/map/safe-command documents; closest
applicable agent contracts; project `START_HERE.md`, `README.md`, code map,
current state, and docs index; then only task-relevant contracts, code, tests,
config, and evidence. A compaction summary does not replace any step.

For research, the task-relevant evidence normally begins at
`EXPERIMENT_INDEX.md`, the selected campaign README, and the experiment
`README.md`, `STATUS.yaml`, and evidence map. Follow linked prior decisions and
`MILESTONES.md` only when an accepted result is relevant; do not reread the raw
campaign corpus.

## Delegation pattern

Use subagents explicitly for independent work:

- `target_parity_researcher` — source/rules/settlement;
- `data_engineer` — archives and schemas;
- `meteorology_researcher` — physical experiments;
- `leakage_auditor` — independent read-only audit;
- `reproducibility_reviewer` — independent rerun;
- `market_microstructure_researcher` — market evidence and replay; never order
  execution.

Never ask the same agent that built a result to be its only auditor.

## Parallelism

Safe parallel work:

- source-adapter discovery by independent source family;
- station metadata audits;
- separate physical hypothesis families;
- independent audits after outputs freeze.

Unsafe parallel work:

- multiple agents editing the same experiment;
- model research before target/split contract;
- separate agents selecting winners on the same locked test without multiplicity control.

## Memory discipline

Each experiment must summarize:

- what was believed before;
- what was tested;
- what changed;
- what remains unknown;
- which artifact proves it.

The milestone file is always current. Never rely on chat history as the only record.

## Task completion format

Every Codex task response should state:

```text
Goal / experiment:
Files changed:
Commands run:
Evidence produced:
Gate status:
New risks:
Next dependency:
```

## Evidence discipline

When researching external facts:

- prefer official provider documentation;
- archive page/PDF and retrieval date;
- quote minimally;
- distinguish source fact from inference;
- record unknowns rather than inventing behavior.

## Code discipline

- full implementation and tests;
- type hints;
- fail-closed errors;
- useful error messages;
- no silent `except`;
- no hard-coded secrets;
- no untracked manual step;
- small modules with explicit contracts;
- immutable outputs.

## Context-efficient experiment reading

Start from:

- the campaign `README.md`;
- the experiment `README.md`, especially its status, result, decision, and
  limitations sections;
- `STATUS.yaml`;
- `DATA_MANIFEST.yaml` and `results/metrics.json` when present.

Read logs/artifacts only when needed.

## Milestone discipline

`MILESTONES.md` is not a lab notebook. It includes only:

- accepted gains;
- precise metric deltas;
- experiment IDs;
- caveats;
- current champion;
- blockers and rejected high-level ideas.

The governed experiment README records the research decision and evidence map.
Raw payloads, predictions, plots, logs, models, and mutable run outputs stay
under the configured external data/run roots.
