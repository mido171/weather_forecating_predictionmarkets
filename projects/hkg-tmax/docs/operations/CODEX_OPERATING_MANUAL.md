# Codex Operating Manual

## Orientation after context compaction

Read in this order:

1. `MILESTONES.md`
2. `EXPERIMENT_INDEX.md`
3. `FIRST_GOALS.md`
4. `AGENTS.md`
5. active experiment `README.md`, `STATUS.yaml`, and `CONCLUSION.md`
6. linked prior experiments
7. relevant docs/config.

This provides state without rereading every raw artifact.

## Delegation pattern

Use subagents explicitly for independent work:

- `target_parity_researcher` — source/rules/settlement;
- `data_engineer` — archives and schemas;
- `meteorology_researcher` — physical experiments;
- `leakage_auditor` — independent read-only audit;
- `reproducibility_reviewer` — independent rerun;
- `market_microstructure_researcher` — execution.

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

- `CONCLUSION.md`;
- `RESULTS.md`;
- `STATUS.yaml`;
- `DATA_MANIFEST.yaml`.

Read logs/artifacts only when needed.

## Milestone discipline

`MILESTONES.md` is not a lab notebook. It includes only:

- accepted gains;
- precise metric deltas;
- experiment IDs;
- caveats;
- current champion;
- blockers and rejected high-level ideas.

All raw exploration belongs in experiment folders.
