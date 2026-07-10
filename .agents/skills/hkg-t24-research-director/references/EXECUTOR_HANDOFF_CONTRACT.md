# Research Director → Experiment Executor Handoff Contract

## Role separation

The Director selects and specifies. The Executor implements and verifies. Neither may silently perform the other's scientific decisions.

## Handoff package

The Director must provide:

- absolute repository root;
- canonical experiments root;
- experiment specification path;
- specification SHA-256;
- requested experiment ID only when reserved;
- applicable cutoff contract;
- canonical frame artifact;
- baseline artifact or deterministic reproduction rule;
- source eligibility snapshot;
- station dossier snapshot when stations are used;
- explicit instruction to keep 2024+ sealed.

## Specification completeness

The Executor must not guess:

- feature formulas;
- response sign;
- cutoff;
- station selection;
- missingness policy;
- training windows;
- hyperparameter budget;
- fold boundaries;
- baseline;
- correction cap;
- primary metric;
- acceptance threshold.

If missing, the Executor creates `REJECTED_SPECIFICATION`.

## Invocation template

```text
Use $hkg-t24-experiment-executor.

Repository root: <absolute path>
Experiment specification: <absolute path to experiment_spec.json>
Experiments root: <absolute path>/experiments

Execute exactly this one pre-registered experiment. Do not change its
hypothesis, features, response, frame, baseline, parameter budget, or gates.
Assume the decision is at the canonical T-24 cutoff. Reject any predictor
without proof of pre-cutoff availability. Keep 2024+ sealed. Create a complete
experiment folder even if rejected. Run validate_experiment_folder.py --strict
and return the folder path plus baseline MAE, candidate MAE, delta, leakage
status, and promotion decision.
```

## Executor return contract

Return:

- folder path;
- experiment ID;
- status;
- date range and common rows;
- baseline ID/MAE;
- candidate ID/MAE;
- delta;
- RMSE/bias/tails;
- leakage status;
- confirmation rows used;
- validator result;
- promotion decision;
- deterministic repair action if failed.

## Director ingestion gate

The Director does not ingest a headline result until:

- folder exists;
- validator passes or rejection is valid;
- summary matches scoreboard;
- predictions reproduce score;
- row identity is valid;
- source eligibility matches the specification;
- no sealed rows were read.

## Repair behavior

Implementation bug:

- preserve original folder;
- create a new repair/rerun experiment linked to it;
- do not edit historical score claims silently.

Specification flaw:

- complete the rejection folder;
- Director designs a new experiment.

Data blocker:

- complete blocked folder;
- update blocker and source matrices;
- continue another lane if possible.
