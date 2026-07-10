# Experiment Protocol

## One hypothesis, one directory

An experiment is a falsifiable test, not a notebook dump. Closely related parameter sweeps can share one experiment only when the family and selection rule are predeclared.

## Status lifecycle

```text
PLANNED → PREDECLARED → RUNNING → REVIEW →
ACCEPTED | REJECTED | INCONCLUSIVE | BLOCKED
```

No result may jump from ad hoc analysis directly to `ACCEPTED`.

## Required predeclaration

Before holdout outcomes:

- mechanism and rationale;
- exact feature/transform;
- expected direction;
- regimes where effect should and should not work;
- target and horizon;
- source versions and eligibility;
- sample and exclusions;
- split/rolling procedure;
- baseline;
- model or formula;
- hyperparameter search space;
- primary metric;
- guardrail metrics;
- uncertainty method;
- multiplicity family;
- acceptance/falsification criteria;
- computational budget.

## Result reporting

Always include:

- all attempted specifications within the declared family;
- common-sample and available-sample metrics;
- absolute and relative deltas;
- confidence interval;
- year-by-year values;
- regime slices;
- calibration;
- worst cases;
- missingness;
- runtime and operational cost;
- ablation;
- sensitivity;
- negative controls.

Do not highlight only the best seed, cutoff, season, or subgroup.

## Multiple testing

The project will generate many hypotheses. Control false discovery through:

- experiment-family registration;
- validation confirmation;
- locked-test restraint;
- hierarchical interpretation;
- false-discovery-rate methods where appropriate;
- effect-size and mechanism requirements;
- live shadow confirmation.

A small p-value after 500 unrecorded trials is not evidence.

## Experiment inheritance

A new experiment may reference prior immutable artifacts by hash, but must state:

- what is reused;
- what changed;
- why a new test is needed.

## Code state

Record:

```text
git commit
branch
dirty state
diff patch hash
Python version
dependency snapshot
OS/platform
random seeds
command
start/end time
```

If Git is not initialized, the first goal initializes it before accepted experiments.

## Data manifest

Every input must be identified by:

- source ID;
- date range;
- station/model;
- raw/derived version;
- manifest or file hashes;
- point-in-time role;
- transformation version;
- row count;
- exclusions.

## Reproduction tolerance

Declare expected tolerance:

- exact bytes for deterministic tables where possible;
- numeric tolerance for floating-point metrics;
- statistical tolerance for stochastic models;
- seeds and number of repeats.

## Experiment index

`STATUS.yaml` is machine-readable truth. `EXPERIMENT_INDEX.md` is generated navigation. `MILESTONES.md` contains only accepted, material findings.

## Notebook policy

Notebooks may explore development data, but accepted logic must move to tested modules/scripts. An experiment that depends on manually executed hidden notebook state cannot pass reproducibility.
