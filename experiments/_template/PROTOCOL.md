# Predeclared Protocol

Complete before inspecting validation/locked-test outcomes.

## Target and horizon

- target version:
- rules/target adapter version:
- horizon:
- exact cutoff:
- prediction unit:

## Sample

- development:
- validation:
- locked test:
- live shadow:
- inclusion:
- exclusion:
- expected row count:

## Baseline

- champion/baseline version:
- frozen prediction artifact:
- reason:

## Candidate

- feature/formula/model:
- transformations:
- allowed hyperparameters:
- selection procedure:
- seeds:
- compute budget:

## Metrics

- primary:
- guardrails:
- calibration:
- subgroup:
- operational:

## Uncertainty

- method:
- block length:
- repetitions:
- confidence level:

## Multiplicity

- experiment family:
- number of variants:
- correction/confirmation approach:

## Acceptance

State exact quantitative and qualitative gates.

## Rejection

State exact falsification or unacceptable degradation criteria.

## Locked-test decision

State whether this experiment is authorized to open the locked test. Add the corresponding `TEST_ACCESS_LOG` entry before access.
