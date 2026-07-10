# Negative Results and Research Memory

## Why preserve negative evidence

The sequential program can otherwise repeat the same weak lane under different names. Every null, adverse, blocked, or invalid result must enter cumulative memory.

## Registry schema

`negative_results_registry.csv` fields:

- experiment ID;
- date;
- hypothesis;
- mechanism;
- data families;
- stations/groups;
- feature family;
- response;
- baseline;
- frame;
- status;
- observed effect;
- fold consistency;
- tail effect;
- reason for failure;
- whether mechanism is falsified;
- whether implementation is inconclusive;
- blocker;
- conditions under which retest is allowed;
- related experiments;
- Director decision.

## Failure taxonomy

- `NO_SIGNAL`
- `REDUNDANT_WITH_BASELINE`
- `UNSTABLE_SIGN`
- `ONE_ERA_ONLY`
- `ONE_SEASON_ONLY`
- `TOO_SPARSE`
- `MISSINGNESS_DRIVEN`
- `SOURCE_ERA_PROXY`
- `WRONG_RESPONSE_ROLE`
- `WRONG_TIMESCALE`
- `HARD_BUCKET_FRAGILITY`
- `COMPLEXITY_DILUTION`
- `ROUTER_DILUTION`
- `ANALOG_DISTANCE_DILUTION`
- `TIMESTAMP_BLOCKED`
- `TARGET_LEAKAGE`
- `FRAME_MISMATCH`
- `DATA_QUALITY`
- `RUNTIME_FAILURE`
- `INCONCLUSIVE`

## Retest rule

A failed lane may be retested only when the new specification states a material reason:

- safer or newly unlocked source;
- more complete frame;
- corrected implementation/data defect;
- response-role change supported by diagnostics;
- smooth/hierarchical replacement for sparse cells;
- role-group abstraction replacing station quirk;
- physically restricted analog distance;
- new interaction implied by evidence;
- independent replication frame.

Changing model seed, bin edge, or learner without a diagnosis is not sufficient.

## Weak prior lessons to carry forward

The current corpus suggests these cautions:

- pure station-only models have not replaced the official anchor;
- diagnostic relationships often produce tiny deployable MAE gains;
- sparse hard cells can be fragile;
- broad expert stacks can dilute strong simple correction;
- unrestricted analog matching can dilute physical similarity;
- pressure level alone may be weaker than pressure-moisture-wind interaction;
- results on 2670-row and 5265-row frames are not directly comparable;
- upper-air/daily/marine mechanisms remain blocked for deployable scoring without timestamp proof;
- short high-frequency feeds cannot support multi-decade claims.

These are priors, not universal prohibitions. A new experiment must say why it escapes the prior failure.
