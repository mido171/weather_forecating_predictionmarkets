# Specialist Training Specification

## Required three-part structure

Every specialist consists of:

1. regime detector;
2. bounded residual correction model;
3. expected-benefit/abstention gate.

No specialist is a hand-written unconditional rule.

## Candidate specialists

- marine suppression;
- weak-wind heat buildup;
- MAM transition;
- cloud/rain/radiation suppression;
- cool-surge breakdown and rebound;
- dry subsidence/ridge heating;
- tropical-cyclone peripheral-flow state;
- high-error-tail prevention;
- hot underforecast and cold overforecast asymmetry;
- source/model disagreement state.

## Training

Regime hypotheses define candidate inputs. Historical OOF residuals determine whether the regime has directional value. Fit the detector and correction within each training fold. Compute OOF benefit:

```text
benefit = abs(anchor_error) - abs(specialist_corrected_error)
```

Train the gate to predict benefit. Activate only when regime probability, expected benefit, support count, and uncertainty meet fold-tuned thresholds.

## Minimum evidence

Default promotion requirements:

- at least 200 active OOF dates overall;
- at least 40 active dates in each of at least three outer folds;
- at least three distinct years and multiple seasonal instances;
- stable correction sign;
- positive MAE lift in at least three folds;
- no material 90th/95th percentile deterioration;
- no contamination outside the activation slice.

Rare specialists must use hierarchical shrinkage toward zero and cannot receive large caps.

## Date ranges

- official/target-memory specialists: 2000–2023 development, era-aware;
- GFS/GEFS-informed specialists: 2021-03-22 through 2023-12-31 development;
- IFS/AI specialists: only after core freeze and sealed opening;
- ARWF/CWA specialists: prospective shadow until sufficient exact-vintage history.
