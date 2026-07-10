# Experiment Synthesis and Novelty Protocol

## Goal

Generate new lanes from the complete evidence corpus without relabeling old work or repeating weak searches.

## Required synthesis inputs

Before candidate generation, load:

- experiment evidence registry;
- all experiment specifications;
- README, RESULTS, CONCLUSION, and leakage audits;
- scoreboards and prediction artifacts for relevant champions;
- negative-results registry;
- source eligibility matrix;
- station dossier and attribute catalog;
- interaction queue;
- blockers;
- current residual anatomy.

## Evidence graph

Represent the corpus as links among:

- experiment;
- data family;
- attribute;
- station/group;
- transformation;
- response;
- regime;
- baseline;
- model form;
- frame;
- result;
- blocker.

Use this graph to find:

- strong signals never tested incrementally;
- complementary weak signals from different experiments;
- mechanisms with diagnostic evidence but no safe proxy;
- high-error clusters without a specialist;
- named-station effects without role abstraction;
- source-era conflicts;
- failed hard cells that may need smoothing;
- complex models that may need simplification;
- blind spots in response coverage.

## Candidate synthesis operators

A new experiment may arise from:

1. **Refinement** — fix a diagnosed defect in a previous implementation.
2. **Combination** — combine complementary signals whose joint mechanism is explicit.
3. **Conditioning** — test a signal only in the regime where its sign should hold.
4. **Role transfer** — move a weak point feature into uncertainty, routing, or tail detection.
5. **Proxy conversion** — approximate a blocked mechanism with safe variables.
6. **Compression** — replace fragile stations/features with a stable group abstraction.
7. **Residualization** — test information beyond the current champion.
8. **Frame repair** — harmonize incomparable evidence.
9. **Era repair** — model or bridge source drift.
10. **Data unlock** — resolve a blocker with large downstream value.
11. **Representation change** — level to anomaly, hard bin to smooth state, static to propagation.
12. **Adversarial simplification** — remove dilution or overfitted experts.
13. **Independent exploration** — test a plausible mechanism not represented in the corpus.

## Novelty audit

For every candidate:

- list nearest prior experiments;
- calculate lexical/spec similarity as a screening aid;
- compare data families;
- compare feature formulas;
- compare response;
- compare baseline;
- compare regime;
- compare validation;
- state the material difference;
- state which prior failure it addresses;
- state what new belief the result will change.

Novelty classes:

- `NEW_MECHANISM`
- `NEW_DEPLOYABLE_PROXY`
- `NEW_RESPONSE_ROLE`
- `MATERIAL_REFINEMENT`
- `FRAME_OR_DATA_REPAIR`
- `REPLICATION`
- `DUPLICATE_OR_COSMETIC`

`DUPLICATE_OR_COSMETIC` is rejected. Replication is allowed when it is intentionally testing robustness on a new frame.

## Candidate priority score

Record components from 0 to 5:

- expected information gain;
- deployable MAE potential;
- physical plausibility;
- prior support;
- novelty;
- readiness;
- sample sufficiency;
- robustness potential;
- downstream breadth;
- backfill durability.

Penalties:

- timestamp risk;
- data-quality risk;
- research-overfit risk;
- complexity;
- implementation cost;
- dependence on one station/year;
- archive continuity.

The score is a decision aid, not a mechanical substitute for red-team review.

## Combination discipline

When combining prior signals:

- identify their individual incremental value;
- explain why interaction is expected;
- compare main effects versus interaction;
- avoid combining two redundant variants;
- predeclare correction allocation;
- use shrinkage;
- test whether one signal merely gates the other;
- preserve interpretability.

“Use all successful features together” is prohibited.

## Reserve queue

Keep at least five ranked candidates when possible. After one experiment completes:

- update scores;
- remove candidates falsified by new evidence;
- refine candidates whose premise strengthened;
- add newly discovered lanes;
- re-run novelty audit.

Do not automatically execute the old number-two candidate without incorporating the latest result.
