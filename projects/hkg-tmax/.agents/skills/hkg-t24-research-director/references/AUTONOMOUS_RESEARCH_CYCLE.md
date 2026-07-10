# Autonomous Sequential Research Cycle

## Purpose

This protocol turns the Research Director and Experiment Executor into a controlled sequential research loop. The loop is allowed to be persistent and ambitious, but it is never allowed to become an unlogged hyperparameter search against one development frame.

## State machine

The canonical state file is `<repo-root>/.hkg_t24_research/research_state.json`.

Valid phases:

- `DISCOVERY`: no experiment is open; the Director is refreshing evidence and selecting the next experiment.
- `EXECUTING`: exactly one pre-registered experiment is open and assigned to the Executor.
- `DEVELOPMENT_GATE_REACHED`: an eligible frozen development candidate has MAE <= 0.45 °C on its declared canonical frame.
- `AWAITING_CONFIRMATION_AUTHORIZATION`: the winning candidate is frozen and 2024+ remains sealed.
- `CONFIRMED_TARGET_REACHED`: a one-time owner-authorized confirmation passed the declared target.
- `BLOCKED`: no valid execution can proceed until a named data, timestamp, code, or environment blocker is repaired.
- `STOPPED`: the owner stopped or changed the program.

Only the Director may move the state from `DISCOVERY` to `EXECUTING`. Only a validated Executor folder may move it back to `DISCOVERY` or into a gate/block state.

## Initialization

1. Resolve the repository root. Never infer a different project when the supplied path exists.
2. Set:
   - experiments root: `<repo-root>/experiments`
   - datasets root: `<repo-root>/data/datasets`
   - research memory: `<repo-root>/.hkg_t24_research`
3. Resolve the exact canonical T−24 cutoff contract. If it cannot be found or is ambiguous, create a rejected cutoff-definition experiment before any deployable model experiment.
4. Run the repository inventory, dataset census, and experiment index.
5. Initialize the source eligibility matrix, station dossier, attribute catalog, evidence registry, negative-results registry, and champion ledger.
6. Keep confirmation rows at `2024-01-01` and later sealed.

## One complete iteration

### A. Ingest and reconcile

- Validate all previously completed experiment folders.
- Read every experiment specification, README, results, conclusion, leakage audit, summary, scoreboard, and fold/year/slice artifact.
- Reconcile duplicated or conflicting metrics.
- Group experiments by canonical frame and response.
- Update the current champion for each frame only when row, target, and leakage comparability pass.
- Record unreadable or incomplete folders as blockers; do not silently omit them.

### B. Diagnose the current bottleneck

Determine whether the highest-value bottleneck is:

- frame comparability;
- timestamp eligibility;
- station identity or coverage;
- target-memory representation;
- station-network mechanism;
- official-anchor bias;
- source-era drift;
- high-error regime;
- blocked-physics proxy;
- uncertainty/routing;
- model form;
- data quality;
- archive continuity;
- acquisition.

Use residual anatomy from the strongest compatible system, not merely raw target correlations.

### C. Generate candidate lanes

Generate at least five distinct candidates when possible. Candidate lanes must differ materially in mechanism, response, or representation. For each, record:

- hypothesis;
- prior evidence;
- negative evidence;
- novelty;
- exact data readiness;
- sample size;
- likely operational role;
- expected information gain;
- plausible MAE lift range;
- failure modes;
- execution cost;
- downstream value;
- timestamp risk;
- research-overfitting risk.

Run the novelty audit against all prior specifications and conclusions.

### D. Red-team and select one

Reject candidates that:

- duplicate a completed lane without a diagnosed refinement;
- require unproven available-at time for promotion;
- depend on target-day information;
- use 2024+ confirmation;
- cannot reproduce the baseline on identical rows;
- have insufficient support;
- leave scientific implementation choices unspecified;
- are unlikely to remain meaningful after forecast archive backfill;
- chase a tiny cell found after outcome inspection;
- merely swap generic model classes.

Select the candidate with the strongest combined value of information and deployable upside after red-team review. Save the ranked queue; do not discard reserve candidates.

### E. Pre-register

Create a complete `experiment_spec.json` using the schema. Fix:

- primary response;
- features and formulas;
- stations;
- cutoff and source eligibility;
- frame;
- baseline;
- folds;
- nested selection;
- parameter budget;
- metrics;
- slices;
- sample rules;
- acceptance gates;
- no-harm gates;
- rejection conditions;
- output artifacts.

Set the state to `EXECUTING` and record the proposed experiment ID and specification hash.

### F. Invoke the Executor

Explicitly invoke `$hkg-t24-experiment-executor` with:

- repository root;
- experiment specification path;
- canonical experiments root;
- instruction to create a new folder without overwriting;
- instruction to complete a rejection folder if eligibility fails;
- instruction to run the folder validator.

Do not ask the Executor to invent missing scientific decisions. If the specification is incomplete, return to pre-registration.

### G. Ingest the result

After the Executor returns:

1. locate the exact new folder;
2. run `validate_experiment_folder.py --strict`;
3. read all saved artifacts;
4. verify identical-row metrics;
5. verify no 2024+ rows;
6. verify leakage status;
7. compare against the appropriate frame champion;
8. classify outcome;
9. update all cumulative registries;
10. close the open experiment.

### H. Decide whether to repeat

- If candidate MAE > 0.45 °C, return to `DISCOVERY` and generate the next lane from the enlarged corpus.
- If candidate MAE <= 0.45 °C but eligibility, stability, or anti-overfit gates fail, do not count it. Record the failure and continue.
- If a valid development candidate reaches <= 0.45 °C, freeze it and enter `DEVELOPMENT_GATE_REACHED`.
- If the experiment is rejected or blocked, ingest the reason and choose a new admissible lane unless it is a global blocker.
- If execution fails from a repairable code defect, create a logged repair/rerun; do not reinterpret it as negative science.

## Champion rules

There is no single universal champion unless all candidates share one canonical frame. Maintain champions by:

- frame ID;
- anchor availability;
- response role;
- deployability class;
- date range;
- source coverage.

A candidate changes a champion only if:

- target definition matches;
- row universe or common-row comparison is valid;
- no confirmation rows were used;
- leakage audit passed;
- metric is recomputed from saved row-level predictions;
- candidate beats the current champion under predeclared metric;
- stability and no-harm gates pass.

## Avoiding autonomous overfitting

The loop must not endlessly vary thresholds against the same outer predictions. Use:

- pre-registered parameter budgets;
- nested temporal selection;
- fresh temporal folds when available;
- late-window holdbacks during development;
- grouped experiment families;
- attempt counts in the decision log;
- stronger promotion thresholds after broad searches;
- freeze-on-gate behavior.

When the next proposal differs only cosmetically from recent failures, the Director must change mechanism, representation, response, or data source.

## Blocked-state report

If no admissible iteration can proceed, `blockers.md` must identify:

- exact blocker;
- affected experiment families;
- work that remains possible without it;
- exact file, timestamp proof, archive backfill, station metadata, or engineering task needed;
- acceptance criterion for unlock;
- whether the blocker is local or global.

“Need more data” is not sufficient.

## Confirmation

The loop cannot open 2024+ automatically. After a development gate:

1. freeze code, model, features, transformations, thresholds, calibration, and environment;
2. write hashes and a one-time confirmation specification;
3. wait for explicit owner authorization;
4. run confirmation once;
5. report pass or fail without retuning.

A failed confirmation is not recycled into development labels.
