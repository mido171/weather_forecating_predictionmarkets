# MAE 0.45 Success and Anti-Overfit Protocol

## Objective

The aspirational objective is a verified MAE of 0.45 °C or better. This protocol defines what counts and prevents the sequential research loop from optimizing itself into a false result.

## Scores that do not count

Do not count:

- training or in-sample MAE;
- randomly shuffled cross-validation;
- a favorable subset selected after scoring;
- a frame with missing hard periods presented as universal;
- candidate and baseline on different rows;
- diagnostic-only or timestamp-blocked predictors;
- target-day leakage;
- future-fitted preprocessing;
- 2024+ rows used repeatedly during development;
- the best score from an undocumented model/feature search;
- a score not reproducible from saved OOF predictions;
- a narrow regime specialist's MAE as full-system MAE.

## Development gate requirements

A candidate may enter `DEVELOPMENT_GATE_REACHED` only if:

1. candidate MAE <= 0.45 °C on a predeclared canonical development frame;
2. all rows are before 2024-01-01;
3. leakage audit is `PASS`;
4. candidate and baseline use identical rows;
5. predictions are genuinely walk-forward/out of fold;
6. all preprocessing and selection are prior/fold-local;
7. the exact score reproduces from saved predictions;
8. coverage is sufficiently representative and documented;
9. no single year or fold dominates the gain;
10. seasonal/source/year stability gates pass;
11. p90/p95 and signed tails do not materially worsen;
12. the candidate beats the relevant simple baselines;
13. research degrees of freedom are disclosed;
14. code, environment, data manifest, feature list, and hashes are frozen.

The candidate should also have a credible operational path at the real cutoff.

## Frame-specific language

Always state:

- frame ID;
- rows;
- date range;
- source coverage;
- anchor availability;
- exclusions;
- baseline.

Use `DEVELOPMENT_0P45_ON_<FRAME_ID>` rather than a universal claim when coverage is limited.

## Freeze package

When the gate is reached, create:

- frozen source code archive/hash;
- model artifact/hash;
- feature whitelist;
- data source contract;
- cutoff contract;
- exact environment lock;
- calibration and thresholds;
- expected input schema;
- one-time confirmation specification;
- no-change declaration.

No additional development is performed on that candidate before confirmation.

## Confirmation

2024+ remains sealed until explicit owner authorization. Confirmation is run once using the frozen system. A pass requires:

- MAE <= 0.45 °C on the declared confirmation frame;
- complete timestamp eligibility;
- no post-freeze modifications;
- acceptable tail/calibration behavior;
- transparent coverage.

A failure is reported. Do not tune using confirmation and rerun it as if untouched.

## Sequential multiple-testing control

The program maintains:

- experiment attempt count;
- model-family attempt count;
- candidate queue history;
- negative results;
- late-window development reserve when available;
- stronger evidence requirements for tiny gains after broad search.

A candidate discovered through a very broad screen requires a dedicated confirmatory development experiment.

## Honest outcome labels

Allowed:

- `DEVELOPMENT_GATE_REACHED`
- `DEVELOPMENT_0P45_ON_LIMITED_FRAME`
- `AWAITING_CONFIRMATION_AUTHORIZATION`
- `CONFIRMED_0P45_OR_BETTER`
- `CONFIRMATION_FAILED`
- `TARGET_NOT_YET_REACHED`

Never use “achieved” without the qualifier supported by evidence.

## If the target remains above 0.45

Continue with distinct admissible experiments. Diagnose:

- residual information not yet captured;
- source gaps;
- timestamp blockers;
- forecast archive continuity;
- data quality;
- irreducible-looking noise;
- missing operational NWP/cloud information;
- calibration versus point-error bottlenecks.

Do not lower standards or claim certainty that 0.45 is achievable.
