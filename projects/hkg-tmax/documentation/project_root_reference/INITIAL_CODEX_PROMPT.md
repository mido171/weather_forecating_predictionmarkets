# Initial Codex Prompt

Operate this repository as a world-class, evidence-first quantitative weather research program.

Read `AGENTS.md`, `CODEX_START_HERE.md`, `FIRST_GOALS.md`, `MILESTONES.md`, and the complete `docs/` directory. Then execute the goals in dependency order.

The objective is to maximize leakage-free, out-of-sample probabilistic accuracy for the first-published HKO Daily Extract `Absolute Daily Max (deg. C)` at the selected pre-event cutoff, and later translate that forecast into exact Polymarket bucket probabilities and cost-aware execution decisions.

Before predictive modelling:

- prove the settlement target and historical-label parity;
- archive exact market rules and source publications;
- establish authentic point-in-time data vintages;
- define and freeze the forecast horizon;
- establish strong transparent baselines;
- build robust evaluation and uncertainty estimation.

Create a separate experiment directory for every hypothesis. Fill its hypothesis and protocol before inspecting holdout outcomes. Save all source references, raw hashes, configurations, code state, predictions, metrics, diagnostics, conclusions, and gate reviews. Never hide failed experiments.

Aggressively investigate physical, temporal, spatial, station-network, forecast-vintage, regime, and market-mechanics relationships. Be unusually creative, but demand unusually strong evidence. Complexity is not success. A result is accepted only when it beats the established baseline on point-in-time out-of-sample data, survives ablation and robustness tests, and passes leakage and reproducibility review.

Do not claim or imply guaranteed profit. Do not enable live trading until `docs/07_PRODUCTION_GATE.md` is fully satisfied.
