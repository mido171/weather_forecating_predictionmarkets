# KLGA experiment rules

The parent `AGENTS.md` applies. Predeclare the hypothesis, cutoff, availability
rules, baseline, split, metrics, resource budget, and failure criteria. Keep
compact code/manifests/summaries in Git and write large predictions, models,
plots, and logs under `KLGA_RUN_ROOT`. Never overwrite a completed result or
tune repeatedly on a locked holdout. Provider calls require explicit execution
and bounded scope even when initiated by an experiment script.
