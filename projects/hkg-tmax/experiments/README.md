# HKG Tmax experiment workspace

Experiments are grouped by research campaign. Every new governed experiment is
created from `templates/standard`, recorded in `registry/registry.yaml`, and
indexed in the project-level `EXPERIMENT_INDEX.md`.

- `campaigns/hkg-tmax`: core HKG Tmax data and forecast experiments
- `campaigns/hkg-t24`: T-24 implementation and acquisition work
- `campaigns/residual-modeling`: residual and routing model research
- `campaigns/probability`: probability and bucket calibration research
- `campaigns/market-edges`: market-comparison research; never order execution
- `campaigns/general`: default destination for newly created experiments
- `registry`: machine-readable allocation registry
- `templates/standard`: mandatory experiment contract

Large outputs, models, raw data, logs, and temporary files belong under the
configured external run/data roots. Only compact protocols, manifests,
summaries, metrics, and conclusions belong in Git.
