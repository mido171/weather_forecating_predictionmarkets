# HKG-T24-001 Feature Matrix Migration

## Status

PASS

## Actions

- `snapshot_feature_matrix_strict` was already a view; no physical migration needed.
- `snapshot_feature_matrix_proxy` was already a view; no physical migration needed.

## Final Contract

`model_features.feature_matrix` is the only physical matrix table; `snapshot_feature_matrix_strict` and `snapshot_feature_matrix_proxy` are compatibility views.
