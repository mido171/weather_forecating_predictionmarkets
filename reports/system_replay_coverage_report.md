# HKG-T24 Strict System Replay Coverage Report

## strict_h24n_matrix_rows

rows=8765; All strict feature-matrix rows requested for replay.

## strict_h24n_usable_forecast_rows

rows=2191; Rows with a pre-distribution point forecast and passed leakage status.

## strict_h24n_scoreable_final_rows

rows=2191; Rows with a rounded final point forecast after distribution handling.

## official_anchor_available_rows

rows=0; Rows with strict pre-freeze official forecast max available in the feature matrix.

## official_anchor_unavailable_rows

rows=8765; Rows where strict H24N official max is unavailable and E0 cannot be active.

## target_memory_feature_available_rows

rows=8765; Rows with at least one repaired long-history target-memory feature.

## target_memory_fallback_rows

rows=2191; Rows where router fallback selected the strict target-memory expert.

## nwp_backed_rows

rows=1185; Rows with at least one strict GFS or GEFS feature present.

## no_forecast_rows

rows=6574; Rows where no strict base forecast was available before distribution.

## no_trade_rows

rows=8738; Rows flagged no-trade after final formula and distribution confidence handling.

## failed_closed_rows

rows=6574; Rows without a passed strict replay prediction.
