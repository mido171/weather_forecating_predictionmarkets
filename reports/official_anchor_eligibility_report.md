# HKG-T24 Strict Official Anchor Eligibility Report

## Status

BLOCKED_STRICT_E0

## Strict H24N Rule

Official rows must have `issue_at_utc <= operational_freeze_utc`.

## Raw Source Coverage

usable_rows=157099; usable_dates=8765; date_range=2000-01-02..2023-12-31.

## Strict Eligibility

eligible_rows=0; eligible_dates=0; minimum_issue_minus_freeze_hours=1.5.

## Persisted Strict Features

official_feature_rows=8765; non_null_official__forecast_max_c=0.

## Blocker

Strict E0 remains unavailable until a source row is proven available by the operational freeze.
