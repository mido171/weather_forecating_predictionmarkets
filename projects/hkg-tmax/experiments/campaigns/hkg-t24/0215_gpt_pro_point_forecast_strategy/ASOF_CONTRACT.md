# As-Of Contract

- Target dates are `2000-01-02` through `2023-12-31` only.
- Confirmation/locked rows beginning `2024-01-01` are excluded.
- Production cutoffs evaluated: `17:00, 18:00, 21:00, 23:59` HKT on T-1.
- Forecast archive rows are usable only when `issue_at_hkt <= asof_cutoff_hkt`, `product_type='local'`, `row_quality_status='usable_local_minmax'`, and `target_issue_lead_days=1`.
- Target-history features use only T-2 and older.
- HKO daily climate features use only T-2 and older.
- Residual climatology, grouped residual shrinkage, and analog residuals are built from prior calendar years only inside each cutoff family.
- Lead-0 and external-source diagnostics are diagnostic-only and cannot be selected.
