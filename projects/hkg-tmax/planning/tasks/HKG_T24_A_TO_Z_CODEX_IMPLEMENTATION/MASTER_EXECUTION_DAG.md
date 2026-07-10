# Master Execution DAG

Tasks must execute only after all listed dependencies have passed. Independent branches may run in parallel after their prerequisites.

## A Foundation
- **T00 — Repository, Database, and Contract Preflight**; dependencies: none
- **T01 — Canonical T−24 Time and Availability Contract**; dependencies: T00
- **T02 — Full Current Data and Experiment Census Reconciliation**; dependencies: T00, T01
- **T03 — GribStream Catalog, Coverage, Licence, and Quota Audit**; dependencies: T00, T01
- **T04 — NWP Database, Object Storage, and Lineage Migrations**; dependencies: T00, T01, T02, T03
- **T05 — Canonical Location, Station, and Geospatial Registry**; dependencies: T02, T04

## B Acquisition
- **T06 — Resumable GribStream Runs Client and Raw Landing Zone**; dependencies: T03, T04, T05
- **T07 — GFS Complete HKG-Relevant Backfill**; dependencies: T06
- **T08 — GEFS Atmospheric Members and Mean Complete Backfill**; dependencies: T06
- **T09 — IFS Deterministic and Ensemble Backfill**; dependencies: T06
- **T10 — Global AI Weather Model Backfill**; dependencies: T06
- **T11 — CWA WRF 15 km Urgent Prospective Collector**; dependencies: T06
- **T12 — Secondary Model Coverage Tests and Selective Acquisition**; dependencies: T03, T06
- **T13 — HKO ARWF Exact-Vintage Prospective Collector**; dependencies: T04, T05
- **T14 — Canonical Official HKO Anchor and Revision Store**; dependencies: T00, T01, T04
- **T15 — Clean and Normalize Every Existing Dataset and Station**; dependencies: T02, T04, T05
- **T16 — Historical Availability Proof and Eligibility Ledger**; dependencies: T01, T03, T07, T08, T09, T10, T11, T13, T14, T15
- **T17 — End-to-End Data Quality, Completeness, and Idempotency Gate**; dependencies: T07, T08, T09, T10, T11, T12, T13, T14, T15, T16

## C Feature Platform
- **T18 — Canonical Target-Date T−24 Snapshot Builder**; dependencies: T01, T14, T16, T17
- **T19 — Official Forecast, Revision, Text, and Trust Feature Family**; dependencies: T14, T18
- **T20 — Causal Long-History Target Memory and Climatology Feature Store**; dependencies: T01, T15, T18
- **T21 — All-Station Spatiotemporal and Graph Feature Store**; dependencies: T05, T15, T18
- **T22 — Deterministic NWP Trajectory, Vertical, and Spatial Feature Store**; dependencies: T16, T18
- **T23 — Ensemble Distribution and Uncertainty Feature Store**; dependencies: T08, T09, T10, T16, T18
- **T24 — Diagnostic Physics Teacher-to-Safe-Student Features**; dependencies: T15, T18, T21, T22
- **T25 — Causal Online Residual and Source Performance State Engine**; dependencies: T14, T18
- **T26 — Feature Registry, Eligibility API, and Automated Leakage Gate**; dependencies: T19, T20, T21, T22, T23, T24, T25

## D Modelling
- **T27 — Canonical Evaluation Frames and Baseline Ladder**; dependencies: T18, T26
- **T28 — Official, Target-Memory, and Station Family Experts**; dependencies: T27
- **T29 — Core GFS and GEFS Local MOS Experts**; dependencies: T22, T23, T27
- **T30 — IFS, AI, CWA WRF, and ARWF Short-History Challenger Experts**; dependencies: T09, T10, T11, T13, T27
- **T31 — Specialist Detector, Correction, and Benefit-Gate Framework**; dependencies: T27, T28, T29
- **T32 — Nested Walk-Forward OOF Prediction Factory**; dependencies: T28, T29, T31
- **T33 — Expected-Error Router with Static Priors, Dynamic Weights, and Abstention**; dependencies: T32
- **T34 — Distributional Calibration and Conditional-Median Point Forecast**; dependencies: T23, T33

## E Validation
- **T35 — Full System Integration, Ablation, and Championship Tournament**; dependencies: T33, T34
- **T36 — One-Time 2024/2025/2026 Sealed Validation and Confirmation**; dependencies: T35

## F Production
- **T37 — Live Daily Inference, Monitoring, and Model Registry**; dependencies: T35, T36
- **T38 — Probability and Market-Threshold Interface with No-Trade Gate**; dependencies: T34, T37
- **T39 — Final Audit, Handoff, and Controlled Research Continuation Loop**; dependencies: T36, T37

## Critical path

`T00 → T01/T02/T03 → T04/T05 → T06 → T07/T08 → T16/T17 → T18 → T22/T23 → T27 → T29 → T32 → T33 → T34 → T35 → T36 → T37 → T39`

CWA WRF and ARWF collectors must start as soon as T06/T05 prerequisites permit; do not wait for modelling tasks because their historical windows are short.