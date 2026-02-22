# External Data Expansion JIRA Pack (WFPM-EPIC-EXTDATA)

This zip contains one epic doc and seven ultra-detailed Jira tickets.

## Files

- `EPIC_WFPM-EPIC-EXTDATA.md`  
  Master epic background (exactly 1500 words).

### Data ingestion Jiras (fetch → parse → persist)

1. `WFPM-101_IEM_ASOS_RING.md` — ASOS/METAR airport ring from IEM `asos.py`
2. `WFPM-102_NDBC_STDMET.md` — NDBC/NOS marine observations (VAKF1, PEGF1)
3. `WFPM-103_FAWN_MESONET.md` — FAWN daily summaries + near-real-time last96
4. `WFPM-104_IGRA_DERIVED.md` — IGRA v2.2 derived sounding parameters (USM00072202)
5. `WFPM-105_OISST_V2_1.md` — OISST v2.1 SST/anomaly near Miami (compact box summary)

### Model / feature engineering Jiras

6. `WFPM-201_EXTERNAL_FEATURES.md` — leakage-free “as-of T-1 12Z” feature engineering
7. `WFPM-202_TRAIN_EVAL_ABLATION_CALIBRATION.md` — retrain + ablation + calibration + report

## Key principles baked into the tickets

- Strict “as-of” cutoffs to prevent leakage / optimistic bias.
- Additional locations are incorporated as features in the KMIA vector (not as extra rows).
- All tasks are written so Codex (no browsing) has the exact endpoints, formats, and parsing rules needed.
