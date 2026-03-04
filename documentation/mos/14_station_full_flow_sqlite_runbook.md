# 14 - Station Full Flow SQLite Runbook

## Prerequisites
1. Python environment with project dependencies installed.
2. Network access to:
   - NCEI daily summaries
   - IEM MOS endpoint
   - Kalshi API endpoints used by downloader scripts
3. Write access to `D:\Ahmed\data`.

## Canonical Read Policy (Mandatory)
1. MOS/NWS canonical reads must come from SQLite under `D:\Ahmed\data\sqlite`.
2. Station SQLite paths are the authoritative source for truth/guidance retrieval in this flow:
   - `D:\Ahmed\data\sqlite\NWS\<STATION>\...`
   - `D:\Ahmed\data\sqlite\MOS\<STATION>\...`
3. Compatibility CSVs under `D:\Ahmed\data\kalshi\training_data\...` are derived outputs for downstream script compatibility only.

## Core Command
```powershell
python tools/station_flow/run_station_full_flow.py ^
  --station-id KXHIGHTATL ^
  --nws-code USW00013874 ^
  --log-level INFO
```

## Common Variants
### Explicit station override when Kalshi metadata is incomplete
```powershell
python tools/station_flow/run_station_full_flow.py ^
  --station-id KXHIGHTATL ^
  --nws-code USW00013874 ^
  --mos-station-id KATL ^
  --station-zoneid America/New_York
```

### With live inference for a target day
```powershell
python tools/station_flow/run_station_full_flow.py ^
  --station-id KXHIGHTATL ^
  --nws-code USW00013874 ^
  --target-date 2026-03-04
```

### Cojoined backtest (current station + others)
```powershell
python tools/station_flow/run_station_full_flow.py ^
  --station-id KXHIGHTATL ^
  --nws-code USW00013874 ^
  --run-cojoined ^
  --cojoined-stations KNYC,KMIA
```

## Phase-by-Phase Outputs
### Phase 0
1. `D:\Ahmed\data\runs\station_full_flow\<series>\<timestamp>\resolved_inputs.json`

### Phase 1
1. NWS SQLite:
`D:\Ahmed\data\sqlite\NWS\<STATION>\<STATION>_nws_truth_....sqlite`
2. MOS SQLite:
`D:\Ahmed\data\sqlite\MOS\<STATION>\<STATION>_mos_....sqlite`
3. Canonical MOS/NWS reads in this workflow are from these SQLite DBs.
4. Compatibility truth:
`D:\Ahmed\data\kalshi\training_data\02_truth\<STATION>_settled_tmax_2002_2026.csv`
5. Compatibility MOS:
`D:\Ahmed\data\kalshi\training_data\04_mos\archive_merged\<STATION>_mos_archive_2002_2026.csv.gz`
6. Kalshi minute history:
`D:\Ahmed\data\kalshi\kalshi_history\<series>_<start>_to_<end>\`

### Phase 2
1. Training outputs:
`D:\Ahmed\data\kalshi\Experiments\MOS_<STATION>\...`
2. Live bundle export:
`D:\Ahmed\data\kalshi\Experiments\MOS_<STATION>\03_blends\blend_12\live_model_bundle_v2_<date>\`

### Phase 3
1. Parity smoke report:
`<run_root>\phase3_parity_smoke\inference_report.json`
2. Required checks:
   - runtime equals policy runtime (`T-1 12:00Z`)
   - leakage guardrails pass

### Phase 4
Optional live inference report under:
`D:\Ahmed\data\live\mos_quantile_live_inference\...`

### Phase 5
Backtest summary/sanity under station experiment backtest dir.

### Phase 6/7
1. `run_manifest.json`
2. `run_notes.json`

## Failure Modes and Recovery
1. Kalshi metadata missing station derivation:
   - Provide `--mos-station-id` and `--station-zoneid`.
2. NWS/MOS download interruptions:
   - Re-run with `--resume`.
3. Parity gate failure:
   - Treat as hard stop; do not use bundle for live.
4. Missing cojoined station artifacts:
   - Ensure pre-existing prediction/truth/kalshi paths exist for additional stations.

## Leakage and Runtime Controls
1. Runtime policy is fixed to `T-1 12:00Z`.
2. Parity inference validates policy-runtime compliance.
3. No runtime fallback accepted in parity gate path.

## Backtest Claim Hygiene
Every claim should include:
1. Entry gate semantics.
2. Stake rule.
3. Exact summary JSON path.
4. Exact sanity JSON path.
