# 0005 Public GFS/GEFS/Himawari fetch smoke

Status: `fetch_smoke_pass`.

## Purpose and result

Prove direct public-provider fetchability without GribStream. The historical
run fetched provider-native payloads for GFS, GEFS control, and Himawari-9:

| Source | Timestamp | Bytes |
|---|---|---:|
| GFS | issued 2026-07-08 00Z, valid 2026-07-09 00Z | 4,062,413 |
| GEFS control | issued 2026-07-08 00Z, valid 2026-07-09 00Z | 3,515,940 |
| Himawari-9 B13 | observed 2026-07-08 06:20Z | 1,347,225 |

Normalized CSV/JSON outputs preserve model station/bbox features and Himawari
header metadata. Himawari pixel calibration remained separate work.

## Historical availability semantics

- GFS/GEFS: issue time plus a conservative publication buffer (historically
  six hours unless a provider audit proves tighter).
- Himawari: observed time plus latency and native file metadata, using the
  conservative later availability.

These are acquisition semantics, not authority to weaken the current H24N
contract.

## Reproduce safely

Inspect both CLIs before any bounded network run:

```powershell
.\.venv\Scripts\python.exe scripts\fetch_public_gfs_gefs_himawari_smoke.py --help
.\.venv\Scripts\python.exe scripts\normalize_public_gfs_gefs_himawari_smoke.py --help
```

## Evidence map

`STATUS.yaml` and the compact `normalized/` CSV/JSON files remain. Historical
raw payload references in the original prose are not current retained files.
`normalized/himawari_b13_decompressed_prefix_hex.txt` is diagnostic data, not
active documentation.
