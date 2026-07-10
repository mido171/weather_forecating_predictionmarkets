# 0008 Two-day GFS/GEFS/radar structured delivery

Status: `complete`. Window: 2026-07-06 00Z to 2026-07-08 00Z, end-exclusive.

## Result

| Output family | Rows |
|---|---:|
| Model request/index manifests | 272 each |
| HKO model point features | 272 |
| HKG bbox summary features | 3,840 |
| Radar manifest/image features | 240 each |
| Attribute catalog | 163 |
| Source-issue glue | 512 |

The run produced compact structured tables and retained no raw payloads.

## As-of and radar caveat

GFS/GEFS kept issue, valid, and availability-proxy clocks. Radar came from an
HKUST ENVF historical display of HKO imagery, not native exact-vintage HKO
metadata. Its observed time plus 30 minutes was used conservatively and the
source remains a proxy.

## Reproduce and evidence

```powershell
.\.venv\Scripts\python.exe scripts\build_last2_gfs_gefs_radar_structured_delivery.py --help
```

The `normalized/` CSV/JSON tables and `STATUS.yaml` are the retained evidence.
Historical references to a removed SQL proposal are non-current provenance.
