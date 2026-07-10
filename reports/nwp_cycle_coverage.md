# NWP Cycle Coverage

- data root: `C:\hkg_tmax_data`
- generated_at_utc: `2026-06-20T01:01:40.281396Z`

| Product | Successful files | Actual bytes | Policy status | Historical period | Blocker |
|---|---:|---:|---|---|---|
| GFS | 80 | 2920872 | partial | earliest official NCEI/NOMADS/cloud archive available through present | historical multi-year archive is byte_budget_required until estimator and date-range chunking are implemented |
| GEFS | 2640 | 22348482 | partial | GEFSv12 2000-2019 plus operational GEFS recoverable from official archives | GEFSv12 reforecast and multi-year operational history require byte_budget_required implementation |
| ECMWF IFS/ENS/AIFS | 3 | 320847 | credential_blocked | TIGGE from 2006-10 where accessible; open-data rolling current prospectively | historical/TIGGE terms and request tooling not activated; live open-data subset adapter still pending |
| DWD ICON/ICON-EPS | 6 | 1716 | byte_budget_required | official DWD archives where available plus TIGGE for ensemble periods | no server-side spatial subset verified; exact byte/retention plan required before bulk |
