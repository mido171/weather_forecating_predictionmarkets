# Results

All three requested public sources returned provider-native payloads.

| Source | Result |
|---|---|
| GFS | Latest accessible filtered GRIB2 subset saved. |
| GEFS | Latest accessible control-member filtered GRIB2 subset saved. |
| Himawari-9 | Latest visible full-disk AHI `.DAT.bz2` object saved. |

See `artifacts/fetch_summary.json` for URLs, hashes, byte counts, issued/observed timestamps, and request details.

## Normalized Outputs

Normalized, readable outputs were generated at `2026-07-08T07:53:09.863891Z` under `normalized/`.

- `normalized/README.md` is the human entrypoint.
- `normalized/model_target_station_features.csv` gives nearest-HKO model features.
- `normalized/model_source_comparison_features.csv` gives GEFS-control minus GFS deltas.
- `normalized/hkg_bbox_grid_long_all_sources.csv` gives all cropped model grid rows in long form.
- `normalized/himawari_b13_header_summary.json` decodes Himawari B13 metadata and records that pixel calibration is still separate work.
