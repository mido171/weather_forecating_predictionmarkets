# Static Context Inventory

Polymarket is excluded. No model features were built.

## Summary

- data root: `C:\hkg_tmax_data`
- retrieval window: `2026-06-19T09:00:42Z` to `2026-06-19T09:06:17Z`
- successful static-context retrievals: `60`
- failed static-context retrievals in this batch: `0`
- unique static-context hashes: `60`
- archived static-context bytes: `215,229,122`
- immutable storage: content-addressed raw objects with metadata sidecars and append-only retrieval-ledger rows

## Downloaded Source Groups

| Group | Source IDs | Coverage |
|---|---|---|
| LandsD/CSDI terrain | `landsd_whole_hk_dtm_5m_asc_zip`, `csdi_landsd_dtm_*`, `data_gov_hk_landsd_dtm_dataset_page`, `landsd_open_data_geospatial_page` | Official Hong Kong-wide 5 m DTM direct ASC zip, CSDI GeoTIFF package, CSDI ISO metadata, dataset pages, and three supporting PDFs |
| LandsD topographic/geocommunity | `landsd_topographic_*`, `landsd_land_boundary_ic1000_revision_csv`, `landsd_georeference_ig1000_revision_csv`, `landsd_igeocom_*` | iB 1:50k/1:100k/1:200k GML packages, topographic GeoTIFF maps, i-Series revision CSVs, and iGeoCommunity CSV/GeoJSON packages |
| PlanD/CSDI land utilization | `csdi_pland_luhk_*`, `data_gov_hk_pland_luhk_*`, `pland_luhk_*`, `pland_land_utilization_page` | LUHK 10 m raster dataset pages, ISO metadata, and GeoTIFF packages for 2018-2024; English LUHK statistics for 2022-2024; English data-description CSVs for 2023-2024 |
| PlanD urban morphology index | `pland_3d_photo_realistic_grid_index_*`, `pland_open_data_page` | 3D photo-realistic model tile index CSVs for Cesium, OBJ, and OSGB; bulk 3D tile payloads were not downloaded |

## Remaining Static Work

- Parse archived DTM/topographic/LUHK/iGeoCom packages into source-native bronze metadata.
- Generate versioned station-to-station distance/elevation/bearing matrices.
- Generate deterministic solar geometry, sunrise/sunset, solar elevation/azimuth, and day-length tables.
- Generate terrain/slope/aspect and coastline/land-water station context from archived official sources.
- Decide whether full PlanD 3D model tile payloads are necessary; tile indexes are archived, but bulk tiles need byte-budget and source-use approval.
