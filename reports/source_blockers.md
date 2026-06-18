# Source Blockers

| Source | Status | Blocker | Evidence | Next action |
|---|---|---|---|---|
| hko_daily_extract_operational | SUPERSEDED_POLLING_FAMILY_CLOSED | template adapter must bind yyyymm from operational date | reports/polling_loop_postmortem.md | implement template-bound once-daily collector without experiment creation |
| hko_daily_climate_elements | DISCOVERY_REQUIRED | official element identifiers must be discovered from HKO documentation, not guessed | reset instruction section 5A | archive HKO documentation and enumerate official dataType identifiers |
| hko_latest_1min_temperature | IMPLEMENTED_INITIAL_FETCH | historical minute backfill unknown | official HKO open-data endpoint in config/data_sources.yaml | parse every station row into bronze and continue prospective collection |
| hko_since_midnight_maxmin | IMPLEMENTED_INITIAL_FETCH | historical intraday max/min backfill unknown | official HKO open-data endpoint in config/data_sources.yaml | parse every station row into bronze and continue prospective collection |
| hko_live_humidity_pressure_wind_rain_radiation | DISCOVERY_REQUIRED | exact official resource URLs and schemas must be resolved | reset instruction section 5C | enumerate DATA.GOV.HK resources and add one adapter per feed |
| operational_nwp_gfs_gefs_ecmwf_icon | NOT_STARTED | server-side subset definitions and byte budget must be finalized before large backfill | reset instruction section 5E | implement NOAA GFS subset adapter first with cycle/member/lead manifest |
| king_park_upper_air_and_regional_soundings | DISCOVERY_REQUIRED | exact HKO or IGRA station identifiers and access URLs must be confirmed | reset instruction section 5F | resolve station IDs and implement public archive backfill |
| radar_nowcast_lightning | DISCOVERY_REQUIRED | product endpoints, color scales and georeferencing must be documented | reset instruction section 5G | archive product pages and implement first radar image collector |
| satellite_cloud_aerosol | NOT_STARTED | lawful provider and product subset must be selected | reset instruction section 5H | document terms for Himawari and HKO imagery before acquisition |
| tropical_cyclone_monsoon_synoptic | DISCOVERY_REQUIRED | exact operational track/advisory endpoints must be confirmed | reset instruction section 5I | split operational advisories from retrospective best-track datasets in catalog |
| marine_ocean_surface_state | NOT_STARTED | source selection and issue-time contracts required | reset instruction section 5J | select HKO local sea temperature and NOAA OISST/OSTIA lawful products |
| reanalysis_mechanism_datasets | CREDENTIAL_BLOCKED | CDS credentials and exact request subsets required | reset instruction section 5K and .env.example CDS fields | configure CDS credentials and write subset requests; mark RETROSPECTIVE_ONLY forever |
| static_geospatial_deterministic_context | NOT_STARTED | source selection and licensing review required | reset instruction section 5L | generate solar geometry locally and select official terrain/coastline sources |
| frontier_context_inventory | DEFERRED_WITH_REASON | P2 must not delay incomplete P0 acquisition | reset instruction section 5M | revisit after P0 and P1 source families have active acquisition paths |
