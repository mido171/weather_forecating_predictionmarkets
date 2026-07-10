# 10 — Station Universe and Coordinate Registry

This file defines the canonical station, coordinate, and pseudo-gridpoint universe. Codex must implement this as a versioned registry table and all acquisition jobs must read from it.

## 1. Purpose

The target market settles on KLGA, but the forecast system must use nearby observations and gridded model points to understand:

```text
local air mass,
marine influence,
sea-breeze risk,
urban heat-island structure,
inland-versus-coastal thermal gradient,
recent model bias,
regional heat-wave strength,
cloud/convective bust regimes.
```

The registry below is mandatory. Do not silently add or remove stations inside individual ingestion jobs. Any change to this file must create a new `station_registry_version`.

## 2. Station id conventions

| Field | Meaning |
|---|---|
| `station_id` | Canonical four-character station id where available, e.g. `KLGA`. |
| `iem_asos_id` | Three-character IEM ASOS id where IEM expects it, e.g. `LGA`. |
| `wunderground_station_id` | Wunderground station id, expected to be `KLGA` for airport stations unless the user's API requires a different id. |
| `mos_station_id` | Three-character MOS station suffix, e.g. `LGA` for `MAVLGA`. |
| `role` | `target`, `nearby_core`, `regional_context`, or `gridded_pseudo_point`. |
| `lat`, `lon` | WGS84 decimal degrees. Use these exact coordinates unless source-native station metadata must be preserved separately. |

## 3. Mandatory airport/station registry

Codex must load the following table exactly. Latitude/longitude values are operational coordinates for acquisition and feature work; source-native metadata may be stored separately but must not overwrite canonical coordinates without an explicit registry-version change.

| station_id | iem_asos_id | wunderground_station_id | mos_station_id | role | lat | lon | primary purpose |
|---|---|---|---|---|---:|---:|---|
| KLGA | LGA | KLGA | LGA | target | 40.77945 | -73.88027 | Market settlement station, LaGuardia Airport. |
| KNYC | NYC | KNYC | NYC | nearby_core | 40.77898 | -73.96925 | Central Park / Manhattan urban reference. |
| KJFK | JFK | KJFK | JFK | nearby_core | 40.63980 | -73.77890 | Marine/coastal Queens/Atlantic influence. |
| KEWR | EWR | KEWR | EWR | nearby_core | 40.69250 | -74.16870 | Newark/inland hot corridor reference. |
| KTEB | TEB | KTEB | TEB | nearby_core | 40.85899 | -74.05600 | North Jersey / lower-Hudson inland reference. |
| KHPN | HPN | KHPN | HPN | nearby_core | 41.06700 | -73.70760 | Northern suburban / inland gradient. |
| KISP | ISP | KISP | ISP | nearby_core | 40.79520 | -73.10020 | Long Island inland/coastal moderation reference. |
| KFRG | FRG | KFRG | FRG | nearby_core | 40.72880 | -73.41340 | Western Long Island / sea-breeze transition. |
| KBDR | BDR | KBDR | BDR | nearby_core | 41.16350 | -73.12620 | Connecticut coast / Sound influence. |
| KSWF | SWF | KSWF | SWF | regional_context | 41.50410 | -74.10480 | Hudson Valley hot/cool air-mass source. |
| KPOU | POU | KPOU | POU | regional_context | 41.62660 | -73.88420 | Mid-Hudson Valley air-mass reference. |
| KMMU | MMU | KMMU | MMU | regional_context | 40.79940 | -74.41490 | Inland New Jersey / terrain/heat gradient. |
| KCDW | CDW | KCDW | CDW | regional_context | 40.87520 | -74.28140 | North Jersey local gradient. |
| KPHL | PHL | KPHL | PHL | regional_context | 39.87190 | -75.24110 | Southwest corridor heat source. |
| KBOS | BOS | KBOS | BOS | regional_context | 42.36560 | -71.00960 | Northeast coastal/backdoor-front reference. |
| KDCA | DCA | KDCA | DCA | regional_context | 38.85120 | -77.04020 | Mid-Atlantic warm sector / upstream urban airport. |
| KBWI | BWI | KBWI | BWI | regional_context | 39.17540 | -76.66830 | Mid-Atlantic upstream air mass. |
| KALB | ALB | KALB | ALB | regional_context | 42.74720 | -73.79910 | Interior northeast/backdoor-front context. |
| KABE | ABE | KABE | ABE | regional_context | 40.65210 | -75.44080 | Inland Pennsylvania heat/cool-front context. |

## 4. Mandatory gridded pseudo-points

For GribStream and Open-Meteo point extraction, use this pseudo-point set. These are not settlement stations. They are designed to capture gradients around KLGA while controlling data cost.

| grid_point_id | role | lat | lon | purpose |
|---|---|---:|---:|---|
| GP_KLGA_EXACT | gridded_pseudo_point | 40.77945 | -73.88027 | Exact target coordinate. |
| GP_KLGA_NORTH | gridded_pseudo_point | 40.87945 | -73.88027 | North of KLGA; Sound/backdoor/easterly gradient. |
| GP_KLGA_SOUTH | gridded_pseudo_point | 40.67945 | -73.88027 | South of KLGA; Queens/Brooklyn influence. |
| GP_KLGA_EAST | gridded_pseudo_point | 40.77945 | -73.78027 | East/coastal Queens and marine influence. |
| GP_KLGA_WEST | gridded_pseudo_point | 40.77945 | -73.98027 | Manhattan/urban west reference. |
| GP_KLGA_NW_INLAND_NJ | gridded_pseudo_point | 40.86000 | -74.15000 | Inland west/northwest warm-source point. |
| GP_KLGA_SW_NEWARK_CORRIDOR | gridded_pseudo_point | 40.70000 | -74.17000 | Newark corridor heat-source point. |
| GP_KLGA_E_LONG_ISLAND | gridded_pseudo_point | 40.78000 | -73.40000 | Long Island sea-breeze penetration reference. |
| GP_KLGA_SOUND_WATER_PROXY | gridded_pseudo_point | 40.90000 | -73.80000 | Long Island Sound/water-side thermal proxy. |
| GP_KLGA_ATLANTIC_PROXY | gridded_pseudo_point | 40.60000 | -73.70000 | Atlantic/Jamaica Bay marine proxy. |

## 5. Cost-controlled GribStream extraction coordinate tiers

### 5.1 Tier A: minimum viable production pull

Use when costs are high or for first proof-of-value backfill.

```text
GP_KLGA_EXACT
GP_KLGA_NW_INLAND_NJ
GP_KLGA_E_LONG_ISLAND
GP_KLGA_ATLANTIC_PROXY
```

### 5.2 Tier B: recommended production pull

Use for normal training and live production.

```text
GP_KLGA_EXACT
GP_KLGA_NORTH
GP_KLGA_SOUTH
GP_KLGA_EAST
GP_KLGA_WEST
GP_KLGA_NW_INLAND_NJ
GP_KLGA_SW_NEWARK_CORRIDOR
GP_KLGA_E_LONG_ISLAND
GP_KLGA_SOUND_WATER_PROXY
GP_KLGA_ATLANTIC_PROXY
```

### 5.3 Tier C: expanded research pull

Use only if GribStream confirms acceptable bulk pricing. Generate a 5×5 lat/lon neighborhood around KLGA with spacing 0.05°:

```text
lat_offsets = [-0.10, -0.05, 0.00, 0.05, 0.10]
lon_offsets = [-0.10, -0.05, 0.00, 0.05, 0.10]
base = (40.77945, -73.88027)
```

Each generated point must have deterministic id:

```text
GP_KLGA_GRID_DLAT_{offset_lat:+.2f}_DLON_{offset_lon:+.2f}
```

Do not use Tier C by default. Tier C is an audition for additional skill and must be validated against Tier B.

## 6. Observation station tiers

Observation data from Wunderground/IEM is cheaper than gridded model extraction. Therefore, ingest all mandatory airport stations for actuals and observations.

| tier | stations | use |
|---|---|---|
| target | KLGA | settlement label and target features |
| nearby_core | KNYC, KJFK, KEWR, KTEB, KHPN, KISP, KFRG, KBDR | always used in features |
| regional_context | KSWF, KPOU, KMMU, KCDW, KPHL, KBOS, KDCA, KBWI, KALB, KABE | used in air-mass, gradient, frontal-regime features |

## 7. Derived station groups

Codex must define these station groups as constants for feature generation.

```python
TARGET_STATION = "KLGA"
NYC_CORE_STATIONS = ["KLGA", "KNYC", "KJFK", "KEWR", "KTEB"]
COASTAL_MARINE_STATIONS = ["KJFK", "KISP", "KFRG", "KBDR", "KBOS"]
INLAND_HOT_REFERENCE_STATIONS = ["KEWR", "KTEB", "KMMU", "KCDW", "KSWF", "KPOU", "KABE"]
UPSTREAM_SOUTHWEST_STATIONS = ["KPHL", "KDCA", "KBWI", "KABE"]
BACKDOOR_FRONT_STATIONS = ["KBOS", "KBDR", "KALB", "KHPN"]
LONG_ISLAND_SOUND_STATIONS = ["KLGA", "KBDR", "KHPN", "KISP", "KFRG"]
```

## 8. Required registry table schema

```text
CREATE TABLE station_registry (
    station_registry_version TEXT NOT NULL,
    station_id TEXT NOT NULL,
    iem_asos_id TEXT,
    wunderground_station_id TEXT,
    mos_station_id TEXT,
    grid_point_id TEXT,
    role TEXT NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    elevation_m DOUBLE PRECISION,
    source_native_metadata_json JSON,
    active_from_date DATE NOT NULL DEFAULT '1900-01-01',
    active_until_date DATE,
    notes TEXT,
    PRIMARY KEY (station_registry_version, station_id, COALESCE(grid_point_id,''))
);
```

Initial registry version:

```text
station_registry_version = "v2026_06_27_klga_core"
```

## 9. Acceptance tests

```text
[ ] Every station in the mandatory station registry exists exactly once.
[ ] `KLGA` is marked `target` and has Wunderground id `KLGA`, IEM id `LGA`, MOS id `LGA`.
[ ] All GribStream jobs read pseudo-points from this file, not from hard-coded local lists.
[ ] All Wunderground/IEM/MOS jobs map provider station ids back to canonical `station_id`.
[ ] Feature code can compute station groups from constants above.
[ ] Tier A, Tier B, and Tier C coordinate sets are deterministic and unit-tested.
```
