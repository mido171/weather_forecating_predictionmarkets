# T05 — Canonical Location, Station, and Geospatial Registry

## Assignment

**Phase:** A Foundation  
**Required dependencies:** T02, T04  
**Bookkeeping folder suffix:** `canonical_location_station_geospatial_registry`

## Mission

Build a date-effective registry for the HKO target, all current HKO stations, all 36 ISD stations, ARWF stations, and designed inland/coastal/marine/synoptic reference points.

## Why this task exists

All station and NWP spatial features depend on correct coordinates, identity, elevation, exposure and historical metadata.

## Non-negotiable controls for this task

- Target date T is forecast at 15:00 HKT on T−1 under cutoff contract `hkg_t24_1500hkt_v1`, unless T01 formally versions an existing different contract.
- No value enters strict scoring unless availability before cutoff is proven.
- GribStream `asOf` alone is not proof of historical API availability.
- Store UTC as timezone-aware canonical time; derive HKT explicitly.
- Preserve raw data and lineage; clean into normalized tables and quarantine invalid rows.
- Keep 2024+ outcomes sealed unless this task is T36 and the frozen protocol authorizes access.
- Never use target T, same-row residuals, realized error flags, post-cutoff revisions, full-history preprocessing, or in-sample expert predictions.
- Candidate and baseline are compared on identical rows.


## Required inputs and prerequisites

1. ISD station dossier
2. static geospatial inventory
3. source station metadata
4. HKO target metadata
5. ARWF station payloads when available

Codex must verify every input path/table/version from T00/T02 manifests. Do not invent a path, table, station, selector, or credential. When a required dependency is absent, complete all independent implementation, create a blocker record, and stop before making unsupported claims.

## Exact implementation steps

1. Resolve station IDs to authoritative names and date-effective coordinates.
2. Reject impossible row-level coordinates and use authoritative station history.
3. Calculate distance/bearing to HKO, elevation difference, coast distance, terrain, urban/coastal/inland/island/hill roles.
4. Define HKO target coordinate and reference point sets.
5. Define local and synoptic grid domains and upwind candidate points.
6. Do not fabricate unknown station identities; mark unresolved.
7. Version the registry and hash all geospatial inputs.

## Database/code objects that must exist or be updated

1. catalog.location
2. catalog.station
3. catalog.station_metadata_history
4. catalog.location_group

## Required task-folder artifacts

In addition to the global folder contract, create:

1. location_registry.csv
2. station_dossier_complete.csv
3. location_groups.csv
4. local_domain.geojson
5. synoptic_domain.geojson
6. unresolved_station_blockers.csv

`README.md` must explain the hypothesis/mission and exact implementation. `RESULTS.md` must present executed results and comparisons. `CONCLUSION.md` must state pass, reject, blocked, diagnostic-only, shadow, or promoted status and the exact downstream consequence.

## Mandatory tests and audits

1. Coordinate bounds
2. duplicate station identity
3. distance/bearing recomputation
4. date-effective metadata tests

## Acceptance criteria

1. Every station used by any feature is registered
2. All NWP coordinates have stable codes
3. No impossible coordinates remain in clean views

The task may not be marked complete until every applicable criterion is demonstrated by a file, query result, test, checksum, or executed command.

## Rejection and blocker behavior

1. Unresolved station metadata blocks physical interpretation but not raw storage

A rejected or blocked task still creates the complete bookkeeping folder. Do not silently weaken the specification, reduce source coverage, change the cutoff, open sealed outcomes, or substitute retrospective availability.

## Handoff to downstream tasks

Write `handoff_manifest.json` containing:

```json
{
  "task_id": "T05",
  "status": "passed|rejected|blocked|partial",
  "git_commit": "...",
  "database_migration_version": "...",
  "input_manifest_sha256": "...",
  "output_manifest_sha256": "...",
  "created_tables_or_views": [],
  "created_files": [],
  "open_blockers": [],
  "downstream_ready": true
}
```

Every path in the handoff must be repository-relative and every listed artifact must exist.
