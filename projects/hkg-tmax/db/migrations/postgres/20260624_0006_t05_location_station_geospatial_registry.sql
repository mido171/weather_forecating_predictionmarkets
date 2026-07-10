-- T05 canonical location, station, and geospatial registry.
-- PostgreSQL 15+. Additive and idempotent.

BEGIN;

CREATE SCHEMA IF NOT EXISTS catalog;
CREATE SCHEMA IF NOT EXISTS governance;

CREATE TABLE IF NOT EXISTS catalog.location (
    location_id bigserial PRIMARY KEY,
    location_code text NOT NULL UNIQUE,
    name text NOT NULL,
    latitude double precision NOT NULL CHECK (latitude BETWEEN -90 AND 90),
    longitude double precision NOT NULL CHECK (longitude BETWEEN -180 AND 180),
    elevation_m double precision,
    location_role text NOT NULL,
    valid_from date,
    valid_to date,
    metadata_source text NOT NULL,
    metadata_sha256 char(64) NOT NULL,
    created_at_utc timestamptz NOT NULL DEFAULT now(),
    CHECK (valid_to IS NULL OR valid_from IS NULL OR valid_to >= valid_from)
);

CREATE TABLE IF NOT EXISTS catalog.station (
    station_id bigserial PRIMARY KEY,
    station_code text NOT NULL UNIQUE,
    station_name text NOT NULL,
    network text NOT NULL,
    icao text,
    country_code text,
    location_id bigint REFERENCES catalog.location(location_id),
    station_role text NOT NULL,
    target_station boolean NOT NULL DEFAULT false,
    valid_from date,
    valid_to date,
    metadata_status text NOT NULL,
    source_uri text NOT NULL,
    source_sha256 char(64) NOT NULL,
    created_at_utc timestamptz NOT NULL DEFAULT now(),
    CHECK (valid_to IS NULL OR valid_from IS NULL OR valid_to >= valid_from)
);

CREATE TABLE IF NOT EXISTS catalog.station_metadata_history (
    station_metadata_history_id bigserial PRIMARY KEY,
    station_id bigint NOT NULL REFERENCES catalog.station(station_id),
    field_name text NOT NULL,
    old_value text,
    new_value text,
    valid_from date,
    valid_to date,
    evidence_uri text NOT NULL,
    metadata_sha256 char(64) NOT NULL,
    created_at_utc timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS catalog.location_group (
    group_code text PRIMARY KEY,
    group_name text NOT NULL,
    group_type text NOT NULL,
    description text NOT NULL,
    created_at_utc timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS catalog.location_group_member (
    group_code text NOT NULL REFERENCES catalog.location_group(group_code),
    location_id bigint NOT NULL REFERENCES catalog.location(location_id),
    member_role text NOT NULL,
    valid_from date,
    valid_to date,
    created_at_utc timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (group_code, location_id, member_role),
    CHECK (valid_to IS NULL OR valid_from IS NULL OR valid_to >= valid_from)
);

CREATE INDEX IF NOT EXISTS idx_location_lat_lon
    ON catalog.location (latitude, longitude);

CREATE INDEX IF NOT EXISTS idx_station_network
    ON catalog.station (network, station_code);

CREATE INDEX IF NOT EXISTS idx_location_group_member_location
    ON catalog.location_group_member (location_id);

INSERT INTO governance.schema_version (migration_version, description)
VALUES (
    '20260624_0006_t05_location_station_geospatial_registry',
    'T05 canonical date-effective location, station, metadata history, and geospatial group registry'
)
ON CONFLICT (migration_version) DO NOTHING;

COMMIT;
