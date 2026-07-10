-- Tactical H24N GribStream schema.
--
-- This migration replaces the broad row-per-parameter GribStream acquisition
-- surface with an exact-cycle, compact, wide-row storage contract for the
-- HKG Tmax H24N use case.

BEGIN;

CREATE SCHEMA IF NOT EXISTS nwp_tactical;
CREATE SCHEMA IF NOT EXISTS feature_store;
CREATE SCHEMA IF NOT EXISTS governance;

CREATE TABLE IF NOT EXISTS nwp_tactical.location_stencil (
    location_code text PRIMARY KEY,
    cutoff_id text NOT NULL DEFAULT 'H24N',
    location_role text NOT NULL,
    latitude double precision NOT NULL CHECK (latitude BETWEEN -90 AND 90),
    longitude double precision NOT NULL CHECK (longitude BETWEEN -180 AND 180),
    lat_offset_deg double precision NOT NULL,
    lon_offset_deg double precision NOT NULL,
    description text NOT NULL,
    created_at_utc timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS nwp_tactical.model_plan (
    dataset_code text PRIMARY KEY,
    priority text NOT NULL,
    stage text NOT NULL,
    endpoint text NOT NULL DEFAULT 'runs' CHECK (endpoint = 'runs'),
    archive_run_start_utc timestamptz,
    archive_run_end_utc timestamptz,
    target_date_start date,
    target_date_end date,
    exact_cycle_template text NOT NULL,
    min_lead_hours integer NOT NULL,
    max_lead_hours integer NOT NULL,
    expected_native_step_hours integer,
    expected_valid_steps_per_run integer,
    location_policy text NOT NULL CHECK (
        location_policy IN ('deterministic_12_point_stencil', 'hko_center_only', 'nbmoc_probe_3_point')
    ),
    member_policy text NOT NULL CHECK (
        member_policy IN ('deterministic', 'mean', 'members_0_30', 'members_0_50', 'prospective_latest_complete', 'probe')
    ),
    availability_grade text NOT NULL,
    promotion_status text NOT NULL,
    expected_wide_rows bigint,
    approximate_credits bigint,
    notes text NOT NULL DEFAULT '',
    updated_at_utc timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS nwp_tactical.variable_plan (
    dataset_code text NOT NULL REFERENCES nwp_tactical.model_plan(dataset_code),
    alias text NOT NULL,
    native_name text NOT NULL,
    native_level text NOT NULL,
    native_info text NOT NULL DEFAULT '',
    required boolean NOT NULL DEFAULT true,
    variable_role text NOT NULL,
    canonical_unit text,
    notes text NOT NULL DEFAULT '',
    PRIMARY KEY (dataset_code, alias)
);

CREATE TABLE IF NOT EXISTS nwp_tactical.acquisition_chunk (
    chunk_id text PRIMARY KEY,
    acquisition_version text NOT NULL DEFAULT 'tactical_h24n_v1',
    dataset_code text NOT NULL REFERENCES nwp_tactical.model_plan(dataset_code),
    endpoint text NOT NULL DEFAULT 'runs' CHECK (endpoint = 'runs'),
    time_selector text NOT NULL DEFAULT 'timesList' CHECK (time_selector = 'timesList'),
    run_times_utc timestamptz[] NOT NULL,
    min_lead_hours integer NOT NULL,
    max_lead_hours integer NOT NULL,
    location_policy text NOT NULL,
    variable_bundle_id text NOT NULL,
    member_policy text NOT NULL,
    members integer[],
    expected_rows integer NOT NULL CHECK (expected_rows >= 0),
    expected_credits integer NOT NULL CHECK (expected_credits >= 0),
    request_json jsonb NOT NULL,
    request_sha256 char(64) NOT NULL UNIQUE,
    status text NOT NULL CHECK (
        status IN ('planned', 'running', 'completed', 'completed_empty', 'failed', 'blocked', 'purged')
    ),
    raw_object_uri text,
    response_sha256 char(64),
    http_status integer,
    row_count integer,
    error_class text,
    error_message text,
    created_at_utc timestamptz NOT NULL DEFAULT now(),
    started_at_utc timestamptz,
    completed_at_utc timestamptz,
    CHECK (max_lead_hours >= min_lead_hours),
    CHECK (NOT (request_json ? 'forecastedFrom')),
    CHECK (NOT (request_json ? 'forecastedUntil')),
    CHECK (request_json ? 'timesList')
);

CREATE INDEX IF NOT EXISTS idx_tactical_chunk_dataset_status
    ON nwp_tactical.acquisition_chunk(dataset_code, status);

CREATE TABLE IF NOT EXISTS nwp_tactical.raw_response_object (
    response_object_id bigserial PRIMARY KEY,
    chunk_id text NOT NULL REFERENCES nwp_tactical.acquisition_chunk(chunk_id) ON DELETE CASCADE,
    object_uri text NOT NULL,
    byte_size bigint NOT NULL CHECK (byte_size >= 0),
    sha256 char(64) NOT NULL,
    content_type text NOT NULL,
    retrieved_at_utc timestamptz NOT NULL,
    row_count integer NOT NULL CHECK (row_count >= 0),
    UNIQUE (chunk_id, sha256)
);

CREATE TABLE IF NOT EXISTS nwp_tactical.forecast_wide (
    dataset_code text NOT NULL REFERENCES nwp_tactical.model_plan(dataset_code),
    acquisition_version text NOT NULL DEFAULT 'tactical_h24n_v1',
    target_date_hkt date NOT NULL,
    cutoff_id text NOT NULL DEFAULT 'H24N',
    run_time_utc timestamptz NOT NULL,
    valid_time_utc timestamptz NOT NULL,
    lead_hours numeric(6,2) NOT NULL,
    location_code text NOT NULL REFERENCES nwp_tactical.location_stencil(location_code),
    requested_latitude double precision NOT NULL,
    requested_longitude double precision NOT NULL,
    returned_latitude double precision,
    returned_longitude double precision,
    returned_grid_distance_km double precision,
    member_number integer NOT NULL DEFAULT 0,
    temperature_2m_k double precision,
    interval_tmax_2m_k double precision,
    dewpoint_2m_k double precision,
    relative_humidity_2m_pct double precision,
    u_wind_10m_mps double precision,
    v_wind_10m_mps double precision,
    mslp_pa double precision,
    low_cloud_pct double precision,
    accumulated_precip_kg_m2 double precision,
    downward_shortwave_w_m2 double precision,
    net_shortwave_w_m2 double precision,
    total_precip_m double precision,
    shortwave_down_j_m2 double precision,
    total_column_water_vapour_kg_m2 double precision,
    pwat_kg_m2 double precision,
    temperature_925_k double precision,
    temperature_850_k double precision,
    relative_humidity_700_pct double precision,
    geopotential_height_500_m double precision,
    raw_values_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb,
    source_response_object_id bigint REFERENCES nwp_tactical.raw_response_object(response_object_id),
    quality_status text NOT NULL DEFAULT 'raw_valid',
    created_at_utc timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (
        dataset_code,
        acquisition_version,
        run_time_utc,
        valid_time_utc,
        location_code,
        member_number
    ),
    CHECK (lead_hours >= 0)
);

CREATE INDEX IF NOT EXISTS idx_tactical_forecast_target_model
    ON nwp_tactical.forecast_wide(target_date_hkt, dataset_code, location_code);

CREATE TABLE IF NOT EXISTS nwp_tactical.validation_issue (
    validation_issue_id bigserial PRIMARY KEY,
    chunk_id text REFERENCES nwp_tactical.acquisition_chunk(chunk_id) ON DELETE CASCADE,
    dataset_code text,
    issue_class text NOT NULL,
    issue_severity text NOT NULL CHECK (issue_severity IN ('info', 'warning', 'error', 'fatal')),
    evidence_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at_utc timestamptz NOT NULL DEFAULT now()
);

CREATE OR REPLACE VIEW feature_store.h24n_tactical_nwp_feature_source AS
SELECT
    fw.*
FROM nwp_tactical.forecast_wide fw
JOIN nwp_tactical.model_plan mp ON mp.dataset_code = fw.dataset_code
WHERE fw.quality_status = 'raw_valid'
  AND mp.promotion_status IN ('primary_candidate', 'shadow_only', 'diagnostic_shadow', 'prospective_only');

INSERT INTO nwp_tactical.location_stencil (
    location_code, location_role, latitude, longitude, lat_offset_deg, lon_offset_deg, description
)
VALUES
    ('hko_center', 'target_grid', 22.301944, 114.174167, 0.00, 0.00, 'Hong Kong Observatory target grid point'),
    ('local_n', 'local_context', 22.551944, 114.174167, 0.25, 0.00, 'North local stencil point'),
    ('local_s', 'local_context', 22.051944, 114.174167, -0.25, 0.00, 'South local stencil point'),
    ('local_e', 'local_context', 22.301944, 114.424167, 0.00, 0.25, 'East local stencil point'),
    ('local_w', 'local_context', 22.301944, 113.924167, 0.00, -0.25, 'West local stencil point'),
    ('local_ne', 'local_context', 22.551944, 114.424167, 0.25, 0.25, 'Northeast local stencil point'),
    ('local_nw', 'local_context', 22.551944, 113.924167, 0.25, -0.25, 'Northwest local stencil point'),
    ('local_se', 'local_context', 22.051944, 114.424167, -0.25, 0.25, 'Southeast local stencil point'),
    ('local_sw', 'local_context', 22.051944, 113.924167, -0.25, -0.25, 'Southwest local stencil point'),
    ('inland_nw_far', 'far_context', 22.801944, 113.674167, 0.50, -0.50, 'Pearl River Delta inland heat point'),
    ('marine_s_far', 'far_context', 21.801944, 114.174167, -0.50, 0.00, 'South China Sea marine state point'),
    ('marine_e_far', 'far_context', 22.301944, 114.674167, 0.00, 0.50, 'Easterly marine inflow point')
ON CONFLICT (location_code) DO UPDATE SET
    location_role = EXCLUDED.location_role,
    latitude = EXCLUDED.latitude,
    longitude = EXCLUDED.longitude,
    lat_offset_deg = EXCLUDED.lat_offset_deg,
    lon_offset_deg = EXCLUDED.lon_offset_deg,
    description = EXCLUDED.description;

INSERT INTO nwp_tactical.model_plan (
    dataset_code, priority, stage, archive_run_start_utc, archive_run_end_utc,
    target_date_start, target_date_end, exact_cycle_template, min_lead_hours,
    max_lead_hours, expected_native_step_hours, expected_valid_steps_per_run,
    location_policy, member_policy, availability_grade, promotion_status,
    expected_wide_rows, approximate_credits, notes
)
VALUES
    ('gfs', 'P0', 'core_1', '2021-03-22T00:00:00Z', '2026-06-22T00:00:00Z', '2021-03-23', '2026-06-23', 'T-1 00:00:00Z', 15, 39, 1, 25, 'deterministic_12_point_stencil', 'deterministic', 'PROVIDER_SCHEDULE_CONSERVATIVE', 'primary_candidate', 575700, 623675, 'Main deterministic NWP trajectory and HKO MOS expert'),
    ('gefsatmosmean', 'P0', 'core_1', '2020-10-01T18:00:00Z', '2026-06-21T18:00:00Z', '2020-10-03', '2026-06-23', 'T-2 18:00:00Z', 24, 45, 3, 8, 'deterministic_12_point_stencil', 'mean', 'PROVIDER_SCHEDULE_CONSERVATIVE', 'primary_candidate', 200640, 133760, 'Low-volume ensemble mean context'),
    ('gefsatmos', 'P0', 'core_1', '2020-10-01T18:00:00Z', '2026-06-21T18:00:00Z', '2020-10-03', '2026-06-23', 'T-2 18:00:00Z', 24, 45, 3, 8, 'hko_center_only', 'members_0_30', 'PROVIDER_SCHEDULE_CONSERVATIVE', 'primary_candidate', 518320, 518320, 'GEFS full-member HKO Tmax distribution'),
    ('ifsoper', 'P1', 'core_2_sealed_short_history', '2024-02-28T18:00:00Z', '2026-06-21T18:00:00Z', '2024-03-01', '2026-06-23', 'T-2 18:00:00Z', 21, 45, 3, 9, 'deterministic_12_point_stencil', 'deterministic', 'PROVIDER_SCHEDULE_CONSERVATIVE', 'shadow_only', 91260, 91260, 'Independent ECMWF deterministic expert'),
    ('ifsenfo', 'P1', 'core_2_sealed_short_history', '2024-03-01T18:00:00Z', '2026-06-21T18:00:00Z', '2024-03-03', '2026-06-23', 'T-2 18:00:00Z', 24, 45, 3, 8, 'hko_center_only', 'members_0_50', 'PROVIDER_SCHEDULE_CONSERVATIVE', 'shadow_only', 343944, 343944, 'Independent ECMWF ensemble distribution'),
    ('cwawrf15', 'P0-live', 'live_collection_immediate', NULL, NULL, NULL, NULL, 'latest complete exact run first seen by T-1 06:45Z', 12, 42, 6, 5, 'deterministic_12_point_stencil', 'prospective_latest_complete', 'PROSPECTIVE_EXACT_FIRST_SEEN', 'prospective_only', NULL, NULL, 'Regional East Asia expert; rolling last-three-day historical window only'),
    ('aifsoper', 'optional', 'useful_if_cheap', '2025-02-25T18:00:00Z', '2026-06-21T18:00:00Z', NULL, NULL, 'T-2 18:00:00Z', 18, 42, 6, 5, 'deterministic_12_point_stencil', 'deterministic', 'PROVIDER_SCHEDULE_CONSERVATIVE', 'shadow_only', NULL, NULL, 'Deterministic AI challenger'),
    ('aifsenfo', 'optional', 'useful_if_cheap', '2025-07-02T18:00:00Z', '2026-06-21T18:00:00Z', NULL, NULL, 'T-2 18:00:00Z', 24, 42, 6, 4, 'hko_center_only', 'members_0_50', 'PROVIDER_SCHEDULE_CONSERVATIVE', 'shadow_only', NULL, NULL, 'AI ensemble challenger'),
    ('aigfssfc', 'optional', 'useful_if_cheap', '2026-04-16T18:00:00Z', '2026-06-21T18:00:00Z', NULL, NULL, 'T-2 18:00:00Z', 18, 42, 6, 5, 'deterministic_12_point_stencil', 'deterministic', 'PROVIDER_SCHEDULE_CONSERVATIVE', 'shadow_only', NULL, NULL, 'NOAA AI deterministic surface challenger'),
    ('aigfspres', 'optional', 'useful_if_cheap', '2026-04-16T18:00:00Z', '2026-06-21T18:00:00Z', NULL, NULL, 'T-2 18:00:00Z', 18, 42, 6, 5, 'deterministic_12_point_stencil', 'deterministic', 'PROVIDER_SCHEDULE_CONSERVATIVE', 'shadow_only', NULL, NULL, 'Selected NOAA AI pressure-level context'),
    ('aigefssfc', 'optional', 'useful_if_cheap', '2025-06-01T18:00:00Z', '2026-06-21T18:00:00Z', NULL, NULL, 'T-2 18:00:00Z', 24, 42, 6, 4, 'hko_center_only', 'members_0_30', 'PROVIDER_SCHEDULE_CONSERVATIVE', 'shadow_only', NULL, NULL, 'NOAA AI ensemble surface challenger'),
    ('graphcast', 'optional', 'useful_if_cheap', '2024-04-25T18:00:00Z', '2026-05-05T00:00:00Z', NULL, NULL, 'T-2 18:00:00Z', 18, 42, 6, 5, 'deterministic_12_point_stencil', 'deterministic', 'MODEL_RUN_TIME_PROXY_ONLY', 'diagnostic_shadow', NULL, NULL, 'Historical AI-versus-physics expert'),
    ('fourcastnetgfs', 'optional', 'useful_if_cheap', '2024-05-02T18:00:00Z', '2026-03-01T12:00:00Z', NULL, NULL, 'T-2 18:00:00Z', 18, 42, 6, 5, 'deterministic_12_point_stencil', 'deterministic', 'MODEL_RUN_TIME_PROXY_ONLY', 'diagnostic_shadow', NULL, NULL, 'Historical AI-versus-physics expert'),
    ('nbmoc', 'probe', 'coverage_probe', NULL, NULL, NULL, NULL, 'single exact probe run', 18, 42, NULL, NULL, 'nbmoc_probe_3_point', 'probe', 'MODEL_RUN_TIME_PROXY_ONLY', 'probe_only', NULL, NULL, 'Tiny marine coverage probe only; no full backfill authorized')
ON CONFLICT (dataset_code) DO UPDATE SET
    priority = EXCLUDED.priority,
    stage = EXCLUDED.stage,
    archive_run_start_utc = EXCLUDED.archive_run_start_utc,
    archive_run_end_utc = EXCLUDED.archive_run_end_utc,
    target_date_start = EXCLUDED.target_date_start,
    target_date_end = EXCLUDED.target_date_end,
    exact_cycle_template = EXCLUDED.exact_cycle_template,
    min_lead_hours = EXCLUDED.min_lead_hours,
    max_lead_hours = EXCLUDED.max_lead_hours,
    expected_native_step_hours = EXCLUDED.expected_native_step_hours,
    expected_valid_steps_per_run = EXCLUDED.expected_valid_steps_per_run,
    location_policy = EXCLUDED.location_policy,
    member_policy = EXCLUDED.member_policy,
    availability_grade = EXCLUDED.availability_grade,
    promotion_status = EXCLUDED.promotion_status,
    expected_wide_rows = EXCLUDED.expected_wide_rows,
    approximate_credits = EXCLUDED.approximate_credits,
    notes = EXCLUDED.notes,
    updated_at_utc = now();

INSERT INTO nwp_tactical.variable_plan (
    dataset_code, alias, native_name, native_level, native_info, required, variable_role, canonical_unit, notes
)
VALUES
    ('gfs', 'temperature_2m_k', 'TMP', '2 m above ground', '', true, 'temperature', 'K', ''),
    ('gfs', 'interval_tmax_2m_k', 'TMAX', '2 m above ground', '', true, 'temperature_max_interval', 'K', ''),
    ('gfs', 'dewpoint_2m_k', 'DPT', '2 m above ground', '', true, 'dewpoint', 'K', ''),
    ('gfs', 'u_wind_10m_mps', 'UGRD', '10 m above ground', '', true, 'wind_u', 'm s-1', ''),
    ('gfs', 'v_wind_10m_mps', 'VGRD', '10 m above ground', '', true, 'wind_v', 'm s-1', ''),
    ('gfs', 'mslp_pa', 'PRMSL', 'mean sea level', '', true, 'pressure', 'Pa', ''),
    ('gfs', 'low_cloud_pct', 'LCDC', 'low cloud layer', '', true, 'cloud', 'percent', ''),
    ('gfs', 'accumulated_precip_kg_m2', 'APCP', 'surface', '', true, 'accumulation', 'kg m-2', ''),
    ('gfs', 'downward_shortwave_w_m2', 'DSWRF', 'surface', '', true, 'radiation', 'W m-2', ''),
    ('gfs', 'temperature_925_k', 'TMP', '925 mb', '', true, 'temperature_pressure_level', 'K', ''),
    ('gfs', 'temperature_850_k', 'TMP', '850 mb', '', true, 'temperature_pressure_level', 'K', ''),
    ('gfs', 'relative_humidity_700_pct', 'RH', '700 mb', '', true, 'humidity_pressure_level', 'percent', ''),
    ('gfs', 'geopotential_height_500_m', 'HGT', '500 mb', '', true, 'height_pressure_level', 'm', ''),

    ('gefsatmosmean', 'temperature_2m_mean_k', 'TMP', '2 m above ground', 'ens mean', true, 'temperature', 'K', ''),
    ('gefsatmosmean', 'interval_tmax_mean_k', 'TMAX', '2 m above ground', 'ens mean', true, 'temperature_max_interval', 'K', ''),
    ('gefsatmosmean', 'dewpoint_2m_mean_k', 'DPT', '2 m above ground', 'ens mean', true, 'dewpoint', 'K', ''),
    ('gefsatmosmean', 'rh_2m_mean_pct', 'RH', '2 m above ground', 'ens mean', true, 'humidity', 'percent', ''),
    ('gefsatmosmean', 'u10_mean_mps', 'UGRD', '10 m above ground', 'ens mean', true, 'wind_u', 'm s-1', ''),
    ('gefsatmosmean', 'v10_mean_mps', 'VGRD', '10 m above ground', 'ens mean', true, 'wind_v', 'm s-1', ''),
    ('gefsatmosmean', 'mslp_mean_pa', 'PRMSL', 'mean sea level', 'ens mean', true, 'pressure', 'Pa', ''),
    ('gefsatmosmean', 'pwat_mean_kg_m2', 'PWAT', 'entire atmosphere (considered as a single layer)', 'ens mean', true, 'precipitable_water', 'kg m-2', ''),

    ('gefsatmos', 'member_interval_tmax_k', 'TMAX', '2 m above ground', '', true, 'member_temperature_max_interval', 'K', 'HKO center only, members 0-30'),

    ('ifsoper', 'temperature_2m_k', '2t', 'sfc', '', true, 'temperature', 'K', ''),
    ('ifsoper', 'dewpoint_2m_k', '2d', 'sfc', '', true, 'dewpoint', 'K', ''),
    ('ifsoper', 'u_wind_10m_mps', '10u', 'sfc', '', true, 'wind_u', 'm s-1', ''),
    ('ifsoper', 'v_wind_10m_mps', '10v', 'sfc', '', true, 'wind_v', 'm s-1', ''),
    ('ifsoper', 'mslp_pa', 'msl', 'sfc', '', true, 'pressure', 'Pa', ''),
    ('ifsoper', 'total_precip_m', 'tp', 'sfc', '', true, 'accumulation', 'm', ''),
    ('ifsoper', 'shortwave_down_j_m2', 'ssrd', 'sfc', '', true, 'radiation_accumulation', 'J m-2', ''),
    ('ifsoper', 'total_column_water_vapour_kg_m2', 'tcwv', 'sfc', '', true, 'precipitable_water', 'kg m-2', ''),
    ('ifsoper', 'temperature_925_k', 't', 'pl 925', '', true, 'temperature_pressure_level', 'K', ''),
    ('ifsoper', 'temperature_850_k', 't', 'pl 850', '', true, 'temperature_pressure_level', 'K', ''),
    ('ifsoper', 'relative_humidity_700_pct', 'r', 'pl 700', '', true, 'humidity_pressure_level', 'percent', ''),
    ('ifsoper', 'geopotential_height_500_m', 'gh', 'pl 500', '', true, 'height_pressure_level', 'm', ''),

    ('ifsenfo', 'member_temperature_2m_k', '2t', 'sfc', '', true, 'member_temperature', 'K', 'HKO center only, members 0-50'),

    ('cwawrf15', 'temperature_2m_k', 'TMP', '2 m above ground', '', true, 'temperature', 'K', ''),
    ('cwawrf15', 'dewpoint_2m_k', 'DPT', '2 m above ground', '', true, 'dewpoint', 'K', ''),
    ('cwawrf15', 'u_wind_10m_mps', 'UGRD', '10 m above ground', '', true, 'wind_u', 'm s-1', ''),
    ('cwawrf15', 'v_wind_10m_mps', 'VGRD', '10 m above ground', '', true, 'wind_v', 'm s-1', ''),
    ('cwawrf15', 'mslp_pa', 'PRMSL', 'mean sea level', '', true, 'pressure', 'Pa', ''),
    ('cwawrf15', 'accumulated_precip_kg_m2', 'APCP', 'surface', '', true, 'accumulation', 'kg m-2', ''),
    ('cwawrf15', 'net_shortwave_w_m2', 'NSWRF', 'surface', '', true, 'radiation', 'W m-2', ''),
    ('cwawrf15', 'temperature_850_k', 'TMP', '850 mb', '', true, 'temperature_pressure_level', 'K', ''),
    ('cwawrf15', 'relative_humidity_700_pct', 'RH', '700 mb', '', true, 'humidity_pressure_level', 'percent', ''),
    ('cwawrf15', 'geopotential_height_500_m', 'HGT', '500 mb', '', true, 'height_pressure_level', 'm', ''),

    ('aifsoper', 'temperature_2m_k', '2t', 'sfc', '', true, 'temperature', 'K', ''),
    ('aifsoper', 'dewpoint_2m_k', '2d', 'sfc', '', true, 'dewpoint', 'K', ''),
    ('aifsoper', 'u_wind_10m_mps', '10u', 'sfc', '', true, 'wind_u', 'm s-1', ''),
    ('aifsoper', 'v_wind_10m_mps', '10v', 'sfc', '', true, 'wind_v', 'm s-1', ''),
    ('aifsoper', 'mslp_pa', 'msl', 'sfc', '', true, 'pressure', 'Pa', ''),
    ('aifsoper', 'total_precip_m', 'tp', 'sfc', '', true, 'accumulation', 'm', ''),
    ('aifsoper', 'shortwave_down_j_m2', 'ssrd', 'sfc', '', true, 'radiation_accumulation', 'J m-2', ''),
    ('aifsoper', 'temperature_850_k', 't', 'pl 850', '', true, 'temperature_pressure_level', 'K', ''),

    ('aifsenfo', 'member_temperature_2m_k', '2t', 'sfc', '', true, 'member_temperature', 'K', 'HKO center only, members 0-50'),

    ('aigfssfc', 'temperature_2m_k', 'TMP', '2 m above ground', '', true, 'temperature', 'K', ''),
    ('aigfssfc', 'dewpoint_2m_k', 'DPT', '2 m above ground', '', true, 'dewpoint', 'K', ''),
    ('aigfssfc', 'u_wind_10m_mps', 'UGRD', '10 m above ground', '', true, 'wind_u', 'm s-1', ''),
    ('aigfssfc', 'v_wind_10m_mps', 'VGRD', '10 m above ground', '', true, 'wind_v', 'm s-1', ''),
    ('aigfssfc', 'mslp_pa', 'PRMSL', 'mean sea level', '', true, 'pressure', 'Pa', ''),

    ('aigfspres', 'temperature_850_k', 'TMP', '850 mb', '', true, 'temperature_pressure_level', 'K', ''),
    ('aigfspres', 'geopotential_height_500_m', 'HGT', '500 mb', '', true, 'height_pressure_level', 'm', ''),

    ('aigefssfc', 'member_temperature_2m_k', 'TMP', '2 m above ground', '', true, 'member_temperature', 'K', 'HKO center only, members 0-30'),

    ('graphcast', 'temperature_2m_k', 'TMP', '2 m above ground', '', true, 'temperature', 'K', ''),
    ('graphcast', 'u_wind_10m_mps', 'UGRD', '10 m above ground', '', true, 'wind_u', 'm s-1', ''),
    ('graphcast', 'v_wind_10m_mps', 'VGRD', '10 m above ground', '', true, 'wind_v', 'm s-1', ''),
    ('graphcast', 'mslp_pa', 'PRMSL', 'mean sea level', '', true, 'pressure', 'Pa', ''),
    ('graphcast', 'temperature_850_k', 'TMP', '850 mb', '', true, 'temperature_pressure_level', 'K', ''),
    ('graphcast', 'geopotential_height_500_m', 'HGT', '500 mb', '', true, 'height_pressure_level', 'm', ''),

    ('fourcastnetgfs', 'temperature_2m_k', 'TMP', '2 m above ground', '', true, 'temperature', 'K', ''),
    ('fourcastnetgfs', 'u_wind_10m_mps', 'UGRD', '10 m above ground', '', true, 'wind_u', 'm s-1', ''),
    ('fourcastnetgfs', 'v_wind_10m_mps', 'VGRD', '10 m above ground', '', true, 'wind_v', 'm s-1', ''),
    ('fourcastnetgfs', 'mslp_pa', 'PRMSL', 'mean sea level', '', true, 'pressure', 'Pa', ''),
    ('fourcastnetgfs', 'temperature_850_k', 'TMP', '850 mb', '', true, 'temperature_pressure_level', 'K', ''),
    ('fourcastnetgfs', 'geopotential_height_500_m', 'HGT', '500 mb', '', true, 'height_pressure_level', 'm', ''),

    ('nbmoc', 'temperature_2m_k', 'TMP', '2 m above ground', '', true, 'temperature', 'K', 'Probe only'),
    ('nbmoc', 'u_wind_10m_mps', 'UGRD', '10 m above ground', '', true, 'wind_u', 'm s-1', 'Probe only'),
    ('nbmoc', 'v_wind_10m_mps', 'VGRD', '10 m above ground', '', true, 'wind_v', 'm s-1', 'Probe only'),
    ('nbmoc', 'mslp_pa', 'PRMSL', 'mean sea level', '', true, 'pressure', 'Pa', 'Probe only')
ON CONFLICT (dataset_code, alias) DO UPDATE SET
    native_name = EXCLUDED.native_name,
    native_level = EXCLUDED.native_level,
    native_info = EXCLUDED.native_info,
    required = EXCLUDED.required,
    variable_role = EXCLUDED.variable_role,
    canonical_unit = EXCLUDED.canonical_unit,
    notes = EXCLUDED.notes;

INSERT INTO governance.schema_version (migration_version, description)
VALUES (
    '20260625_0007_tactical_gribstream_h24n_schema',
    'Tactical H24N GribStream exact-cycle model plan, 12-point stencil, chunk ledger, raw object manifest, and wide forecast storage'
)
ON CONFLICT (migration_version) DO NOTHING;

COMMIT;
