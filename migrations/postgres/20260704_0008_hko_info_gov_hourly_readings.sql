-- Dedicated one-table archive for Info.gov HKO "HOURLY READINGS" dispatches.
-- One row represents one PRESS WEATHER HOURLY READINGS dispatch page.

BEGIN;

CREATE TABLE IF NOT EXISTS public.hko_info_gov_hourly_readings_1998_2026 (
    bulletin_id text PRIMARY KEY,
    source text NOT NULL,
    source_url text NOT NULL UNIQUE,
    index_date_hkt date,
    title text,
    press_weather_no integer,
    dispatch_at_hkt timestamp without time zone,
    dispatch_at_utc timestamp with time zone,
    observation_at_hkt timestamp without time zone,
    observation_at_utc timestamp with time zone,
    available_at_utc timestamp with time zone,
    retrieved_at_utc timestamp with time zone,
    hko_air_temp_c double precision,
    hko_relative_humidity_pct double precision,
    rainfall_text text,
    warning_text text,
    lightning_text text,
    tropical_cyclone_text text,
    tropical_cyclone_name text,
    tropical_cyclone_lat double precision,
    tropical_cyclone_lon double precision,
    station_readings_jsonb jsonb NOT NULL DEFAULT '[]'::jsonb,
    station_count integer NOT NULL,
    station_missing_count integer NOT NULL,
    station_temp_min_c double precision,
    station_temp_max_c double precision,
    station_temp_mean_c double precision,
    station_temp_spread_c double precision,
    target_station_present boolean NOT NULL,
    full_text text NOT NULL,
    raw_html_path text NOT NULL,
    raw_sha256 text NOT NULL,
    parse_status text NOT NULL,
    parse_notes text,
    ingested_at_utc timestamp with time zone NOT NULL,
    CHECK (source = 'info_gov'),
    CHECK (hko_air_temp_c IS NULL OR hko_air_temp_c BETWEEN -20 AND 60),
    CHECK (hko_relative_humidity_pct IS NULL OR hko_relative_humidity_pct BETWEEN 0 AND 100),
    CHECK (station_count >= 0),
    CHECK (station_missing_count >= 0),
    CHECK (station_missing_count <= station_count),
    CHECK (station_temp_min_c IS NULL OR station_temp_min_c BETWEEN -20 AND 60),
    CHECK (station_temp_max_c IS NULL OR station_temp_max_c BETWEEN -20 AND 60),
    CHECK (station_temp_min_c IS NULL OR station_temp_max_c IS NULL OR station_temp_min_c <= station_temp_max_c),
    CHECK (parse_status IN ('parsed', 'partial', 'failed'))
);

CREATE INDEX IF NOT EXISTS hko_info_gov_hourly_readings_1998_2026_dispatch_utc_idx
    ON public.hko_info_gov_hourly_readings_1998_2026 (dispatch_at_utc);

CREATE INDEX IF NOT EXISTS hko_info_gov_hourly_readings_1998_2026_obs_utc_idx
    ON public.hko_info_gov_hourly_readings_1998_2026 (observation_at_utc);

CREATE INDEX IF NOT EXISTS hko_info_gov_hourly_readings_1998_2026_index_date_idx
    ON public.hko_info_gov_hourly_readings_1998_2026 (index_date_hkt);

CREATE INDEX IF NOT EXISTS hko_info_gov_hourly_readings_1998_2026_raw_sha_idx
    ON public.hko_info_gov_hourly_readings_1998_2026 (raw_sha256);

CREATE INDEX IF NOT EXISTS hko_info_gov_hourly_readings_1998_2026_station_jsonb_gin_idx
    ON public.hko_info_gov_hourly_readings_1998_2026 USING gin (station_readings_jsonb);

COMMENT ON TABLE public.hko_info_gov_hourly_readings_1998_2026 IS
    'Canonical one-table Info.gov HKO hourly readings archive. One row per PRESS WEATHER HOURLY READINGS dispatch; station readings preserved in station_readings_jsonb.';

COMMIT;
