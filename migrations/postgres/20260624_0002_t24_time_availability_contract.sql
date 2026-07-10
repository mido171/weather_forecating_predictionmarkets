-- Canonical HKG T+24 time, availability, and sealing contract.
-- PostgreSQL 15+. Additive and idempotent.

BEGIN;

CREATE SCHEMA IF NOT EXISTS governance;

CREATE OR REPLACE FUNCTION governance.hkg_t24_cutoff_utc(target_date date)
RETURNS timestamptz
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT (((target_date - 1)::timestamp + time '15:00') AT TIME ZONE 'Asia/Hong_Kong');
$$;

CREATE TABLE IF NOT EXISTS governance.operational_contract (
    contract_version text PRIMARY KEY,
    target_station text NOT NULL,
    target_variable text NOT NULL,
    timezone_name text NOT NULL CHECK (timezone_name = 'Asia/Hong_Kong'),
    decision_cutoff_local_time time NOT NULL,
    decision_day_offset integer NOT NULL,
    cutoff_function regproc NOT NULL,
    development_end_date date NOT NULL,
    locked_validation_year integer NOT NULL,
    final_historical_test_year integer NOT NULL,
    post_final_holdout_year integer NOT NULL,
    status text NOT NULL CHECK (status IN ('ACTIVE','SUPERSEDED','BLOCKED')),
    created_at_utc timestamptz NOT NULL DEFAULT now(),
    notes text NOT NULL DEFAULT ''
);

INSERT INTO governance.operational_contract (
    contract_version,
    target_station,
    target_variable,
    timezone_name,
    decision_cutoff_local_time,
    decision_day_offset,
    cutoff_function,
    development_end_date,
    locked_validation_year,
    final_historical_test_year,
    post_final_holdout_year,
    status,
    notes
)
VALUES (
    'hkg_t24_1500hkt_v1',
    'Hong Kong Observatory',
    'daily_maximum_temperature_c',
    'Asia/Hong_Kong',
    time '15:00:00',
    -1,
    'governance.hkg_t24_cutoff_utc(date)'::regprocedure,
    date '2023-12-31',
    2024,
    2025,
    2026,
    'ACTIVE',
    'Forecast target date T at 15:00 HKT on T-1, equivalent to 07:00 UTC on T-1.'
)
ON CONFLICT (contract_version) DO UPDATE SET
    target_station = EXCLUDED.target_station,
    target_variable = EXCLUDED.target_variable,
    timezone_name = EXCLUDED.timezone_name,
    decision_cutoff_local_time = EXCLUDED.decision_cutoff_local_time,
    decision_day_offset = EXCLUDED.decision_day_offset,
    cutoff_function = EXCLUDED.cutoff_function,
    development_end_date = EXCLUDED.development_end_date,
    locked_validation_year = EXCLUDED.locked_validation_year,
    final_historical_test_year = EXCLUDED.final_historical_test_year,
    post_final_holdout_year = EXCLUDED.post_final_holdout_year,
    status = EXCLUDED.status,
    notes = EXCLUDED.notes;

CREATE TABLE IF NOT EXISTS governance.availability_grade (
    grade_code text PRIMARY KEY,
    grade_rank smallint NOT NULL UNIQUE,
    strict_eligible boolean NOT NULL,
    description text NOT NULL
);

INSERT INTO governance.availability_grade (grade_code, grade_rank, strict_eligible, description)
VALUES
    ('A_EXACT_FIRST_SEEN', 1, true, 'Collector observed the provider response before the cutoff.'),
    ('B_PROVIDER_SCHEDULE_PROVEN', 2, true, 'Authoritative provider schedule plus conservative latency proves availability before cutoff.'),
    ('C_RUN_TIME_ONLY', 3, false, 'Only model initialization or run time is known; diagnostic until promoted.'),
    ('D_RETROSPECTIVE_ONLY', 4, false, 'Finalized, corrected, reanalysis, archive-only, or otherwise retrospective source.'),
    ('E_REJECTED', 5, false, 'Timestamp, parser, coverage, or source integrity failure.')
ON CONFLICT (grade_code) DO UPDATE SET
    grade_rank = EXCLUDED.grade_rank,
    strict_eligible = EXCLUDED.strict_eligible,
    description = EXCLUDED.description;

CREATE TABLE IF NOT EXISTS governance.sealed_period (
    sealed_period_id text PRIMARY KEY,
    local_date_start date NOT NULL,
    local_date_end date,
    label_schema text NOT NULL,
    label_table text NOT NULL,
    read_role text NOT NULL,
    status text NOT NULL CHECK (status IN ('DEVELOPMENT','SEALED','OPENED_ONCE','CONTAMINATED','RETIRED')),
    access_policy text NOT NULL,
    created_at_utc timestamptz NOT NULL DEFAULT now(),
    CHECK (local_date_end IS NULL OR local_date_end >= local_date_start)
);

INSERT INTO governance.sealed_period (
    sealed_period_id,
    local_date_start,
    local_date_end,
    label_schema,
    label_table,
    read_role,
    status,
    access_policy
)
VALUES (
    'sealed_confirmation_2024_forward',
    date '2024-01-01',
    NULL,
    'sealed_confirmation',
    'hko_daily_tmax',
    'hkg_tmax_confirmation_admin',
    'SEALED',
    '2024+ labels may be opened only by the one-time T36 confirmation protocol.'
)
ON CONFLICT (sealed_period_id) DO UPDATE SET
    local_date_start = EXCLUDED.local_date_start,
    local_date_end = EXCLUDED.local_date_end,
    label_schema = EXCLUDED.label_schema,
    label_table = EXCLUDED.label_table,
    read_role = EXCLUDED.read_role,
    status = EXCLUDED.status,
    access_policy = EXCLUDED.access_policy;

CREATE OR REPLACE FUNCTION governance.is_available_for_cutoff(
    available_at_utc timestamptz,
    cutoff_utc timestamptz,
    grade_code text
)
RETURNS boolean
LANGUAGE sql
STABLE
AS $$
    SELECT available_at_utc IS NOT NULL
       AND cutoff_utc IS NOT NULL
       AND available_at_utc <= cutoff_utc
       AND EXISTS (
            SELECT 1
            FROM governance.availability_grade grade
            WHERE grade.grade_code = is_available_for_cutoff.grade_code
              AND grade.strict_eligible
       );
$$;

CREATE OR REPLACE FUNCTION governance.hkg_t24_is_eligible(
    target_date date,
    available_at_utc timestamptz,
    grade_code text
)
RETURNS boolean
LANGUAGE sql
STABLE
AS $$
    SELECT governance.is_available_for_cutoff(
        available_at_utc,
        governance.hkg_t24_cutoff_utc(target_date),
        grade_code
    );
$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hkg_tmax_live_inference') THEN
        CREATE ROLE hkg_tmax_live_inference;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hkg_tmax_research_dev') THEN
        CREATE ROLE hkg_tmax_research_dev;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hkg_tmax_confirmation_admin') THEN
        CREATE ROLE hkg_tmax_confirmation_admin;
    END IF;
END $$;

REVOKE ALL ON SCHEMA sealed_confirmation FROM PUBLIC;
REVOKE ALL ON SCHEMA sealed_confirmation FROM hkg_tmax_live_inference;
REVOKE ALL ON SCHEMA label_core FROM hkg_tmax_live_inference;

GRANT USAGE ON SCHEMA governance, feature_safe TO hkg_tmax_live_inference;
GRANT SELECT ON governance.operational_contract, governance.availability_grade, governance.sealed_period
TO hkg_tmax_live_inference;

GRANT USAGE ON SCHEMA sealed_confirmation TO hkg_tmax_confirmation_admin;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA sealed_confirmation
TO hkg_tmax_confirmation_admin;

INSERT INTO governance.schema_version (migration_version, description)
VALUES ('20260624_0002_t24_time_availability_contract', 'Canonical T+24 time, availability grade, sealed period, and eligibility contract')
ON CONFLICT (migration_version) DO NOTHING;

COMMIT;
