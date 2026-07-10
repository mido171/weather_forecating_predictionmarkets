-- HKG-T24-001 GribStream safe-row contract.
-- Source of truth: hkg_t24.db.ddl.NWP_SAFE_VIEW_SQL and hkg_t24.features.gribstream_safe_rows.

CREATE OR REPLACE VIEW model_features.v_nwp_h24n_safe_rows AS
SELECT
  fw.target_date_hkt::date AS target_date_hkt,
  fw.cutoff_id,
  fw.dataset_code,
  fw.run_time_utc,
  fw.valid_time_utc,
  fw.source_response_object_id,
  r.object_uri,
  true AS row_is_safe_h24n,
  NULL::text AS exclusion_reason,
  'full_tactical_backfill_ok_tmax'::text AS source_scope
FROM nwp_tactical.forecast_wide fw
JOIN nwp_tactical.raw_response_object r
  ON r.response_object_id = fw.source_response_object_id
WHERE r.object_uri LIKE '%full_tactical_backfill_ok_tmax%'
  AND fw.cutoff_id = 'H24N'
  AND fw.dataset_code NOT IN ('nbmoc','aigfspres','aigefssfc')
  AND fw.run_time_utc + interval '6 hours'
      <= ((fw.target_date_hkt::date - 1) + time '15:00') AT TIME ZONE 'Asia/Hong_Kong';
