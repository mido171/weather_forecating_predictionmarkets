package com.predictionmarkets.weather.repository;

import java.math.BigDecimal;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.jdbc.core.namedparam.SqlParameterSource;
import org.springframework.stereotype.Repository;

@Repository
public class MosAsofFeatureUpsertRepository {
  private static final String UPSERT_SQL = """
      INSERT INTO mos_asof_feature (
        station_id,
        target_date_local,
        asof_policy_id,
        model,
        asof_utc,
        asof_local,
        station_zoneid,
        chosen_runtime_utc,
        tmax_f,
        missing_reason,
        raw_payload_hash_ref,
        retrieved_at_utc
      ) VALUES (
        :stationId,
        :targetDateLocal,
        :asofPolicyId,
        :model,
        :asofUtc,
        :asofLocal,
        :stationZoneid,
        :chosenRuntimeUtc,
        :tmaxF,
        :missingReason,
        :rawPayloadHashRef,
        :retrievedAtUtc
      )
      ON DUPLICATE KEY UPDATE
        asof_utc = VALUES(asof_utc),
        asof_local = VALUES(asof_local),
        station_zoneid = VALUES(station_zoneid),
        chosen_runtime_utc = VALUES(chosen_runtime_utc),
        tmax_f = VALUES(tmax_f),
        missing_reason = VALUES(missing_reason),
        raw_payload_hash_ref = VALUES(raw_payload_hash_ref),
        retrieved_at_utc = VALUES(retrieved_at_utc)
      """;

  private final NamedParameterJdbcTemplate jdbcTemplate;

  public MosAsofFeatureUpsertRepository(NamedParameterJdbcTemplate jdbcTemplate) {
    this.jdbcTemplate = jdbcTemplate;
  }

  public int[] upsertAll(List<UpsertRow> rows) {
    SqlParameterSource[] batch = rows.stream()
        .map(MosAsofFeatureUpsertRepository::toParams)
        .toArray(SqlParameterSource[]::new);
    return jdbcTemplate.batchUpdate(UPSERT_SQL, batch);
  }

  private static SqlParameterSource toParams(UpsertRow row) {
    return new MapSqlParameterSource()
        .addValue("stationId", row.stationId)
        .addValue("targetDateLocal", row.targetDateLocal)
        .addValue("asofPolicyId", row.asofPolicyId)
        .addValue("model", row.model)
        .addValue("asofUtc", row.asofUtc)
        .addValue("asofLocal", row.asofLocal)
        .addValue("stationZoneid", row.stationZoneid)
        .addValue("chosenRuntimeUtc", row.chosenRuntimeUtc)
        .addValue("tmaxF", row.tmaxF)
        .addValue("missingReason", row.missingReason)
        .addValue("rawPayloadHashRef", row.rawPayloadHashRef)
        .addValue("retrievedAtUtc", row.retrievedAtUtc);
  }

  public static final class UpsertRow {
    private final String stationId;
    private final LocalDate targetDateLocal;
    private final Long asofPolicyId;
    private final String model;
    private final Instant asofUtc;
    private final LocalDateTime asofLocal;
    private final String stationZoneid;
    private final Instant chosenRuntimeUtc;
    private final BigDecimal tmaxF;
    private final String missingReason;
    private final String rawPayloadHashRef;
    private final Instant retrievedAtUtc;

    public UpsertRow(String stationId,
                     LocalDate targetDateLocal,
                     Long asofPolicyId,
                     String model,
                     Instant asofUtc,
                     LocalDateTime asofLocal,
                     String stationZoneid,
                     Instant chosenRuntimeUtc,
                     BigDecimal tmaxF,
                     String missingReason,
                     String rawPayloadHashRef,
                     Instant retrievedAtUtc) {
      this.stationId = stationId;
      this.targetDateLocal = targetDateLocal;
      this.asofPolicyId = asofPolicyId;
      this.model = model;
      this.asofUtc = asofUtc;
      this.asofLocal = asofLocal;
      this.stationZoneid = stationZoneid;
      this.chosenRuntimeUtc = chosenRuntimeUtc;
      this.tmaxF = tmaxF;
      this.missingReason = missingReason;
      this.rawPayloadHashRef = rawPayloadHashRef;
      this.retrievedAtUtc = retrievedAtUtc;
    }
  }
}
