package com.predictionmarkets.weather.repository;

import java.math.BigDecimal;
import java.time.Instant;
import java.time.LocalDate;
import java.util.List;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.jdbc.core.namedparam.SqlParameterSource;
import org.springframework.stereotype.Repository;

@Repository
public class CliDailyUpsertRepository {
  private static final String UPSERT_SQL = """
      INSERT INTO cli_daily (
        station_id,
        target_date_local,
        tmax_f,
        tmin_f,
        report_issued_at_utc,
        raw_payload_hash,
        retrieved_at_utc,
        updated_at_utc
      ) VALUES (
        :stationId,
        :targetDateLocal,
        :tmaxF,
        :tminF,
        :reportIssuedAtUtc,
        :rawPayloadHash,
        :retrievedAtUtc,
        :updatedAtUtc
      )
      ON DUPLICATE KEY UPDATE
        tmax_f = VALUES(tmax_f),
        tmin_f = VALUES(tmin_f),
        report_issued_at_utc = VALUES(report_issued_at_utc),
        raw_payload_hash = VALUES(raw_payload_hash),
        retrieved_at_utc = VALUES(retrieved_at_utc),
        updated_at_utc = VALUES(updated_at_utc)
      """;

  private final NamedParameterJdbcTemplate jdbcTemplate;

  public CliDailyUpsertRepository(NamedParameterJdbcTemplate jdbcTemplate) {
    this.jdbcTemplate = jdbcTemplate;
  }

  public int[] upsertAll(List<UpsertRow> rows) {
    SqlParameterSource[] batch = rows.stream()
        .map(CliDailyUpsertRepository::toParams)
        .toArray(SqlParameterSource[]::new);
    return jdbcTemplate.batchUpdate(UPSERT_SQL, batch);
  }

  private static SqlParameterSource toParams(UpsertRow row) {
    return new MapSqlParameterSource()
        .addValue("stationId", row.stationId)
        .addValue("targetDateLocal", row.targetDateLocal)
        .addValue("tmaxF", row.tmaxF)
        .addValue("tminF", row.tminF)
        .addValue("reportIssuedAtUtc", row.reportIssuedAtUtc)
        .addValue("rawPayloadHash", row.rawPayloadHash)
        .addValue("retrievedAtUtc", row.retrievedAtUtc)
        .addValue("updatedAtUtc", row.updatedAtUtc);
  }

  public static final class UpsertRow {
    private final String stationId;
    private final LocalDate targetDateLocal;
    private final BigDecimal tmaxF;
    private final BigDecimal tminF;
    private final Instant reportIssuedAtUtc;
    private final String rawPayloadHash;
    private final Instant retrievedAtUtc;
    private final Instant updatedAtUtc;

    public UpsertRow(String stationId,
                     LocalDate targetDateLocal,
                     BigDecimal tmaxF,
                     BigDecimal tminF,
                     Instant reportIssuedAtUtc,
                     String rawPayloadHash,
                     Instant retrievedAtUtc,
                     Instant updatedAtUtc) {
      this.stationId = stationId;
      this.targetDateLocal = targetDateLocal;
      this.tmaxF = tmaxF;
      this.tminF = tminF;
      this.reportIssuedAtUtc = reportIssuedAtUtc;
      this.rawPayloadHash = rawPayloadHash;
      this.retrievedAtUtc = retrievedAtUtc;
      this.updatedAtUtc = updatedAtUtc;
    }
  }
}
