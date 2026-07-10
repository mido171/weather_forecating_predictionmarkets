package com.predictionmarkets.weather.repository;

import java.time.Instant;
import java.util.Arrays;
import java.util.List;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.jdbc.core.namedparam.SqlParameterSource;
import org.springframework.stereotype.Repository;

@Repository
public class MosRunUpsertRepository {
  private static final String UPSERT_SQL = """
      INSERT INTO mos_run (
        station_id,
        model,
        runtime_utc,
        raw_payload_hash,
        retrieved_at_utc
      ) VALUES (
        :stationId,
        :model,
        :runtimeUtc,
        :rawPayloadHash,
        :retrievedAtUtc
      )
      ON DUPLICATE KEY UPDATE
        raw_payload_hash = VALUES(raw_payload_hash),
        retrieved_at_utc = VALUES(retrieved_at_utc)
      """;
  private static final int DEFAULT_BATCH_SIZE = 200;

  private final NamedParameterJdbcTemplate jdbcTemplate;

  public MosRunUpsertRepository(NamedParameterJdbcTemplate jdbcTemplate) {
    this.jdbcTemplate = jdbcTemplate;
  }

  public int upsertAll(List<UpsertRow> rows) {
    if (rows == null || rows.isEmpty()) {
      return 0;
    }
    int updated = 0;
    for (int start = 0; start < rows.size(); start += DEFAULT_BATCH_SIZE) {
      int end = Math.min(start + DEFAULT_BATCH_SIZE, rows.size());
      List<UpsertRow> slice = rows.subList(start, end);
      SqlParameterSource[] batch = slice.stream()
          .map(MosRunUpsertRepository::toParams)
          .toArray(SqlParameterSource[]::new);
      updated += Arrays.stream(jdbcTemplate.batchUpdate(UPSERT_SQL, batch)).sum();
    }
    return updated;
  }

  private static SqlParameterSource toParams(UpsertRow row) {
    return new MapSqlParameterSource()
        .addValue("stationId", row.stationId)
        .addValue("model", row.model)
        .addValue("runtimeUtc", row.runtimeUtc)
        .addValue("rawPayloadHash", row.rawPayloadHash)
        .addValue("retrievedAtUtc", row.retrievedAtUtc);
  }

  public static final class UpsertRow {
    private final String stationId;
    private final String model;
    private final Instant runtimeUtc;
    private final String rawPayloadHash;
    private final Instant retrievedAtUtc;

    public UpsertRow(String stationId,
                     String model,
                     Instant runtimeUtc,
                     String rawPayloadHash,
                     Instant retrievedAtUtc) {
      this.stationId = stationId;
      this.model = model;
      this.runtimeUtc = runtimeUtc;
      this.rawPayloadHash = rawPayloadHash;
      this.retrievedAtUtc = retrievedAtUtc;
    }
  }
}
