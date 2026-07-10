package com.predictionmarkets.weather.klga.iemmos;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.List;
import java.util.UUID;
import org.junit.jupiter.api.Test;

class IemMosParserTest {
  private final IemMosParser parser = new IemMosParser(new ObjectMapper());

  @Test
  void parseStructuredRowsWithRuntimePlusTwoHourAvailability() {
    String payload = """
        [
          {
            "station": "KLGA",
            "model": "GFS",
            "runtime": "2026-06-28T06:00:00Z",
            "ftime": "2026-06-28T18:00:00Z",
            "n_x": "88",
            "tmp": "86",
            "dpt": "65",
            "wsp": "12",
            "p06": "20",
            "q06": "0"
          }
        ]
        """;
    IemMosChunk chunk = new IemMosChunk(
        "chunk",
        "job",
        new IemMosStation("KLGA", "LGA"),
        IemMosProduct.MAV,
        "T_1245UTC",
        Instant.parse("2026-06-28T00:00:00Z"),
        Instant.parse("2026-06-29T00:00:00Z"),
        "abc123",
        "{}");
    IemMosStoredRequest stored = new IemMosStoredRequest(
        "iem_mos_abc123",
        UUID.randomUUID(),
        "raw.gz",
        "payloadhash",
        payload.length());

    List<IemMosForecastRow> rows = parser.parse(payload.getBytes(StandardCharsets.UTF_8), chunk, stored);

    assertThat(rows).hasSize(1);
    IemMosForecastRow row = rows.get(0);
    assertThat(row.runTimeUtc()).isEqualTo(Instant.parse("2026-06-28T06:00:00Z"));
    assertThat(row.forecastValidTimeUtc()).isEqualTo(Instant.parse("2026-06-28T18:00:00Z"));
    assertThat(row.effectiveAvailableAtUtc()).isEqualTo(Instant.parse("2026-06-28T08:00:00Z"));
    assertThat(row.availabilityMethod()).isEqualTo("conservative_lag_rule");
    assertThat(row.nxF()).isEqualTo(88.0);
    assertThat(row.tmpF()).isEqualTo(86.0);
    assertThat(row.pop()).isEqualTo(20.0);
    assertThat(row.rawValuesJson()).contains("\"n_x\":\"88\"");
  }
}
