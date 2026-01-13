package com.predictionmarkets.weather.gribstream;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.Instant;
import java.util.List;
import org.junit.jupiter.api.Test;

class DailyTmaxComputationTest {

  @Test
  void computesDailyMaxFromKelvin() {
    List<GribstreamRow> rows = List.of(
        new GribstreamRow(
            Instant.parse("2026-01-09T12:00:00Z"),
            Instant.parse("2026-01-10T12:00:00Z"),
            280.0,
            null),
        new GribstreamRow(
            Instant.parse("2026-01-09T12:00:00Z"),
            Instant.parse("2026-01-10T18:00:00Z"),
            282.0,
            null));

    GribstreamDailyMetrics.TmaxResult result = GribstreamDailyMetrics.computeTmax(rows);

    assertThat(result.tmaxK()).isEqualTo(282.0);
    assertThat(result.tmaxF()).isEqualTo(GribstreamDailyMetrics.kelvinToF(282.0));
    assertThat(result.forecastedAtUtc()).isEqualTo(Instant.parse("2026-01-09T12:00:00Z"));
  }
}
