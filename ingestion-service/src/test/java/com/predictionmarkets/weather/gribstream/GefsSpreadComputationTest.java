package com.predictionmarkets.weather.gribstream;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.Instant;
import java.util.List;
import org.junit.jupiter.api.Test;

class GefsSpreadComputationTest {

  @Test
  void computesSpreadAcrossMembersAndTimes() {
    Instant forecastedAt = Instant.parse("2026-01-09T06:00:00Z");
    Instant time1 = Instant.parse("2026-01-10T12:00:00Z");
    Instant time2 = Instant.parse("2026-01-10T18:00:00Z");
    List<GribstreamRow> rows = List.of(
        new GribstreamRow(forecastedAt, time1, 280.0, 0),
        new GribstreamRow(forecastedAt, time1, 281.0, 1),
        new GribstreamRow(forecastedAt, time1, 279.0, 2),
        new GribstreamRow(forecastedAt, time2, 285.0, 0),
        new GribstreamRow(forecastedAt, time2, 287.0, 1),
        new GribstreamRow(forecastedAt, time2, 283.0, 2));

    GribstreamDailyMetrics.SpreadResult result =
        GribstreamDailyMetrics.computeSpread(rows, 3);

    double expectedStddev = Math.sqrt(8.0 / 3.0);
    assertThat(result.spreadK()).isCloseTo(expectedStddev, org.assertj.core.data.Offset.offset(1e-6));
    assertThat(result.spreadF()).isCloseTo(expectedStddev * 9.0 / 5.0,
        org.assertj.core.data.Offset.offset(1e-6));
    assertThat(result.timesUsed()).isEqualTo(2);
  }
}
