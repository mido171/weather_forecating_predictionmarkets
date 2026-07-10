package com.predictionmarkets.weather.klga.wu;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Instant;
import java.time.LocalDate;

import org.junit.jupiter.api.Test;

class WuTruthParserTest {
  private final ObjectMapper objectMapper = new ObjectMapper().findAndRegisterModules();
  private final WuTruthParser parser = new WuTruthParser(objectMapper);

  @Test
  void parserUsesHourlyTempInsteadOfProviderMaxTemp() {
    long highTime = Instant.parse("2026-05-21T04:51:00Z").getEpochSecond();
    long lowTime = Instant.parse("2026-05-21T15:32:00Z").getEpochSecond();
    String body = """
        {
          "metadata": {"id": "KLGA:9:US", "units": "e"},
          "observations": [
            {
              "obs_id": "KLGA",
              "valid_time_gmt": %d,
              "temp": 66,
              "max_temp": 96,
              "min_temp": 56,
              "dewPt": 57,
              "rh": 73,
              "wspd": 9,
              "wx_phrase": "Mostly Cloudy"
            },
            {
              "obs_id": "KLGA",
              "valid_time_gmt": %d,
              "temp": 56,
              "max_temp": 96,
              "min_temp": 56,
              "dewPt": 52,
              "rh": 87,
              "wspd": 5,
              "wx_phrase": "Light Rain"
            }
          ]
        }
        """.formatted(highTime, lowTime);
    WuTruthFetchResult result = new WuTruthFetchResult(
        WuTruthStation.byStationId("KLGA"),
        LocalDate.of(2026, 5, 21),
        LocalDate.of(2026, 5, 21),
        true,
        200,
        body,
        null,
        null,
        Instant.parse("2026-05-22T12:00:00Z"),
        1,
        "https://api.weather.com/v1/location/KLGA:9:US/observations/historical.json?apiKey=REDACTED");

    WuTruthDailyRow row = parser.parse(result).get(0);

    assertThat(row.tmaxF()).isEqualTo(66);
    assertThat(row.tminF()).isEqualTo(56);
    assertThat(row.dailyHighSource()).isEqualTo("hourly_temp_max");
    assertThat(row.validationStatus()).isEqualTo("manual_confirmed");
    assertThat(row.providerMaxTempValuesJson().toString()).contains("\"max_temp\":96.0");
  }

  @Test
  void failureRowsStayInTruthTableContractShape() {
    WuTruthFetchResult result = new WuTruthFetchResult(
        WuTruthStation.byStationId("KLGA"),
        LocalDate.of(2026, 5, 20),
        LocalDate.of(2026, 5, 21),
        false,
        429,
        "rate limited",
        "HTTP_429",
        "rate limited",
        Instant.parse("2026-05-22T12:00:00Z"),
        5,
        "https://api.weather.com/v1/location/KLGA:9:US/observations/historical.json?apiKey=REDACTED");

    var rows = parser.parse(result);

    assertThat(rows).hasSize(2);
    assertThat(rows).allSatisfy(row -> {
      assertThat(row.validationStatus()).isEqualTo("fetch_failed");
      assertThat(row.fetchStatus()).isEqualTo("fetch_failed");
      assertThat(row.dailyHighSource()).isEqualTo("hourly_temp_max");
      assertThat(row.tmaxF()).isNull();
    });
  }

  @Test
  void providerNoDataFailureRowsAreTerminalNoData() {
    String body = """
        {
          "metadata": {"status_code": 400},
          "success": false,
          "errors": [
            {"error": {"code": "NDF-0001", "message": "There was no data found for your historical observations query."}}
          ]
        }
        """;
    WuTruthFetchResult result = new WuTruthFetchResult(
        WuTruthStation.byStationId("KTEB"),
        LocalDate.of(1980, 1, 18),
        LocalDate.of(1980, 1, 19),
        false,
        400,
        body,
        "HTTP_400",
        body,
        Instant.parse("2026-05-22T12:00:00Z"),
        3,
        "https://api.weather.com/v1/location/KTEB:9:US/observations/historical.json?apiKey=REDACTED");

    var rows = parser.parse(result);

    assertThat(rows).hasSize(2);
    assertThat(rows).allSatisfy(row -> {
      assertThat(row.validationStatus()).isEqualTo("no_data");
      assertThat(row.fetchStatus()).isEqualTo("no_data");
      assertThat(row.tmaxF()).isNull();
    });
  }

  @Test
  void parserRemovesOutOfBoundsHourlyTempFromValidExtremaAndMarksDaySuspect() {
    long validHighTime = Instant.parse("2011-08-02T18:51:00Z").getEpochSecond();
    long spikeTime = Instant.parse("2011-08-02T19:51:00Z").getEpochSecond();
    long lowTime = Instant.parse("2011-08-02T09:51:00Z").getEpochSecond();
    String body = """
        {
          "metadata": {"id": "KLGA:9:US", "units": "e"},
          "observations": [
            {"obs_id": "KLGA", "valid_time_gmt": %d, "temp": 88, "dewPt": 70, "rh": 55, "wspd": 8},
            {"obs_id": "KLGA", "valid_time_gmt": %d, "temp": 127, "dewPt": 70, "rh": 55, "wspd": 8},
            {"obs_id": "KLGA", "valid_time_gmt": %d, "temp": 75, "dewPt": 70, "rh": 70, "wspd": 5}
          ]
        }
        """.formatted(validHighTime, spikeTime, lowTime);
    WuTruthFetchResult result = new WuTruthFetchResult(
        WuTruthStation.byStationId("KLGA"),
        LocalDate.of(2011, 8, 2),
        LocalDate.of(2011, 8, 2),
        true,
        200,
        body,
        null,
        null,
        Instant.parse("2026-05-22T12:00:00Z"),
        1,
        "https://api.weather.com/v1/location/KLGA:9:US/observations/historical.json?apiKey=REDACTED");

    WuTruthDailyRow row = parser.parse(result).get(0);

    assertThat(row.tmaxF()).isEqualTo(88);
    assertThat(row.tminF()).isEqualTo(75);
    assertThat(row.validationStatus()).isEqualTo("suspect");
    assertThat(row.hourlyObservationsJson().toString()).contains("\"temp_f\":null");
    assertThat(row.hourlyObservationsJson().toString()).doesNotContain("\"temp_f\":127");
    assertThat(row.validationNotesJson().path("notes").toString())
        .contains("out_of_bounds_hourly_temperature_removed");
  }
}
