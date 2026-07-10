package com.predictionmarkets.weather.weathercom;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.weathercom.client.dto.WeatherComHistoricalResponse;
import java.io.IOException;
import java.time.Instant;
import org.junit.jupiter.api.Test;
import org.springframework.core.io.ClassPathResource;

class WeatherComDtoMappingTest {
  private final ObjectMapper objectMapper = new ObjectMapper();

  @Test
  void parsesHistoricalObservationFixtureWithReservedAndNullableFields() throws IOException {
    WeatherComHistoricalResponse payload = objectMapper.readValue(
        new ClassPathResource("weathercom/historical_observations_fixture.json").getInputStream(),
        WeatherComHistoricalResponse.class);

    assertThat(payload.getMetadata()).isNotNull();
    assertThat(payload.getMetadata().getLocationId()).isEqualTo("KNYC:9:US");
    assertThat(payload.getMetadata().getTransactionId()).isEqualTo("txn-12345");
    assertThat(payload.getObservations()).hasSize(2);

    var first = payload.getObservations().get(0);
    assertThat(first.getObservationClass()).isEqualTo("observation");
    assertThat(first.getDewPt()).isEqualTo(44);
    assertThat(first.getQualifier()).isNull();
  }

  @Test
  void convertsValidTimeGmtToUtcInstantAsExpected() throws IOException {
    WeatherComHistoricalResponse payload = objectMapper.readValue(
        new ClassPathResource("weathercom/historical_observations_fixture.json").getInputStream(),
        WeatherComHistoricalResponse.class);

    long validTimeGmt = payload.getObservations().get(0).getValidTimeGmt();
    Instant validTimeUtc = Instant.ofEpochSecond(validTimeGmt);
    assertThat(validTimeUtc).isEqualTo(Instant.parse("2023-11-14T22:13:20Z"));
  }
}
