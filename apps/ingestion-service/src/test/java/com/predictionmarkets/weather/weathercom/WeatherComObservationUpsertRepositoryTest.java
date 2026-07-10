package com.predictionmarkets.weather.weathercom;

import static org.assertj.core.api.Assertions.assertThat;

import com.predictionmarkets.weather.models.WeatherComApiCall;
import com.predictionmarkets.weather.models.WeatherComIngestionRun;
import com.predictionmarkets.weather.models.WeatherComIngestionStatus;
import com.predictionmarkets.weather.repository.WeatherComApiCallRepository;
import com.predictionmarkets.weather.repository.WeatherComIngestionRunRepository;
import com.predictionmarkets.weather.repository.WeatherComObservationRepository;
import com.predictionmarkets.weather.repository.WeatherComObservationUpsertRepository;
import java.math.BigDecimal;
import java.time.Instant;
import java.time.LocalDate;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.data.domain.PageRequest;
import org.springframework.test.context.ActiveProfiles;

@SpringBootTest
@ActiveProfiles("test")
class WeatherComObservationUpsertRepositoryTest {
  @Autowired
  private WeatherComObservationUpsertRepository upsertRepository;

  @Autowired
  private WeatherComObservationRepository observationRepository;

  @Autowired
  private WeatherComApiCallRepository apiCallRepository;

  @Autowired
  private WeatherComIngestionRunRepository runRepository;

  @BeforeEach
  void setUp() {
    upsertRepository.deleteAll();
    apiCallRepository.deleteAll();
    runRepository.deleteAll();
  }

  @Test
  void reingestingSameDedupKeyUpdatesInPlaceWithoutCreatingDuplicates() {
    WeatherComIngestionRun run = saveRun();
    WeatherComApiCall firstCall = saveApiCall(run, Instant.parse("2026-02-17T12:00:00Z"));

    upsertRepository.upsertAll(List.of(row(firstCall.getId(), 54, 12)), 100);

    assertThat(observationRepository.countByRequestLocationIdAndObsIdAndValidTimeGmt(
        "KNYC:9:US", "KLGA", 1700000000L)).isEqualTo(1L);

    WeatherComApiCall secondCall = saveApiCall(run, Instant.parse("2026-02-17T13:00:00Z"));
    upsertRepository.upsertAll(List.of(row(secondCall.getId(), 61, 20)), 100);

    assertThat(observationRepository.countByRequestLocationIdAndObsIdAndValidTimeGmt(
        "KNYC:9:US", "KLGA", 1700000000L)).isEqualTo(1L);

    var observations = observationRepository.search(
            "KNYC:9:US",
            "KLGA",
            1700000000L,
            1700000000L,
            PageRequest.of(0, 10))
        .getContent();
    assertThat(observations).hasSize(1);
    assertThat(observations.get(0).getTemp()).isEqualTo(61);
    assertThat(observations.get(0).getWspd()).isEqualTo(20);
    assertThat(observations.get(0).getApiCallId()).isEqualTo(secondCall.getId());
  }

  private WeatherComIngestionRun saveRun() {
    Instant now = Instant.parse("2026-02-17T11:55:00Z");
    WeatherComIngestionRun run = new WeatherComIngestionRun();
    run.setStatus(WeatherComIngestionStatus.RUNNING);
    run.setStartedAtUtc(now);
    run.setRequestedBy("test");
    run.setRequestPayloadJson("{\"test\":true}");
    run.setTotalTasks(1);
    run.setSucceededTasks(0);
    run.setFailedTasks(0);
    run.setCreatedAtUtc(now);
    run.setUpdatedAtUtc(now);
    return runRepository.save(run);
  }

  private WeatherComApiCall saveApiCall(WeatherComIngestionRun run, Instant fetchedAt) {
    WeatherComApiCall apiCall = new WeatherComApiCall();
    apiCall.setIngestionRun(run);
    apiCall.setRequestLocationId("KNYC:9:US");
    apiCall.setUnits("e");
    apiCall.setStartDate(LocalDate.of(2026, 2, 17));
    apiCall.setEndDate(LocalDate.of(2026, 2, 17));
    apiCall.setResponseLocationId("KNYC:9:US");
    apiCall.setResponseUnits("e");
    apiCall.setResponseLanguage("en-US");
    apiCall.setHttpStatus(200);
    apiCall.setFetchedAtUtc(fetchedAt);
    apiCall.setCreatedAtUtc(fetchedAt);
    apiCall.setUpdatedAtUtc(fetchedAt);
    return apiCallRepository.save(apiCall);
  }

  private WeatherComObservationUpsertRepository.UpsertRow row(Long apiCallId, int temp, int wspd) {
    Instant now = Instant.parse("2026-02-17T12:10:00Z");
    return new WeatherComObservationUpsertRepository.UpsertRow(
        apiCallId,
        "KNYC:9:US",
        "KLGA",
        "KLGA",
        "LaGuardia Airport",
        1700000000L,
        Instant.ofEpochSecond(1700000000L),
        "D",
        temp,
        44,
        54,
        68,
        new BigDecimal("30.01"),
        0,
        "Steady",
        new BigDecimal("10.0"),
        54,
        120,
        "ESE",
        16,
        wspd,
        "Mostly Cloudy",
        26,
        2600,
        BigDecimal.ZERO,
        BigDecimal.ZERO,
        BigDecimal.ZERO,
        null,
        null,
        null,
        null,
        52,
        "BKN",
        null,
        null,
        null,
        "Cloudy",
        "observation",
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        now,
        now);
  }
}
