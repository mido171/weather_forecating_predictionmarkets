package com.predictionmarkets.weather.weathercom;

import static org.assertj.core.api.Assertions.assertThat;

import com.predictionmarkets.weather.models.WeatherComIngestionRun;
import com.predictionmarkets.weather.models.WeatherComIngestionStatus;
import com.predictionmarkets.weather.models.WeatherComLocation;
import com.predictionmarkets.weather.repository.WeatherComApiCallRepository;
import com.predictionmarkets.weather.repository.WeatherComIngestionRunRepository;
import com.predictionmarkets.weather.repository.WeatherComLocationRepository;
import com.predictionmarkets.weather.repository.WeatherComObservationUpsertRepository;
import com.predictionmarkets.weather.weathercom.service.WeatherComIngestionService;
import java.io.IOException;
import java.time.Instant;
import java.time.LocalDate;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.data.domain.PageRequest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;

@SpringBootTest
@ActiveProfiles("test")
class WeatherComIngestionConcurrencyTest {
  private static final MockWebServer SERVER = new MockWebServer();

  static {
    try {
      SERVER.start();
    } catch (IOException ex) {
      throw new ExceptionInInitializerError(ex);
    }
  }

  @DynamicPropertySource
  static void registerProperties(DynamicPropertyRegistry registry) {
    registry.add("weathercom.api.base-url", () -> SERVER.url("/").toString());
    registry.add("weathercom.api.api-key", () -> "test-weathercom-key");
    registry.add("weathercom.ingestion.enabled", () -> true);
    registry.add("weathercom.ingestion.thread-pool-size", () -> 2);
    registry.add("weathercom.ingestion.queue-capacity", () -> 2);
    registry.add("weathercom.ingestion.chunk-days", () -> 1);
    registry.add("weathercom.ingestion.max-retries", () -> 1);
    registry.add("weathercom.ingestion.retry-backoff-ms", () -> 1);
    registry.add("weathercom.ingestion.max-backoff-ms", () -> 5);
    registry.add("weathercom.ingestion.retry-jitter-ms", () -> 0);
    registry.add("weathercom.ingestion.rate-limit.permits-per-second", () -> 0.0d);
    registry.add("weathercom.ingestion.store-response-body", () -> false);
  }

  @Autowired
  private WeatherComIngestionService ingestionService;

  @Autowired
  private WeatherComLocationRepository locationRepository;

  @Autowired
  private WeatherComIngestionRunRepository runRepository;

  @Autowired
  private WeatherComApiCallRepository apiCallRepository;

  @Autowired
  private WeatherComObservationUpsertRepository observationUpsertRepository;

  @BeforeEach
  void setUp() {
    observationUpsertRepository.deleteAll();
    apiCallRepository.deleteAll();
    runRepository.deleteAll();
    locationRepository.deleteAll();

    locationRepository.save(location("KNYC:9:US"));
    locationRepository.save(location("KPHL:9:US"));
  }

  @AfterAll
  static void shutdownServer() throws IOException {
    SERVER.shutdown();
  }

  @Test
  void ingestionCompletesWithConfiguredSmallThreadPoolAndAccurateCounters() throws Exception {
    int taskCount = 2 * 3; // 2 locations * 3 days
    int baselineRequestCount = SERVER.getRequestCount();
    for (int i = 0; i < taskCount; i++) {
      SERVER.enqueue(new MockResponse()
          .setResponseCode(200)
          .setHeader("Content-Type", "application/json")
          .setBody(successPayload(i)));
    }

    WeatherComIngestionRun run = ingestionService.triggerIngestion(
        null,
        LocalDate.of(2026, 2, 17),
        LocalDate.of(2026, 2, 19),
        "e",
        "concurrency-test");

    WeatherComIngestionRun completed = awaitCompletion(run.getId());
    assertThat(completed.getStatus()).isEqualTo(WeatherComIngestionStatus.SUCCEEDED);
    assertThat(completed.getTotalTasks()).isEqualTo(taskCount);
    assertThat(completed.getSucceededTasks()).isEqualTo(taskCount);
    assertThat(completed.getFailedTasks()).isZero();
    assertThat(completed.getFinishedAtUtc()).isNotNull();

    long apiCallCount = apiCallRepository
        .findByIngestionRunIdOrderByIdDesc(run.getId(), PageRequest.of(0, 100))
        .getTotalElements();
    assertThat(apiCallCount).isEqualTo(taskCount);
    assertThat(SERVER.getRequestCount()).isEqualTo(baselineRequestCount + taskCount);
  }

  private WeatherComIngestionRun awaitCompletion(Long runId) throws InterruptedException {
    WeatherComIngestionRun run = null;
    for (int i = 0; i < 200; i++) {
      run = ingestionService.getRun(runId);
      if (run.getStatus() != WeatherComIngestionStatus.RUNNING) {
        return run;
      }
      Thread.sleep(50L);
    }
    return run;
  }

  private WeatherComLocation location(String locationId) {
    WeatherComLocation location = new WeatherComLocation();
    location.setLocationId(locationId);
    location.setDisplayName(locationId);
    location.setActive(true);
    location.setCreatedAtUtc(Instant.now());
    location.setUpdatedAtUtc(Instant.now());
    return location;
  }

  private String successPayload(int index) {
    long validTime = 1700000000L + (index * 60L);
    return """
        {
          "metadata": {
            "id": "KNYC:9:US",
            "units": "e",
            "language": "en-US",
            "transaction_id": "txn-%d",
            "version": "1",
            "expire_time_gmt": %d
          },
          "observations": [
            {
              "obs_id": "KLGA",
              "key": "KLGA",
              "obs_name": "LaGuardia Airport",
              "valid_time_gmt": %d,
              "temp": 60,
              "dewPt": 50,
              "rh": 70,
              "pressure": 30.01,
              "wspd": 10,
              "wx_phrase": "Cloudy",
              "class": "observation"
            }
          ]
        }
        """.formatted(index, validTime + 600L, validTime);
  }
}

