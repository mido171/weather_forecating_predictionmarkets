package com.predictionmarkets.weather.mos;

import static org.assertj.core.api.Assertions.assertThat;

import com.predictionmarkets.weather.common.Hashing;
import com.predictionmarkets.weather.models.AsofPolicy;
import com.predictionmarkets.weather.models.KalshiSeries;
import com.predictionmarkets.weather.models.MosAsofFeature;
import com.predictionmarkets.weather.models.MosModel;
import com.predictionmarkets.weather.models.MosRun;
import com.predictionmarkets.weather.models.MosRunId;
import com.predictionmarkets.weather.models.StationMappingStatus;
import com.predictionmarkets.weather.models.StationRegistry;
import com.predictionmarkets.weather.repository.AsofPolicyRepository;
import com.predictionmarkets.weather.repository.CliDailyRepository;
import com.predictionmarkets.weather.repository.IngestCheckpointRepository;
import com.predictionmarkets.weather.repository.KalshiSeriesRepository;
import com.predictionmarkets.weather.repository.MosAsofFeatureRepository;
import com.predictionmarkets.weather.repository.MosForecastValueUpsertRepository;
import com.predictionmarkets.weather.repository.MosRunRepository;
import com.predictionmarkets.weather.repository.StationRegistryRepository;
import java.io.IOException;
import java.math.BigDecimal;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.List;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;

@SpringBootTest
@ActiveProfiles("test")
class MosAsofMaterializeServiceTest {
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
    registry.add("iem.base-url", () -> SERVER.url("/").toString());
  }

  @Autowired
  private MosAsofMaterializeService materializeService;

  @Autowired
  private MosAsofFeatureRepository mosAsofFeatureRepository;

  @Autowired
  private MosRunRepository mosRunRepository;

  @Autowired
  private CliDailyRepository cliDailyRepository;

  @Autowired
  private IngestCheckpointRepository ingestCheckpointRepository;

  @Autowired
  private StationRegistryRepository stationRegistryRepository;

  @Autowired
  private MosForecastValueUpsertRepository mosForecastValueUpsertRepository;

  @Autowired
  private KalshiSeriesRepository kalshiSeriesRepository;

  @Autowired
  private AsofPolicyRepository asofPolicyRepository;

  private Long defaultPolicyId;

  @BeforeEach
  void setUp() {
    mosAsofFeatureRepository.deleteAll();
    mosRunRepository.deleteAll();
    cliDailyRepository.deleteAll();
    ingestCheckpointRepository.deleteAll();
    mosForecastValueUpsertRepository.deleteAll();
    stationRegistryRepository.deleteAll();
    kalshiSeriesRepository.deleteAll();
    asofPolicyRepository.deleteAll();
    seedStation();
    seedAsofPolicy();
  }

  @AfterAll
  static void shutdown() throws IOException {
    SERVER.shutdown();
  }

  @Test
  void materializesTmaxFromLatestEligibleRunAndIsIdempotent() {
    Instant chosenRuntime = Instant.parse("2023-12-14T00:00:00Z");
    Instant futureRuntime = Instant.parse("2023-12-15T12:00:00Z");
    Instant retrievedAt = Instant.parse("2023-12-14T01:00:00Z");
    seedMosRun("KMIA", MosModel.GFS, chosenRuntime, retrievedAt, "hash-one");
    seedMosRun("KMIA", MosModel.GFS, futureRuntime, Instant.parse("2023-12-15T13:00:00Z"), "hash-two");

    String payload = buildMosJson("KMIA", MosModel.GFS, chosenRuntime, List.of(
        new ForecastEntry(Instant.parse("2023-12-15T06:00:00Z"), "72/60"),
        new ForecastEntry(Instant.parse("2023-12-15T09:00:00Z"), "74/61"),
        new ForecastEntry(Instant.parse("2023-12-16T06:00:00Z"), "76/62")
    ));
    SERVER.enqueue(new MockResponse()
        .setResponseCode(200)
        .setHeader("Content-Type", "application/json")
        .setBody(payload));
    SERVER.enqueue(new MockResponse()
        .setResponseCode(200)
        .setHeader("Content-Type", "application/json")
        .setBody(payload));

    LocalDate targetDate = LocalDate.of(2023, 12, 15);
    materializeService.materializeForTargetDate(
        "KMIA", targetDate, defaultPolicyId, List.of(MosModel.GFS));
    materializeService.materializeForTargetDate(
        "KMIA", targetDate, defaultPolicyId, List.of(MosModel.GFS));

    assertThat(mosAsofFeatureRepository.count()).isEqualTo(1);
    MosAsofFeature feature = mosAsofFeatureRepository
        .findByIdStationIdAndIdTargetDateLocalAndIdAsofPolicyIdAndIdModel(
            "KMIA", targetDate, defaultPolicyId, MosModel.GFS)
        .orElseThrow();

    assertThat(feature.getChosenRuntimeUtc()).isEqualTo(chosenRuntime);
    assertThat(feature.getTmaxF()).isEqualByComparingTo(new BigDecimal("74"));
    assertThat(feature.getMissingReason()).isNull();
    assertThat(feature.getRawPayloadHashRef()).isEqualTo("hash-one");
    assertThat(feature.getRetrievedAtUtc()).isEqualTo(retrievedAt);
    assertThat(feature.getAsofUtc()).isEqualTo(Instant.parse("2023-12-15T04:00:00Z"));
    assertThat(feature.getAsofLocal()).isEqualTo(LocalDateTime.of(2023, 12, 14, 23, 0));
    assertThat(feature.getStationZoneid()).isEqualTo("America/New_York");
  }

  @Test
  void writesMissingRowWhenNoEligibleRun() {
    LocalDate targetDate = LocalDate.of(2023, 12, 10);
    materializeService.materializeForTargetDate(
        "KMIA", targetDate, defaultPolicyId, List.of(MosModel.GFS));

    MosAsofFeature feature = mosAsofFeatureRepository
        .findByIdStationIdAndIdTargetDateLocalAndIdAsofPolicyIdAndIdModel(
            "KMIA", targetDate, defaultPolicyId, MosModel.GFS)
        .orElseThrow();

    assertThat(feature.getChosenRuntimeUtc()).isNull();
    assertThat(feature.getTmaxF()).isNull();
    assertThat(feature.getMissingReason()).isEqualTo("NO_ELIGIBLE_RUN");
    assertThat(feature.getRawPayloadHashRef()).isNull();
    assertThat(feature.getRetrievedAtUtc()).isNotNull();
  }

  private void seedStation() {
    Instant now = Instant.now();
    KalshiSeries series = new KalshiSeries();
    series.setSeriesTicker("KXHIGHMIA");
    series.setTitle("Test Series");
    series.setCategory("weather");
    series.setSettlementSourceName("NWS CLI");
    series.setSettlementSourceUrl("https://example.test/cli");
    series.setRetrievedAtUtc(now);
    series.setRawPayloadHash(Hashing.sha256Hex("kalshi"));
    kalshiSeriesRepository.save(series);

    StationRegistry station = new StationRegistry();
    station.setStationId("KMIA");
    station.setIssuedby("MIA");
    station.setWfoSite("MFL");
    station.setSeriesTicker("KXHIGHMIA");
    station.setZoneId("America/New_York");
    station.setStandardOffsetMinutes(-300);
    station.setMappingStatus(StationMappingStatus.AUTO_OK);
    station.setCreatedAtUtc(now);
    station.setUpdatedAtUtc(now);
    stationRegistryRepository.save(station);
  }

  private void seedAsofPolicy() {
    AsofPolicy policy = new AsofPolicy();
    policy.setName("default");
    policy.setAsofLocalTime(LocalTime.of(23, 0));
    policy.setEnabled(true);
    defaultPolicyId = asofPolicyRepository.save(policy).getId();
  }

  private void seedMosRun(String stationId,
                          MosModel model,
                          Instant runtimeUtc,
                          Instant retrievedAtUtc,
                          String rawPayloadHash) {
    MosRun run = new MosRun();
    run.setId(new MosRunId(stationId, model, runtimeUtc));
    run.setRetrievedAtUtc(retrievedAtUtc);
    run.setRawPayloadHash(rawPayloadHash);
    mosRunRepository.save(run);
  }

  private String buildMosJson(String stationId,
                              MosModel model,
                              Instant runtimeUtc,
                              List<ForecastEntry> entries) {
    StringBuilder builder = new StringBuilder();
    builder.append('[');
    for (int i = 0; i < entries.size(); i++) {
      ForecastEntry entry = entries.get(i);
      if (i > 0) {
        builder.append(',');
      }
      builder.append("{\"runtime\":").append(runtimeUtc.toEpochMilli()).append(',');
      builder.append("\"ftime\":").append(entry.forecastTimeUtc().toEpochMilli()).append(',');
      builder.append("\"model\":\"").append(model.name()).append("\",");
      builder.append("\"station\":\"").append(stationId).append("\",");
      builder.append("\"n_x\":\"").append(entry.nxValue()).append("\"}");
    }
    builder.append(']');
    return builder.toString();
  }

  private record ForecastEntry(Instant forecastTimeUtc, String nxValue) {
  }
}
