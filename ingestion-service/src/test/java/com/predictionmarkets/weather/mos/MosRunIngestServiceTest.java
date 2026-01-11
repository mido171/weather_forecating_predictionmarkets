package com.predictionmarkets.weather.mos;

import static org.assertj.core.api.Assertions.assertThat;

import com.predictionmarkets.weather.common.Hashing;
import com.predictionmarkets.weather.models.KalshiSeries;
import com.predictionmarkets.weather.models.MosModel;
import com.predictionmarkets.weather.models.MosRun;
import com.predictionmarkets.weather.models.StationMappingStatus;
import com.predictionmarkets.weather.models.StationRegistry;
import com.predictionmarkets.weather.repository.CliDailyRepository;
import com.predictionmarkets.weather.repository.IngestCheckpointRepository;
import com.predictionmarkets.weather.repository.KalshiSeriesRepository;
import com.predictionmarkets.weather.repository.MosAsofFeatureRepository;
import com.predictionmarkets.weather.repository.MosRunRepository;
import com.predictionmarkets.weather.repository.StationRegistryRepository;
import java.io.IOException;
import java.time.Instant;
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
class MosRunIngestServiceTest {
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
  private MosRunIngestService ingestService;

  @Autowired
  private MosRunRepository mosRunRepository;

  @Autowired
  private MosAsofFeatureRepository mosAsofFeatureRepository;

  @Autowired
  private CliDailyRepository cliDailyRepository;

  @Autowired
  private IngestCheckpointRepository checkpointRepository;

  @Autowired
  private StationRegistryRepository stationRegistryRepository;

  @Autowired
  private KalshiSeriesRepository kalshiSeriesRepository;

  @BeforeEach
  void setUp() {
    checkpointRepository.deleteAll();
    mosAsofFeatureRepository.deleteAll();
    mosRunRepository.deleteAll();
    cliDailyRepository.deleteAll();
    stationRegistryRepository.deleteAll();
    kalshiSeriesRepository.deleteAll();
    seedStation();
  }

  @AfterAll
  static void shutdown() throws IOException {
    SERVER.shutdown();
  }

  @Test
  void ingestsWindowAndIsIdempotent() {
    String payload = buildMosJson();
    SERVER.enqueue(new MockResponse()
        .setResponseCode(200)
        .setHeader("Content-Type", "application/json")
        .setBody(payload));

    Instant start = Instant.parse("2023-12-14T00:00:00Z");
    Instant end = Instant.parse("2023-12-15T00:00:00Z");
    int upserted = ingestService.ingestWindow("KMIA", MosModel.GFS, start, end);

    assertThat(upserted).isEqualTo(2);
    assertThat(mosRunRepository.countByIdStationIdAndIdModel("KMIA", MosModel.GFS)).isEqualTo(2);

    String expectedHash = Hashing.sha256Hex(payload);
    List<MosRun> runs = mosRunRepository.findAll();
    assertThat(runs).allSatisfy(run -> {
      assertThat(run.getRawPayloadHash()).isEqualTo(expectedHash);
      assertThat(run.getRetrievedAtUtc()).isNotNull();
    });

    SERVER.enqueue(new MockResponse()
        .setResponseCode(200)
        .setHeader("Content-Type", "application/json")
        .setBody(payload));
    ingestService.ingestWindow("KMIA", MosModel.GFS, start, end);

    assertThat(mosRunRepository.countByIdStationIdAndIdModel("KMIA", MosModel.GFS)).isEqualTo(2);
  }

  @Test
  void retriesOnTransientFailure() {
    String payload = buildMosJson();
    int initialRequests = SERVER.getRequestCount();
    SERVER.enqueue(new MockResponse().setResponseCode(503));
    SERVER.enqueue(new MockResponse()
        .setResponseCode(200)
        .setHeader("Content-Type", "application/json")
        .setBody(payload));

    Instant start = Instant.parse("2023-12-14T00:00:00Z");
    Instant end = Instant.parse("2023-12-15T00:00:00Z");
    ingestService.ingestWindow("KMIA", MosModel.GFS, start, end);

    assertThat(SERVER.getRequestCount()).isEqualTo(initialRequests + 2);
  }

  @Test
  void handlesEmptyPayload() {
    SERVER.enqueue(new MockResponse()
        .setResponseCode(200)
        .setHeader("Content-Type", "application/json")
        .setBody("[]"));

    Instant start = Instant.parse("2023-12-14T00:00:00Z");
    Instant end = Instant.parse("2023-12-15T00:00:00Z");
    int upserted = ingestService.ingestWindow("KMIA", MosModel.GFS, start, end);

    assertThat(upserted).isEqualTo(0);
    assertThat(mosRunRepository.count()).isZero();
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

  private String buildMosJson() {
    return "[" +
        "{\"runtime\":1702512000000,\"ftime\":1702533600000,\"model\":\"GFS\",\"station\":\"KMIA\",\"n_x\":72.0}," +
        "{\"runtime\":1702512000000,\"ftime\":1702544400000,\"model\":\"GFS\",\"station\":\"KMIA\",\"n_x\":71.0}," +
        "{\"runtime\":1702598400000,\"ftime\":1702620000000,\"model\":\"GFS\",\"station\":\"KMIA\",\"n_x\":75.0}" +
        "]";
  }
}
