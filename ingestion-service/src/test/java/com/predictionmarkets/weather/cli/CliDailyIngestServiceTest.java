package com.predictionmarkets.weather.cli;

import static org.assertj.core.api.Assertions.assertThat;

import com.predictionmarkets.weather.common.Hashing;
import com.predictionmarkets.weather.models.KalshiSeries;
import com.predictionmarkets.weather.models.StationMappingStatus;
import com.predictionmarkets.weather.models.StationRegistry;
import com.predictionmarkets.weather.repository.CliDailyRepository;
import com.predictionmarkets.weather.repository.IngestCheckpointRepository;
import com.predictionmarkets.weather.repository.KalshiSeriesRepository;
import com.predictionmarkets.weather.repository.MosAsofFeatureRepository;
import com.predictionmarkets.weather.repository.MosForecastValueUpsertRepository;
import com.predictionmarkets.weather.repository.MosDailyValueRepository;
import com.predictionmarkets.weather.repository.MosRunRepository;
import com.predictionmarkets.weather.repository.StationRegistryRepository;
import java.io.IOException;
import java.time.Instant;
import java.time.LocalDate;
import java.time.Year;
import java.time.format.DateTimeFormatter;
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
class CliDailyIngestServiceTest {
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
  private CliDailyIngestService ingestService;

  @Autowired
  private CliDailyRepository cliDailyRepository;

  @Autowired
  private MosAsofFeatureRepository mosAsofFeatureRepository;

  @Autowired
  private MosRunRepository mosRunRepository;

  @Autowired
  private IngestCheckpointRepository checkpointRepository;

  @Autowired
  private StationRegistryRepository stationRegistryRepository;

  @Autowired
  private MosForecastValueUpsertRepository mosForecastValueUpsertRepository;

  @Autowired
  private MosDailyValueRepository mosDailyValueRepository;

  @Autowired
  private KalshiSeriesRepository kalshiSeriesRepository;

  @BeforeEach
  void setUp() {
    checkpointRepository.deleteAll();
    mosAsofFeatureRepository.deleteAll();
    mosRunRepository.deleteAll();
    cliDailyRepository.deleteAll();
    mosForecastValueUpsertRepository.deleteAll();
    mosDailyValueRepository.deleteAll();
    stationRegistryRepository.deleteAll();
    kalshiSeriesRepository.deleteAll();
    seedStation();
  }

  @AfterAll
  static void shutdown() throws IOException {
    SERVER.shutdown();
  }

  @Test
  void ingestsYearPayloadAndUpsertsRows() {
    String payload = buildCliJson("KMIA", 2025);
    SERVER.enqueue(new MockResponse()
        .setResponseCode(200)
        .setHeader("Content-Type", "application/json")
        .setBody(payload));

    int upserted = ingestService.ingestYear("KMIA", 2025);

    LocalDate start = LocalDate.of(2025, 1, 1);
    LocalDate end = LocalDate.of(2025, 12, 31);
    long rowCount = cliDailyRepository.countByIdStationIdAndIdTargetDateLocalBetween("KMIA", start, end);
    long tmaxCount = cliDailyRepository.countByIdStationIdAndTmaxFIsNotNullAndIdTargetDateLocalBetween(
        "KMIA", start, end);

    assertThat(upserted).isGreaterThan(300);
    assertThat(rowCount).isGreaterThan(300);
    assertThat(tmaxCount).isGreaterThan(300);
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

  private String buildCliJson(String stationId, int year) {
    int daysInYear = Year.of(year).length();
    LocalDate start = LocalDate.of(year, 1, 1);
    StringBuilder builder = new StringBuilder();
    builder.append("{\"results\":[");
    DateTimeFormatter dateFormatter = DateTimeFormatter.ISO_LOCAL_DATE;
    for (int day = 0; day < daysInYear; day++) {
      LocalDate date = start.plusDays(day);
      int high = 80 + (day % 5);
      int low = 60 + (day % 5);
      String issuedAt = date.plusDays(1).format(DateTimeFormatter.BASIC_ISO_DATE) + "0600";
      if (day > 0) {
        builder.append(',');
      }
      builder.append("{\"station\":\"").append(stationId).append("\",");
      builder.append("\"valid\":\"").append(date.format(dateFormatter)).append("\",");
      builder.append("\"high\":").append(high).append(',');
      builder.append("\"low\":").append(low).append(',');
      builder.append("\"product\":\"").append(issuedAt).append("-KMFL-CDUS42-CLIMIA\"}");
    }
    builder.append("],\"generated_at\":\"").append(Instant.now().toString()).append("\"}");
    return builder.toString();
  }
}
