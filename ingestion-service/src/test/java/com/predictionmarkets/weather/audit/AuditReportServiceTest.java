package com.predictionmarkets.weather.audit;

import static org.assertj.core.api.Assertions.assertThat;

import com.predictionmarkets.weather.common.Hashing;
import com.predictionmarkets.weather.models.AsofPolicy;
import com.predictionmarkets.weather.models.CliDaily;
import com.predictionmarkets.weather.models.CliDailyId;
import com.predictionmarkets.weather.models.KalshiSeries;
import com.predictionmarkets.weather.models.MosAsofFeature;
import com.predictionmarkets.weather.models.MosAsofFeatureId;
import com.predictionmarkets.weather.models.MosModel;
import com.predictionmarkets.weather.models.MosRun;
import com.predictionmarkets.weather.models.MosRunId;
import com.predictionmarkets.weather.models.StationMappingStatus;
import com.predictionmarkets.weather.models.StationRegistry;
import com.predictionmarkets.weather.repository.AsofPolicyRepository;
import com.predictionmarkets.weather.repository.CliDailyRepository;
import com.predictionmarkets.weather.repository.KalshiSeriesRepository;
import com.predictionmarkets.weather.repository.MosAsofFeatureRepository;
import com.predictionmarkets.weather.repository.MosForecastValueUpsertRepository;
import com.predictionmarkets.weather.repository.MosDailyValueRepository;
import com.predictionmarkets.weather.repository.MosRunRepository;
import com.predictionmarkets.weather.repository.StationRegistryRepository;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneOffset;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;

@SpringBootTest
@ActiveProfiles("test")
class AuditReportServiceTest {
  @Autowired
  private AuditReportService auditReportService;

  @Autowired
  private MosAsofFeatureRepository mosAsofFeatureRepository;

  @Autowired
  private MosRunRepository mosRunRepository;

  @Autowired
  private CliDailyRepository cliDailyRepository;

  @Autowired
  private StationRegistryRepository stationRegistryRepository;

  @Autowired
  private MosForecastValueUpsertRepository mosForecastValueUpsertRepository;

  @Autowired
  private MosDailyValueRepository mosDailyValueRepository;

  @Autowired
  private KalshiSeriesRepository kalshiSeriesRepository;

  @Autowired
  private AsofPolicyRepository asofPolicyRepository;

  @TempDir
  Path tempDir;

  private Long asofPolicyId;

  @BeforeEach
  void setUp() {
    mosAsofFeatureRepository.deleteAll();
    mosRunRepository.deleteAll();
    cliDailyRepository.deleteAll();
    mosForecastValueUpsertRepository.deleteAll();
    mosDailyValueRepository.deleteAll();
    stationRegistryRepository.deleteAll();
    kalshiSeriesRepository.deleteAll();
    asofPolicyRepository.deleteAll();
    seedStation();
    seedPolicy();
  }

  @Test
  void generatesReportsAndFlagsViolations() throws Exception {
    seedCliDaily(LocalDate.of(2023, 12, 14));
    seedMosRun(LocalDate.of(2023, 12, 14));
    seedMosRun(LocalDate.of(2023, 12, 15));
    seedLeakyFeature();

    AuditRequest request = new AuditRequest(
        List.of("KMIA"),
        LocalDate.of(2023, 12, 14),
        LocalDate.of(2023, 12, 15),
        asofPolicyId,
        List.of(MosModel.GFS),
        10,
        5);
    AuditReportArtifacts artifacts = auditReportService.generateAndWriteReport(request, tempDir);

    AuditReport report = artifacts.report();
    assertThat(report.noLeakage().status()).isEqualTo(AuditReport.Status.FAIL);
    assertThat(report.stationCoverage().status()).isEqualTo(AuditReport.Status.FAIL);
    assertThat(report.mosAvailability().status()).isEqualTo(AuditReport.Status.PASS);
    assertThat(report.featureCoverage().status()).isEqualTo(AuditReport.Status.FAIL);
    assertThat(report.alignment().status()).isEqualTo(AuditReport.Status.PASS);

    assertThat(Files.exists(artifacts.markdownPath())).isTrue();
    assertThat(Files.exists(artifacts.jsonPath())).isTrue();
    String markdown = Files.readString(artifacts.markdownPath());
    assertThat(markdown).contains("Data Quality Audit Report");
    String json = Files.readString(artifacts.jsonPath());
    assertThat(json).contains("\"schemaVersion\"");
  }

  private void seedStation() {
    Instant now = Instant.parse("2023-12-14T00:00:00Z");
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

  private void seedPolicy() {
    AsofPolicy policy = new AsofPolicy();
    policy.setName("default");
    policy.setAsofLocalTime(LocalTime.of(23, 0));
    policy.setEnabled(true);
    asofPolicyId = asofPolicyRepository.save(policy).getId();
  }

  private void seedCliDaily(LocalDate date) {
    CliDaily daily = new CliDaily();
    daily.setId(new CliDailyId("KMIA", date));
    daily.setTmaxF(new BigDecimal("75.0"));
    daily.setTminF(new BigDecimal("60.0"));
    daily.setReportIssuedAtUtc(Instant.parse("2023-12-14T12:00:00Z"));
    daily.setRawPayloadHash(Hashing.sha256Hex("cli"));
    daily.setRetrievedAtUtc(Instant.parse("2023-12-14T12:00:00Z"));
    daily.setUpdatedAtUtc(Instant.parse("2023-12-14T12:00:00Z"));
    cliDailyRepository.save(daily);
  }

  private void seedMosRun(LocalDate date) {
    Instant runtimeUtc = date.atStartOfDay().toInstant(ZoneOffset.UTC);
    MosRun run = new MosRun();
    run.setId(new MosRunId("KMIA", MosModel.GFS, runtimeUtc));
    run.setRawPayloadHash(Hashing.sha256Hex("mos-" + date));
    run.setRetrievedAtUtc(runtimeUtc.plusSeconds(3600));
    mosRunRepository.save(run);
  }

  private void seedLeakyFeature() {
    MosAsofFeature feature = new MosAsofFeature();
    feature.setId(new MosAsofFeatureId("KMIA",
        LocalDate.of(2023, 12, 15), asofPolicyId, MosModel.GFS));
    feature.setAsofUtc(Instant.parse("2023-12-15T04:00:00Z"));
    feature.setAsofLocal(LocalDateTime.of(2023, 12, 14, 23, 0));
    feature.setStationZoneid("America/New_York");
    feature.setChosenRuntimeUtc(Instant.parse("2023-12-15T12:00:00Z"));
    feature.setTmaxF(new BigDecimal("77.0"));
    feature.setRawPayloadHashRef(Hashing.sha256Hex("mos-run"));
    feature.setRetrievedAtUtc(Instant.parse("2023-12-15T12:00:00Z"));
    mosAsofFeatureRepository.save(feature);
  }
}
