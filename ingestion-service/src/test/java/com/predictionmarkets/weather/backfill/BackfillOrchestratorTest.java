package com.predictionmarkets.weather.backfill;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;

import com.predictionmarkets.weather.common.Hashing;
import com.predictionmarkets.weather.cli.CliDailyIngestService;
import com.predictionmarkets.weather.kalshi.KalshiSeriesResolver;
import com.predictionmarkets.weather.models.IngestCheckpoint;
import com.predictionmarkets.weather.models.IngestCheckpointStatus;
import com.predictionmarkets.weather.models.KalshiSeries;
import com.predictionmarkets.weather.models.MosModel;
import com.predictionmarkets.weather.models.StationMappingStatus;
import com.predictionmarkets.weather.models.StationRegistry;
import com.predictionmarkets.weather.mos.MosAsofMaterializeService;
import com.predictionmarkets.weather.mos.MosAsofFeatureReportService;
import com.predictionmarkets.weather.mos.MosRunIngestService;
import com.predictionmarkets.weather.repository.IngestCheckpointRepository;
import com.predictionmarkets.weather.repository.KalshiSeriesRepository;
import com.predictionmarkets.weather.repository.MosAsofFeatureRepository;
import com.predictionmarkets.weather.repository.MosRunRepository;
import com.predictionmarkets.weather.repository.CliDailyRepository;
import com.predictionmarkets.weather.repository.StationRegistryRepository;
import java.time.Instant;
import java.time.LocalDate;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.context.ActiveProfiles;

@SpringBootTest
@ActiveProfiles("test")
class BackfillOrchestratorTest {
  @Autowired
  private BackfillOrchestrator orchestrator;

  @Autowired
  private IngestCheckpointRepository checkpointRepository;

  @Autowired
  private MosAsofFeatureRepository mosAsofFeatureRepository;

  @Autowired
  private MosRunRepository mosRunRepository;

  @Autowired
  private CliDailyRepository cliDailyRepository;

  @Autowired
  private StationRegistryRepository stationRegistryRepository;

  @Autowired
  private KalshiSeriesRepository kalshiSeriesRepository;

  @MockBean
  private CliDailyIngestService cliDailyIngestService;

  @MockBean
  private MosRunIngestService mosRunIngestService;

  @MockBean
  private MosAsofMaterializeService mosAsofMaterializeService;

  @MockBean
  private MosAsofFeatureReportService mosAsofFeatureReportService;

  @MockBean
  private KalshiSeriesResolver kalshiSeriesResolver;

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

  @Test
  void cliIngestYearResumesFromCheckpoint() {
    IngestCheckpoint checkpoint = new IngestCheckpoint();
    checkpoint.setJobName(BackfillJobType.CLI_INGEST_YEAR.jobName());
    checkpoint.setStationId("KMIA");
    checkpoint.setCursorDate(LocalDate.of(2023, 12, 31));
    checkpoint.setStatus(IngestCheckpointStatus.RUNNING);
    checkpoint.setUpdatedAtUtc(Instant.now());
    checkpointRepository.save(checkpoint);

    BackfillRequest request = new BackfillRequest(
        BackfillJobType.CLI_INGEST_YEAR,
        List.of("KXHIGHMIA"),
        LocalDate.of(2023, 1, 1),
        LocalDate.of(2024, 12, 31),
        null,
        List.of(MosModel.GFS),
        1);

    orchestrator.run(request);

    verify(cliDailyIngestService).ingestRange(
        "KMIA", LocalDate.of(2024, 1, 1), LocalDate.of(2024, 12, 31));
    verifyNoMoreInteractions(cliDailyIngestService);

    Optional<IngestCheckpoint> updated = checkpointRepository
        .findByJobNameAndStationIdAndModelIsNull(BackfillJobType.CLI_INGEST_YEAR.jobName(), "KMIA");
    assertThat(updated).isPresent();
    assertThat(updated.get().getStatus()).isEqualTo(IngestCheckpointStatus.COMPLETE);
    assertThat(updated.get().getCursorDate()).isEqualTo(LocalDate.of(2024, 12, 31));
  }

  @Test
  void mosIngestWindowResumesFromCheckpoint() {
    IngestCheckpoint checkpoint = new IngestCheckpoint();
    checkpoint.setJobName(BackfillJobType.MOS_INGEST_WINDOW.jobName());
    checkpoint.setStationId("KMIA");
    checkpoint.setModel(MosModel.GFS);
    checkpoint.setCursorRuntimeUtc(Instant.parse("2023-12-15T00:00:00Z"));
    checkpoint.setStatus(IngestCheckpointStatus.RUNNING);
    checkpoint.setUpdatedAtUtc(Instant.now());
    checkpointRepository.save(checkpoint);

    BackfillRequest request = new BackfillRequest(
        BackfillJobType.MOS_INGEST_WINDOW,
        List.of("KXHIGHMIA"),
        LocalDate.of(2023, 12, 14),
        LocalDate.of(2023, 12, 15),
        null,
        List.of(MosModel.GFS),
        1);

    orchestrator.run(request);

    verify(mosRunIngestService).ingestWindow(
        "KMIA",
        MosModel.GFS,
        Instant.parse("2023-12-15T00:00:00Z"),
        Instant.parse("2023-12-16T00:00:00Z"));

    Optional<IngestCheckpoint> updated = checkpointRepository
        .findByJobNameAndStationIdAndModel(
            BackfillJobType.MOS_INGEST_WINDOW.jobName(), "KMIA", MosModel.GFS);
    assertThat(updated).isPresent();
    assertThat(updated.get().getStatus()).isEqualTo(IngestCheckpointStatus.COMPLETE);
    assertThat(updated.get().getCursorRuntimeUtc()).isEqualTo(Instant.parse("2023-12-16T00:00:00Z"));
  }

  @Test
  void mosIngestWindowMarksFailureWithErrorDetails() {
    doThrow(new RuntimeException("boom"))
        .when(mosRunIngestService)
        .ingestWindow(anyString(), eq(MosModel.GFS), any(Instant.class), any(Instant.class));

    BackfillRequest request = new BackfillRequest(
        BackfillJobType.MOS_INGEST_WINDOW,
        List.of("KXHIGHMIA"),
        LocalDate.of(2023, 12, 14),
        LocalDate.of(2023, 12, 14),
        null,
        List.of(MosModel.GFS),
        1);

    assertThatThrownBy(() -> orchestrator.run(request))
        .isInstanceOf(RuntimeException.class)
        .hasMessageContaining("boom");

    Optional<IngestCheckpoint> updated = checkpointRepository
        .findByJobNameAndStationIdAndModel(
            BackfillJobType.MOS_INGEST_WINDOW.jobName(), "KMIA", MosModel.GFS);
    assertThat(updated).isPresent();
    assertThat(updated.get().getStatus()).isEqualTo(IngestCheckpointStatus.FAILED);
    assertThat(updated.get().getErrorDetails()).contains("RuntimeException: boom");
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
}
