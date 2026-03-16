package com.predictionmarkets.weather.pilot.jobs;

import com.predictionmarkets.weather.pilot.catalog.JobRunService;
import com.predictionmarkets.weather.pilot.config.PilotConfigLoader;
import com.predictionmarkets.weather.pilot.config.PilotIngestionProperties;
import com.predictionmarkets.weather.pilot.config.StationConfig;
import com.predictionmarkets.weather.pilot.metrics.JobMetricsAccumulator;
import com.predictionmarkets.weather.pilot.metrics.MetricsService;
import com.predictionmarkets.weather.pilot.source.iem.IemAfosCliFetcher;
import com.predictionmarkets.weather.pilot.source.iem.IemAsos1MinFetcher;
import com.predictionmarkets.weather.pilot.source.iem.IemAsosFetcher;
import com.predictionmarkets.weather.pilot.source.iem.IemCliJsonFetcher;
import com.predictionmarkets.weather.pilot.source.iem.IemClimodatFetcher;
import com.predictionmarkets.weather.pilot.source.iem.IemMosFetcher;
import com.predictionmarkets.weather.pilot.source.ncei.ClimateNormalsFetcher;
import com.predictionmarkets.weather.pilot.source.ncei.Lcdv2Fetcher;
import com.predictionmarkets.weather.pilot.source.ncei.ThreadexAcisFetcher;
import com.predictionmarkets.weather.pilot.source.nws.NdfdHistoricalCatalogFetcher;
import com.predictionmarkets.weather.pilot.source.nws.NdfdLivePointFetcher;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Map;
import org.springframework.stereotype.Service;

@Service
public class BackfillLightweightSourcesJob {
  private final PilotConfigLoader configLoader;
  private final PilotIngestionProperties properties;
  private final MetricsService metricsService;
  private final JobRunService jobRunService;
  private final IemCliJsonFetcher cliJsonFetcher;
  private final IemAfosCliFetcher afosCliFetcher;
  private final IemAsosFetcher asosFetcher;
  private final IemAsos1MinFetcher asos1MinFetcher;
  private final IemMosFetcher mosFetcher;
  private final NdfdLivePointFetcher ndfdLivePointFetcher;
  private final NdfdHistoricalCatalogFetcher ndfdHistoricalCatalogFetcher;
  private final ThreadexAcisFetcher threadexAcisFetcher;
  private final ClimateNormalsFetcher climateNormalsFetcher;
  private final IemClimodatFetcher iemClimodatFetcher;
  private final Lcdv2Fetcher lcdv2Fetcher;

  public BackfillLightweightSourcesJob(PilotConfigLoader configLoader,
                                       PilotIngestionProperties properties,
                                       MetricsService metricsService,
                                       JobRunService jobRunService,
                                       IemCliJsonFetcher cliJsonFetcher,
                                       IemAfosCliFetcher afosCliFetcher,
                                       IemAsosFetcher asosFetcher,
                                       IemAsos1MinFetcher asos1MinFetcher,
                                       IemMosFetcher mosFetcher,
                                       NdfdLivePointFetcher ndfdLivePointFetcher,
                                       NdfdHistoricalCatalogFetcher ndfdHistoricalCatalogFetcher,
                                       ThreadexAcisFetcher threadexAcisFetcher,
                                       ClimateNormalsFetcher climateNormalsFetcher,
                                       IemClimodatFetcher iemClimodatFetcher,
                                       Lcdv2Fetcher lcdv2Fetcher) {
    this.configLoader = configLoader;
    this.properties = properties;
    this.metricsService = metricsService;
    this.jobRunService = jobRunService;
    this.cliJsonFetcher = cliJsonFetcher;
    this.afosCliFetcher = afosCliFetcher;
    this.asosFetcher = asosFetcher;
    this.asos1MinFetcher = asos1MinFetcher;
    this.mosFetcher = mosFetcher;
    this.ndfdLivePointFetcher = ndfdLivePointFetcher;
    this.ndfdHistoricalCatalogFetcher = ndfdHistoricalCatalogFetcher;
    this.threadexAcisFetcher = threadexAcisFetcher;
    this.climateNormalsFetcher = climateNormalsFetcher;
    this.iemClimodatFetcher = iemClimodatFetcher;
    this.lcdv2Fetcher = lcdv2Fetcher;
  }

  public String run() {
    StationConfig station = configLoader.requireDefaultStation();
    String runId = jobRunService.startRun("backfillLightweightSourcesJob", station.getStationKey());
    JobMetricsAccumulator metrics = metricsService.newAccumulator();
    try {
      boolean hadFailures = false;
      jobRunService.logStructuredEvent("backfillLightweightSourcesJob", runId, station.getStationKey(),
          "backfillLightweightSourcesJob", "phase_start", "RUNNING",
          Map.of("phase", "cli_and_afos",
              "start_year", properties.getJobs().getSmokeCliStartYear(),
              "end_year", properties.getJobs().getSmokeCliEndYear()));
      hadFailures |= !runPhase("backfillLightweightSourcesJob", runId, station.getStationKey(),
          "cli_and_afos", metrics, () -> {
            for (int year = properties.getJobs().getSmokeCliStartYear();
                 year <= properties.getJobs().getSmokeCliEndYear();
                 year++) {
              cliJsonFetcher.ingestYear("backfillLightweightSourcesJob", runId, station, year, metrics);
              afosCliFetcher.ingestYear("backfillLightweightSourcesJob", runId, station, year, metrics);
            }
          });
      ZoneId stationZone = ZoneId.of(station.getTimezone());
      LocalDate stationToday = LocalDate.now(stationZone);
      LocalDate hourlyStart = stationToday.minusDays(properties.getJobs().getSmokeHourlyLookbackDays());
      jobRunService.logStructuredEvent("backfillLightweightSourcesJob", runId, station.getStationKey(),
          "iem_asos_hourly", "phase_start", "RUNNING",
          Map.of("phase", "asos_hourly",
              "start_date_local", hourlyStart.toString(),
              "end_date_local", stationToday.toString()));
      hadFailures |= !runPhase("backfillLightweightSourcesJob", runId, station.getStationKey(),
          "asos_hourly", metrics,
          () -> asosFetcher.ingestRange("backfillLightweightSourcesJob", runId, station, hourlyStart, stationToday, metrics));
      Instant oneMinEnd = stationToday.minusDays(1).atStartOfDay(stationZone).toInstant();
      Instant oneMinStart = oneMinEnd.minusSeconds(properties.getJobs().getSmokeOneMinuteLookbackDays() * 24L * 3600L);
      jobRunService.logStructuredEvent("backfillLightweightSourcesJob", runId, station.getStationKey(),
          "iem_asos_1min", "phase_start", "RUNNING",
          Map.of("phase", "asos_1min",
              "start_time_utc", oneMinStart.toString(),
              "end_time_utc", oneMinEnd.toString()));
      hadFailures |= !runPhase("backfillLightweightSourcesJob", runId, station.getStationKey(),
          "asos_1min", metrics,
          () -> asos1MinFetcher.ingestRange("backfillLightweightSourcesJob", runId, station, oneMinStart, oneMinEnd, metrics));
      jobRunService.logStructuredEvent("backfillLightweightSourcesJob", runId, station.getStationKey(),
          "iem_mos", "phase_start", "RUNNING",
          Map.of("phase", "mos",
              "lookback_days", properties.getJobs().getSmokeMosLookbackDays(),
              "models", List.of("GFS", "NAM")));
      hadFailures |= !runPhase("backfillLightweightSourcesJob", runId, station.getStationKey(),
          "mos", metrics,
          () -> mosFetcher.ingestRecentRuntimes("backfillLightweightSourcesJob", runId, station,
              List.of("GFS", "NAM"), properties.getJobs().getSmokeMosLookbackDays(), metrics));
      LocalDate targetDate = LocalDate.parse(properties.getJobs().getSmokeTargetDateLocal());
      jobRunService.logStructuredEvent("backfillLightweightSourcesJob", runId, station.getStationKey(),
          "ndfd_live_point", "phase_start", "RUNNING",
          Map.of("phase", "ndfd_live",
              "target_date_local", targetDate.toString()));
      hadFailures |= !runPhase("backfillLightweightSourcesJob", runId, station.getStationKey(),
          "ndfd_live", metrics,
          () -> ndfdLivePointFetcher.ingestLiveWindow("backfillLightweightSourcesJob", runId, station,
              targetDate.atStartOfDay(ZoneOffset.UTC).toString(),
              targetDate.plusDays(1).atStartOfDay(ZoneOffset.UTC).toString(),
              metrics));
      runPhase("backfillLightweightSourcesJob", runId, station.getStationKey(),
          "ndfd_historical", metrics,
          () -> ndfdHistoricalCatalogFetcher.markNotImplemented("backfillLightweightSourcesJob", runId,
              station.getStationKey(), properties.getJobs().getSmokeNdfdHistoricalDateLocal()));
      jobRunService.logStructuredEvent("backfillLightweightSourcesJob", runId, station.getStationKey(),
          "threadex_acis", "phase_start", "RUNNING",
          Map.of("phase", "threadex",
              "start_date_local", properties.getJobs().getSmokeThreadexStartDateLocal(),
              "end_date_local", targetDate.toString()));
      hadFailures |= !runPhase("backfillLightweightSourcesJob", runId, station.getStationKey(),
          "threadex", metrics,
          () -> threadexAcisFetcher.ingestRange("backfillLightweightSourcesJob", runId, station,
              properties.getJobs().getSmokeThreadexStartDateLocal(), targetDate.toString(), metrics));
      hadFailures |= !runPhase("backfillLightweightSourcesJob", runId, station.getStationKey(),
          "climate_normals", metrics,
          () -> climateNormalsFetcher.ingestStationNormals("backfillLightweightSourcesJob", runId, station, metrics));
      jobRunService.logStructuredEvent("backfillLightweightSourcesJob", runId, station.getStationKey(),
          "iem_climodat", "phase_start", "RUNNING",
          Map.of("phase", "climodat",
              "start_date_local", properties.getJobs().getSmokeClimodatStartDateLocal(),
              "end_date_local", targetDate.toString()));
      hadFailures |= !runPhase("backfillLightweightSourcesJob", runId, station.getStationKey(),
          "climodat", metrics,
          () -> iemClimodatFetcher.ingestRange("backfillLightweightSourcesJob", runId, station,
              properties.getJobs().getSmokeClimodatStartDateLocal(), targetDate.toString(), metrics));
      hadFailures |= !runPhase("backfillLightweightSourcesJob", runId, station.getStationKey(),
          "lcdv2_aux", metrics,
          () -> lcdv2Fetcher.ingestRange("backfillLightweightSourcesJob", runId, station,
              hourlyStart.toString(), stationToday.toString(), metrics));
      jobRunService.completeRun(runId, "backfillLightweightSourcesJob", station.getStationKey(),
          hadFailures ? "COMPLETE_WITH_ERRORS" : "COMPLETE", metrics);
      return runId;
    } catch (Exception ex) {
      jobRunService.logStructuredEvent("backfillLightweightSourcesJob", runId, station.getStationKey(),
          "backfillLightweightSourcesJob", "job_failed", "FAILED", Map.of("message", ex.getMessage()));
      jobRunService.completeRun(runId, "backfillLightweightSourcesJob", station.getStationKey(), "FAILED", metrics);
      throw new IllegalStateException("Failed backfillLightweightSourcesJob", ex);
    }
  }

  private boolean runPhase(String jobId,
                           String runId,
                           String stationKey,
                           String phaseName,
                           JobMetricsAccumulator metrics,
                           PhaseOperation operation) {
    try {
      operation.run();
      return true;
    } catch (Exception ex) {
      metrics.recordParserFailure(phaseName);
      jobRunService.logStructuredEvent(jobId, runId, stationKey,
          phaseName, "phase_failed", "FAILED", Map.of("message", ex.getMessage()));
      return false;
    }
  }

  @FunctionalInterface
  private interface PhaseOperation {
    void run() throws Exception;
  }
}
