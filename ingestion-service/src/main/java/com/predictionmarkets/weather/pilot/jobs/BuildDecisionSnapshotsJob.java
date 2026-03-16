package com.predictionmarkets.weather.pilot.jobs;

import com.predictionmarkets.weather.pilot.catalog.JobRunService;
import com.predictionmarkets.weather.pilot.catalog.SqliteCatalogService;
import com.predictionmarkets.weather.pilot.config.PilotConfigLoader;
import com.predictionmarkets.weather.pilot.config.PilotIngestionProperties;
import com.predictionmarkets.weather.pilot.config.StationConfig;
import com.predictionmarkets.weather.pilot.metrics.GapAuditService;
import com.predictionmarkets.weather.pilot.metrics.MetricsService;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.stereotype.Service;

@Service
public class BuildDecisionSnapshotsJob {
  private final PilotConfigLoader configLoader;
  private final PilotIngestionProperties properties;
  private final SqliteCatalogService catalogService;
  private final JobRunService jobRunService;
  private final MetricsService metricsService;
  private final GapAuditService gapAuditService;

  public BuildDecisionSnapshotsJob(PilotConfigLoader configLoader,
                                   PilotIngestionProperties properties,
                                   SqliteCatalogService catalogService,
                                   JobRunService jobRunService,
                                   MetricsService metricsService,
                                   GapAuditService gapAuditService) {
    this.configLoader = configLoader;
    this.properties = properties;
    this.catalogService = catalogService;
    this.jobRunService = jobRunService;
    this.metricsService = metricsService;
    this.gapAuditService = gapAuditService;
  }

  public String run() {
    StationConfig station = configLoader.requireDefaultStation();
    String runId = jobRunService.startRun("buildDecisionSnapshotsJob", station.getStationKey());
    var metrics = metricsService.newAccumulator();
    try {
      LocalDate targetDate = LocalDate.parse(properties.getJobs().getSmokeTargetDateLocal());
      List<Instant> decisionTimes = List.of(
          targetDate.minusDays(2).atStartOfDay(ZoneOffset.UTC).toInstant(),
          targetDate.minusDays(1).atStartOfDay(ZoneOffset.UTC).toInstant(),
          targetDate.minusDays(1).atTime(12, 0).toInstant(ZoneOffset.UTC),
          targetDate.atStartOfDay(ZoneOffset.UTC).toInstant().minusSeconds(6 * 3600L),
          targetDate.atStartOfDay(ZoneOffset.UTC).toInstant().minusSeconds(3 * 3600L),
          targetDate.atStartOfDay(ZoneOffset.UTC).toInstant().minusSeconds(3600L));
      for (Instant decisionTime : decisionTimes) {
        buildSnapshot(runId, station, targetDate, decisionTime);
      }
      metrics.recordRequest(0.0d, 0, decisionTimes.size(), "SUCCESS");
      jobRunService.completeRun(runId, "buildDecisionSnapshotsJob", station.getStationKey(), "COMPLETE", metrics);
      return runId;
    } catch (Exception ex) {
      jobRunService.completeRun(runId, "buildDecisionSnapshotsJob", station.getStationKey(), "FAILED", metrics);
      throw new IllegalStateException("Failed buildDecisionSnapshotsJob", ex);
    }
  }

  private void buildSnapshot(String runId,
                             StationConfig station,
                             LocalDate targetDate,
                             Instant decisionTime) {
    String jobId = "buildDecisionSnapshotsJob";
    Map<String, Object> snapshot = new LinkedHashMap<>();
    Map<String, Object> latestObs = catalogService.querySingle("""
        SELECT * FROM asos_hourly_obs
        WHERE station_key = ? AND valid_time_utc <= ?
        ORDER BY valid_time_utc DESC
        LIMIT 1
        """, station.getStationKey(), decisionTime.toString());
    if (latestObs.isEmpty()) {
      gapAuditService.recordGap(runId, station.getStationKey(), "asos_hourly_obs", null, decisionTime.toString(),
          1, 0, "MISSING", Map.of("reason", "No hourly obs before decision time"));
    }
    snapshot.put("latestObservation", latestObs);

    List<Map<String, Object>> mosRows = catalogService.query("""
        SELECT * FROM mos_station_guidance
        WHERE station_key = ? AND mos_target_day_local = ? AND runtime_utc <= ?
        ORDER BY model_name ASC, runtime_utc DESC, valid_time_utc ASC
        """, station.getStationKey(), targetDate.toString(), decisionTime.toString());
    Map<String, Map<String, Object>> latestMosByModel = new LinkedHashMap<>();
    for (Map<String, Object> row : mosRows) {
      String model = String.valueOf(row.get("model_name"));
      latestMosByModel.putIfAbsent(model, row);
    }
    snapshot.put("latestMosByModel", latestMosByModel);

    List<Map<String, Object>> ndfdRows = catalogService.query("""
        SELECT * FROM ndfd_point_forecast
        WHERE station_key = ? AND issue_time_utc <= ? AND valid_time_utc >= ?
        ORDER BY issue_time_utc DESC, valid_time_utc ASC
        LIMIT 48
        """, station.getStationKey(), decisionTime.toString(), targetDate.atStartOfDay(ZoneOffset.UTC).toInstant().toString());
    snapshot.put("latestNdfdPath", ndfdRows);

    List<Map<String, Object>> climodatPrior = catalogService.query("""
        SELECT * FROM climodat_daily_aux
        WHERE station_key = ? AND date_localish < ?
        ORDER BY date_localish DESC
        LIMIT 3
        """, station.getStationKey(), targetDate.toString());
    snapshot.put("climodatPrior", climodatPrior);

    Map<String, Object> missingSources = new LinkedHashMap<>();
    missingSources.put("climate_normals", true);
    missingSources.put("lcdv2_aux", true);
    missingSources.put("ndfd_historical", true);
    catalogService.execute("""
        INSERT INTO decision_snapshot_feature_set (
          station_key, target_day_local, decision_time_utc, snapshot_version,
          snapshot_json, missing_sources_json, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(station_key, target_day_local, decision_time_utc, snapshot_version) DO UPDATE SET
          snapshot_json=excluded.snapshot_json,
          missing_sources_json=excluded.missing_sources_json,
          created_at_utc=excluded.created_at_utc
        """,
        station.getStationKey(),
        targetDate.toString(),
        decisionTime.toString(),
        properties.getParserVersion(),
        catalogService.toJson(snapshot),
        catalogService.toJson(missingSources),
        Instant.now().toString());
    jobRunService.logStructuredEvent(jobId, runId, station.getStationKey(),
        "decision_snapshot_feature_set", "snapshot_built", "SUCCESS",
        Map.of("target_day_local", targetDate.toString(),
            "decision_time_utc", decisionTime.toString(),
            "has_latest_observation", !latestObs.isEmpty(),
            "mos_model_count", latestMosByModel.size(),
            "ndfd_row_count", ndfdRows.size(),
            "climodat_prior_count", climodatPrior.size()));
  }
}
