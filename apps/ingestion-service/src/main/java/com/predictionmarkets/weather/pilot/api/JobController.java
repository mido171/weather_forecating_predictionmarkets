package com.predictionmarkets.weather.pilot.api;

import com.predictionmarkets.weather.pilot.jobs.BackfillHeavyModelSourcesJob;
import com.predictionmarkets.weather.pilot.jobs.BackfillLightweightSourcesJob;
import com.predictionmarkets.weather.pilot.jobs.BootstrapKnycStationJob;
import com.predictionmarkets.weather.pilot.jobs.BuildDecisionSnapshotsJob;
import com.predictionmarkets.weather.pilot.metrics.MetricsService;
import java.util.Map;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/internal/ingest/jobs")
@ConditionalOnProperty(prefix = "ingestion.admin-api", name = "enabled", havingValue = "true")
public class JobController {
  private final MetricsService metricsService;
  private final BootstrapKnycStationJob bootstrapJob;
  private final BackfillLightweightSourcesJob lightweightJob;
  private final BackfillHeavyModelSourcesJob heavyJob;
  private final BuildDecisionSnapshotsJob snapshotJob;

  public JobController(MetricsService metricsService,
                       BootstrapKnycStationJob bootstrapJob,
                       BackfillLightweightSourcesJob lightweightJob,
                       BackfillHeavyModelSourcesJob heavyJob,
                       BuildDecisionSnapshotsJob snapshotJob) {
    this.metricsService = metricsService;
    this.bootstrapJob = bootstrapJob;
    this.lightweightJob = lightweightJob;
    this.heavyJob = heavyJob;
    this.snapshotJob = snapshotJob;
  }

  @GetMapping
  public Object jobs() {
    return metricsService.recentJobRuns();
  }

  @GetMapping("/{runId}")
  public Map<String, Object> job(@PathVariable String runId) {
    return metricsService.jobRun(runId);
  }

  @PostMapping("/bootstrap")
  public Map<String, Object> triggerBootstrap() {
    return Map.of("runId", bootstrapJob.run());
  }

  @PostMapping("/lightweight")
  public Map<String, Object> triggerLightweight() {
    return Map.of("runId", lightweightJob.run());
  }

  @PostMapping("/heavy")
  public Map<String, Object> triggerHeavy() {
    return Map.of("runId", heavyJob.run());
  }

  @PostMapping("/snapshots")
  public Map<String, Object> triggerSnapshots() {
    return Map.of("runId", snapshotJob.run());
  }
}
