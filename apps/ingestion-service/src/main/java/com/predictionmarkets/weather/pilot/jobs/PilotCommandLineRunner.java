package com.predictionmarkets.weather.pilot.jobs;

import com.predictionmarkets.weather.pilot.config.PilotIngestionProperties;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

@Component
@ConditionalOnProperty(prefix = "pilot.knyc", name = "enabled", havingValue = "true")
public class PilotCommandLineRunner implements CommandLineRunner {
  private final PilotIngestionProperties properties;
  private final BootstrapKnycStationJob bootstrapJob;
  private final BackfillLightweightSourcesJob lightweightJob;
  private final BackfillHeavyModelSourcesJob heavyJob;
  private final BuildDecisionSnapshotsJob snapshotJob;

  public PilotCommandLineRunner(PilotIngestionProperties properties,
                                BootstrapKnycStationJob bootstrapJob,
                                BackfillLightweightSourcesJob lightweightJob,
                                BackfillHeavyModelSourcesJob heavyJob,
                                BuildDecisionSnapshotsJob snapshotJob) {
    this.properties = properties;
    this.bootstrapJob = bootstrapJob;
    this.lightweightJob = lightweightJob;
    this.heavyJob = heavyJob;
    this.snapshotJob = snapshotJob;
  }

  @Override
  public void run(String... args) {
    if (!properties.isEnabled()) {
      return;
    }
    if (properties.getJobs().isBootstrapEnabled()) {
      bootstrapJob.run();
    }
    if (properties.getJobs().isLightweightEnabled()) {
      lightweightJob.run();
    }
    if (properties.getJobs().isHeavyEnabled()) {
      heavyJob.run();
    }
    if (properties.getJobs().isSnapshotEnabled()) {
      snapshotJob.run();
    }
  }
}
