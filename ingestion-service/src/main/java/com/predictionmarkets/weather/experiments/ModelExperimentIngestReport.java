package com.predictionmarkets.weather.experiments;

import java.util.List;

public record ModelExperimentIngestReport(
    String repoRoot,
    List<String> scanRoots,
    int candidateFiles,
    int ingested,
    int skipped,
    int errors
) {
}
