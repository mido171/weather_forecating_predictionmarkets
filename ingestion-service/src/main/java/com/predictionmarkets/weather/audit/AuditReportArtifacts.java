package com.predictionmarkets.weather.audit;

import java.nio.file.Path;

public record AuditReportArtifacts(
    AuditReport report,
    Path markdownPath,
    Path jsonPath) {
}
