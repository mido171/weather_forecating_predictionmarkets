package com.predictionmarkets.weather.pilot.manifest;

public record HttpRequestLogRecord(
    String runId,
    String jobId,
    String sourceName,
    String sourceFamily,
    String stationKey,
    String action,
    String requestUrlOrKey,
    String httpMethod,
    Integer statusCode,
    String issueTimeUtc,
    String validTimeUtc,
    Double durationMs,
    Integer bytesDownloaded,
    Integer rowsParsed,
    Integer retryCount,
    String status,
    String exceptionClass,
    String exceptionMessage,
    String createdAtUtc) {
}
