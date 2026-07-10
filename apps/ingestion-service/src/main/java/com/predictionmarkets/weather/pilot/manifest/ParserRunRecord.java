package com.predictionmarkets.weather.pilot.manifest;

public record ParserRunRecord(
    String parserRunId,
    String runId,
    String sourceName,
    String parserVersion,
    String objectId,
    String status,
    Integer rowsParsed,
    Double durationMs,
    String detailsJson,
    String createdAtUtc) {
}
