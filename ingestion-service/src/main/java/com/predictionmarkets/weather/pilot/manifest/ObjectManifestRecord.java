package com.predictionmarkets.weather.pilot.manifest;

public record ObjectManifestRecord(
    String objectId,
    String runId,
    String stationKey,
    String sourceName,
    String sourceFamily,
    String sourceIdentifier,
    String requestUrlOrBucketKey,
    String requestedRangeStartUtc,
    String requestedRangeEndUtc,
    String cycleTimeUtc,
    Integer forecastHour,
    String domainName,
    Integer httpStatus,
    Integer contentLength,
    String checksumSha256,
    String payloadEncoding,
    String payloadText,
    byte[] payloadBlob,
    String parserStatus,
    Integer rowCount,
    String duplicateOfChecksum,
    String extractionStatus,
    Integer gribMessageCount,
    String ingestedAtUtc) {
}
