package com.predictionmarkets.weather.klga.iemmos;

import java.util.UUID;

public record IemMosStoredRequest(
    String sourceRequestId,
    UUID sourceRecordId,
    String rawStorageUri,
    String responseBodySha256,
    long responseSizeBytes) {
}
