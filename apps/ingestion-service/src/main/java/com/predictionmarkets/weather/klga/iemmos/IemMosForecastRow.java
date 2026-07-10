package com.predictionmarkets.weather.klga.iemmos;

import java.time.Instant;
import java.util.UUID;

public record IemMosForecastRow(
    String stationId,
    String mosStationId,
    String sourceProduct,
    String endpointModel,
    String cutoffId,
    Instant runTimeUtc,
    Instant forecastValidTimeUtc,
    Double forecastHour,
    String periodType,
    Double nxF,
    Double tmpF,
    Double dptF,
    Double wdr,
    Double wspKt,
    Double gstKt,
    Double skyOrCloud,
    Double pop,
    Double qpf,
    Double tstmProb,
    String rawValuesJson,
    String rawPayloadHash,
    Instant providerAvailableAtUtc,
    Instant effectiveAvailableAtUtc,
    String availabilityMethod,
    String sourceRequestId,
    UUID sourceRecordId,
    String requestSha256,
    String rawRowHash,
    String parserVersion,
    String qualityFlag,
    String qualityNote) {
}
