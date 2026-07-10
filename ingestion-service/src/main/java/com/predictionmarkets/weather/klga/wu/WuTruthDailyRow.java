package com.predictionmarkets.weather.klga.wu;

import com.fasterxml.jackson.databind.JsonNode;
import java.time.Instant;
import java.time.LocalDate;

record WuTruthDailyRow(
    String stationId,
    String wundergroundStationId,
    LocalDate localDate,
    String timezoneName,
    Integer tmaxF,
    Integer tminF,
    int observationCount,
    JsonNode highObservationTimesLocalJson,
    JsonNode hourlyObservationsJson,
    JsonNode providerMaxTempValuesJson,
    JsonNode providerMinTempValuesJson,
    String sourceUrlRedacted,
    String wuPageUrl,
    String payloadHash,
    String parserVersion,
    Instant fetchedAtUtc,
    Instant settlementAvailableAtUtc,
    String dailyHighSource,
    String validationStatus,
    JsonNode validationNotesJson,
    String fetchStatus,
    Integer httpStatus,
    String errorType,
    String errorMessage,
    int attempts
) {
}
