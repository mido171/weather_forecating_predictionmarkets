package com.predictionmarkets.weather.pilot.catalog;

public record SourceInventoryRecord(
    String inventoryKey,
    String stationKey,
    String sourceName,
    String sourceFamily,
    String itemType,
    String itemKey,
    String issueTimeUtc,
    String validTimeUtc,
    String status,
    String detailsJson,
    String createdAtUtc,
    String updatedAtUtc) {
}
