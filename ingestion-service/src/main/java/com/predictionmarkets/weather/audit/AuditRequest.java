package com.predictionmarkets.weather.audit;

import com.predictionmarkets.weather.models.MosModel;
import java.time.LocalDate;
import java.util.List;

public record AuditRequest(
    List<String> stationIds,
    LocalDate startDateLocal,
    LocalDate endDateLocal,
    Long asofPolicyId,
    List<MosModel> models,
    int maxForecastDays,
    int sampleLimit) {
}
