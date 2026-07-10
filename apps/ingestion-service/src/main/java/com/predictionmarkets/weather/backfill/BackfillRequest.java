package com.predictionmarkets.weather.backfill;

import com.predictionmarkets.weather.models.MosModel;
import java.time.LocalDate;
import java.util.List;

public record BackfillRequest(BackfillJobType jobType,
                              List<String> seriesTickers,
                              LocalDate dateStartLocal,
                              LocalDate dateEndLocal,
                              Long asofPolicyId,
                              List<MosModel> models,
                              int mosWindowDays) {
}
