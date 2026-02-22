package com.predictionmarkets.weather.experiments;

import java.time.Instant;

public record ModelExperimentRequest(
    String experimentKey,
    String experimentName,
    String stationId,
    String modelName,
    String modelFamily,
    String sourcePath,
    String metadataJson,
    String metricsTrainJson,
    String metricsValidationJson,
    String metricsTestJson,
    Double trainMae,
    Double trainRmse,
    Double trainBias,
    Double trainMedianAe,
    Double trainMaxAe,
    Double trainCorr,
    Integer trainN,
    Double validationMae,
    Double validationRmse,
    Double validationBias,
    Double validationMedianAe,
    Double validationMaxAe,
    Double validationCorr,
    Integer validationN,
    Double testMae,
    Double testRmse,
    Double testBias,
    Double testMedianAe,
    Double testMaxAe,
    Double testCorr,
    Integer testN,
    String descriptionText,
    String rawPayloadHash,
    Instant retrievedAtUtc
) {
}
