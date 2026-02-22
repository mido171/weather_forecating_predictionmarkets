package com.predictionmarkets.weather.experiments;

import com.predictionmarkets.weather.common.Hashing;
import com.predictionmarkets.weather.models.ModelExperiment;
import com.predictionmarkets.weather.repository.ModelExperimentRepository;
import java.time.Instant;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Optional;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class ModelExperimentService {
  private final ModelExperimentRepository repository;

  public ModelExperimentService(ModelExperimentRepository repository) {
    this.repository = repository;
  }

  @Transactional
  public ModelExperimentResponse createOrUpdateByKey(ModelExperimentRequest request) {
    Objects.requireNonNull(request, "request is required");
    if (request.experimentKey() == null || request.experimentKey().isBlank()) {
      throw new IllegalArgumentException("experimentKey is required");
    }
    ModelExperiment entity = repository.findByExperimentKey(request.experimentKey())
        .orElseGet(ModelExperiment::new);
    boolean isNew = entity.getId() == null;
    applyRequest(entity, request);
    validateRequired(entity, isNew);

    Instant now = Instant.now();
    if (entity.getRetrievedAtUtc() == null) {
      entity.setRetrievedAtUtc(now);
    }
    if (entity.getMetadataJson() != null) {
      entity.setRawPayloadHash(Hashing.sha256Hex(entity.getMetadataJson()));
    }
    if (isNew) {
      entity.setCreatedAtUtc(now);
    }
    entity.setUpdatedAtUtc(now);
    ModelExperiment saved = repository.save(entity);
    return toResponse(saved);
  }

  @Transactional
  public ModelExperimentResponse update(Long id, ModelExperimentRequest request) {
    Objects.requireNonNull(id, "id is required");
    Objects.requireNonNull(request, "request is required");
    ModelExperiment entity = repository.findById(id)
        .orElseThrow(() -> new IllegalArgumentException("ModelExperiment not found: " + id));
    applyRequest(entity, request);
    validateRequired(entity, false);

    Instant now = Instant.now();
    if (entity.getRetrievedAtUtc() == null) {
      entity.setRetrievedAtUtc(now);
    }
    if (entity.getMetadataJson() != null) {
      entity.setRawPayloadHash(Hashing.sha256Hex(entity.getMetadataJson()));
    }
    entity.setUpdatedAtUtc(now);
    ModelExperiment saved = repository.save(entity);
    return toResponse(saved);
  }

  @Transactional(readOnly = true)
  public ModelExperimentResponse get(Long id) {
    return repository.findById(id)
        .map(ModelExperimentService::toResponse)
        .orElseThrow(() -> new IllegalArgumentException("ModelExperiment not found: " + id));
  }

  @Transactional(readOnly = true)
  public Optional<ModelExperimentResponse> getByKey(String experimentKey) {
    if (experimentKey == null || experimentKey.isBlank()) {
      return Optional.empty();
    }
    return repository.findByExperimentKey(experimentKey)
        .map(ModelExperimentService::toResponse);
  }

  @Transactional(readOnly = true)
  public List<ModelExperimentResponse> list(String stationId, String modelFamily, int limit) {
    List<ModelExperiment> experiments;
    if (stationId != null && !stationId.isBlank()) {
      experiments = repository.findAllByStationIdIgnoreCase(stationId.trim());
    } else {
      experiments = repository.findAll();
    }
    List<ModelExperiment> filtered = experiments.stream()
        .filter(exp -> modelFamily == null || modelFamily.isBlank()
            || (exp.getModelFamily() != null
                && exp.getModelFamily().equalsIgnoreCase(modelFamily.trim())))
        .sorted(Comparator
            .comparing(ModelExperiment::getTestMae, Comparator.nullsLast(Double::compareTo))
            .thenComparing(exp -> normalize(exp.getExperimentKey())))
        .toList();
    if (limit > 0 && filtered.size() > limit) {
      filtered = filtered.subList(0, limit);
    }
    return filtered.stream().map(ModelExperimentService::toResponse).toList();
  }

  @Transactional
  public void delete(Long id) {
    if (!repository.existsById(id)) {
      throw new IllegalArgumentException("ModelExperiment not found: " + id);
    }
    repository.deleteById(id);
  }

  @Transactional
  public ModelExperiment upsert(ModelExperiment candidate) {
    Objects.requireNonNull(candidate, "candidate is required");
    if (candidate.getExperimentKey() == null || candidate.getExperimentKey().isBlank()) {
      throw new IllegalArgumentException("candidate.experimentKey is required");
    }
    ModelExperiment entity = repository.findByExperimentKey(candidate.getExperimentKey())
        .orElseGet(ModelExperiment::new);
    boolean isNew = entity.getId() == null;
    copyEntity(candidate, entity);
    validateRequired(entity, true);

    Instant now = Instant.now();
    if (entity.getRetrievedAtUtc() == null) {
      entity.setRetrievedAtUtc(now);
    }
    if (entity.getMetadataJson() != null) {
      entity.setRawPayloadHash(Hashing.sha256Hex(entity.getMetadataJson()));
    }
    if (isNew) {
      entity.setCreatedAtUtc(now);
    }
    entity.setUpdatedAtUtc(now);
    return repository.save(entity);
  }

  private void applyRequest(ModelExperiment entity, ModelExperimentRequest request) {
    if (request.experimentKey() != null) {
      entity.setExperimentKey(request.experimentKey());
    }
    if (request.experimentName() != null) {
      entity.setExperimentName(request.experimentName());
    }
    if (request.stationId() != null) {
      entity.setStationId(request.stationId());
    }
    if (request.modelName() != null) {
      entity.setModelName(request.modelName());
    }
    if (request.modelFamily() != null) {
      entity.setModelFamily(request.modelFamily());
    }
    if (request.sourcePath() != null) {
      entity.setSourcePath(request.sourcePath());
    }
    if (request.metadataJson() != null) {
      entity.setMetadataJson(request.metadataJson());
    }
    if (request.metricsTrainJson() != null) {
      entity.setMetricsTrainJson(request.metricsTrainJson());
    }
    if (request.metricsValidationJson() != null) {
      entity.setMetricsValidationJson(request.metricsValidationJson());
    }
    if (request.metricsTestJson() != null) {
      entity.setMetricsTestJson(request.metricsTestJson());
    }
    if (request.trainMae() != null) {
      entity.setTrainMae(request.trainMae());
    }
    if (request.trainRmse() != null) {
      entity.setTrainRmse(request.trainRmse());
    }
    if (request.trainBias() != null) {
      entity.setTrainBias(request.trainBias());
    }
    if (request.trainMedianAe() != null) {
      entity.setTrainMedianAe(request.trainMedianAe());
    }
    if (request.trainMaxAe() != null) {
      entity.setTrainMaxAe(request.trainMaxAe());
    }
    if (request.trainCorr() != null) {
      entity.setTrainCorr(request.trainCorr());
    }
    if (request.trainN() != null) {
      entity.setTrainN(request.trainN());
    }
    if (request.validationMae() != null) {
      entity.setValidationMae(request.validationMae());
    }
    if (request.validationRmse() != null) {
      entity.setValidationRmse(request.validationRmse());
    }
    if (request.validationBias() != null) {
      entity.setValidationBias(request.validationBias());
    }
    if (request.validationMedianAe() != null) {
      entity.setValidationMedianAe(request.validationMedianAe());
    }
    if (request.validationMaxAe() != null) {
      entity.setValidationMaxAe(request.validationMaxAe());
    }
    if (request.validationCorr() != null) {
      entity.setValidationCorr(request.validationCorr());
    }
    if (request.validationN() != null) {
      entity.setValidationN(request.validationN());
    }
    if (request.testMae() != null) {
      entity.setTestMae(request.testMae());
    }
    if (request.testRmse() != null) {
      entity.setTestRmse(request.testRmse());
    }
    if (request.testBias() != null) {
      entity.setTestBias(request.testBias());
    }
    if (request.testMedianAe() != null) {
      entity.setTestMedianAe(request.testMedianAe());
    }
    if (request.testMaxAe() != null) {
      entity.setTestMaxAe(request.testMaxAe());
    }
    if (request.testCorr() != null) {
      entity.setTestCorr(request.testCorr());
    }
    if (request.testN() != null) {
      entity.setTestN(request.testN());
    }
    if (request.descriptionText() != null) {
      entity.setDescriptionText(request.descriptionText());
    }
    if (request.rawPayloadHash() != null) {
      entity.setRawPayloadHash(request.rawPayloadHash());
    }
    if (request.retrievedAtUtc() != null) {
      entity.setRetrievedAtUtc(request.retrievedAtUtc());
    }
  }

  private void copyEntity(ModelExperiment source, ModelExperiment target) {
    target.setExperimentKey(source.getExperimentKey());
    target.setExperimentName(source.getExperimentName());
    target.setStationId(source.getStationId());
    target.setModelName(source.getModelName());
    target.setModelFamily(source.getModelFamily());
    target.setSourcePath(source.getSourcePath());
    target.setMetadataJson(source.getMetadataJson());
    target.setMetricsTrainJson(source.getMetricsTrainJson());
    target.setMetricsValidationJson(source.getMetricsValidationJson());
    target.setMetricsTestJson(source.getMetricsTestJson());
    target.setTrainMae(source.getTrainMae());
    target.setTrainRmse(source.getTrainRmse());
    target.setTrainBias(source.getTrainBias());
    target.setTrainMedianAe(source.getTrainMedianAe());
    target.setTrainMaxAe(source.getTrainMaxAe());
    target.setTrainCorr(source.getTrainCorr());
    target.setTrainN(source.getTrainN());
    target.setValidationMae(source.getValidationMae());
    target.setValidationRmse(source.getValidationRmse());
    target.setValidationBias(source.getValidationBias());
    target.setValidationMedianAe(source.getValidationMedianAe());
    target.setValidationMaxAe(source.getValidationMaxAe());
    target.setValidationCorr(source.getValidationCorr());
    target.setValidationN(source.getValidationN());
    target.setTestMae(source.getTestMae());
    target.setTestRmse(source.getTestRmse());
    target.setTestBias(source.getTestBias());
    target.setTestMedianAe(source.getTestMedianAe());
    target.setTestMaxAe(source.getTestMaxAe());
    target.setTestCorr(source.getTestCorr());
    target.setTestN(source.getTestN());
    target.setDescriptionText(source.getDescriptionText());
    target.setRawPayloadHash(source.getRawPayloadHash());
    target.setRetrievedAtUtc(source.getRetrievedAtUtc());
  }

  private void validateRequired(ModelExperiment entity, boolean requireAll) {
    if (entity.getExperimentKey() == null || entity.getExperimentKey().isBlank()) {
      throw new IllegalArgumentException("experimentKey is required");
    }
    if (requireAll || entity.getSourcePath() == null || entity.getSourcePath().isBlank()) {
      if (entity.getSourcePath() == null || entity.getSourcePath().isBlank()) {
        throw new IllegalArgumentException("sourcePath is required");
      }
    }
    if (requireAll || entity.getMetadataJson() == null || entity.getMetadataJson().isBlank()) {
      if (entity.getMetadataJson() == null || entity.getMetadataJson().isBlank()) {
        throw new IllegalArgumentException("metadataJson is required");
      }
    }
    if (requireAll || entity.getDescriptionText() == null || entity.getDescriptionText().isBlank()) {
      if (entity.getDescriptionText() == null || entity.getDescriptionText().isBlank()) {
        throw new IllegalArgumentException("descriptionText is required");
      }
    }
  }

  private static ModelExperimentResponse toResponse(ModelExperiment entity) {
    return new ModelExperimentResponse(
        entity.getId(),
        entity.getExperimentKey(),
        entity.getExperimentName(),
        entity.getStationId(),
        entity.getModelName(),
        entity.getModelFamily(),
        entity.getSourcePath(),
        entity.getMetadataJson(),
        entity.getMetricsTrainJson(),
        entity.getMetricsValidationJson(),
        entity.getMetricsTestJson(),
        entity.getTrainMae(),
        entity.getTrainRmse(),
        entity.getTrainBias(),
        entity.getTrainMedianAe(),
        entity.getTrainMaxAe(),
        entity.getTrainCorr(),
        entity.getTrainN(),
        entity.getValidationMae(),
        entity.getValidationRmse(),
        entity.getValidationBias(),
        entity.getValidationMedianAe(),
        entity.getValidationMaxAe(),
        entity.getValidationCorr(),
        entity.getValidationN(),
        entity.getTestMae(),
        entity.getTestRmse(),
        entity.getTestBias(),
        entity.getTestMedianAe(),
        entity.getTestMaxAe(),
        entity.getTestCorr(),
        entity.getTestN(),
        entity.getDescriptionText(),
        entity.getRawPayloadHash(),
        entity.getRetrievedAtUtc(),
        entity.getCreatedAtUtc(),
        entity.getUpdatedAtUtc());
  }

  private static String normalize(String value) {
    return value == null ? "" : value.toLowerCase(Locale.ROOT);
  }
}
