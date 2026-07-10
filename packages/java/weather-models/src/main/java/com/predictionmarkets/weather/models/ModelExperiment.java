package com.predictionmarkets.weather.models;

import java.time.Instant;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Lob;
import jakarta.persistence.Table;
import jakarta.persistence.UniqueConstraint;

@Entity
@Table(
    name = "model_experiment",
    uniqueConstraints = {
        @UniqueConstraint(columnNames = {"experiment_key"})
    }
)
public class ModelExperiment {
  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  @Column(name = "id", nullable = false)
  private Long id;

  @Column(name = "experiment_key", nullable = false, length = 255)
  private String experimentKey;

  @Column(name = "experiment_name", length = 255)
  private String experimentName;

  @Column(name = "station_id", length = 32)
  private String stationId;

  @Column(name = "model_name", length = 128)
  private String modelName;

  @Column(name = "model_family", length = 64)
  private String modelFamily;

  @Column(name = "source_path", nullable = false, length = 1024)
  private String sourcePath;

  @Lob
  @Column(name = "metadata_json", nullable = false)
  private String metadataJson;

  @Lob
  @Column(name = "metrics_train_json")
  private String metricsTrainJson;

  @Lob
  @Column(name = "metrics_validation_json")
  private String metricsValidationJson;

  @Lob
  @Column(name = "metrics_test_json")
  private String metricsTestJson;

  @Column(name = "train_mae")
  private Double trainMae;

  @Column(name = "train_rmse")
  private Double trainRmse;

  @Column(name = "train_bias")
  private Double trainBias;

  @Column(name = "train_median_ae")
  private Double trainMedianAe;

  @Column(name = "train_max_ae")
  private Double trainMaxAe;

  @Column(name = "train_corr")
  private Double trainCorr;

  @Column(name = "train_n")
  private Integer trainN;

  @Column(name = "validation_mae")
  private Double validationMae;

  @Column(name = "validation_rmse")
  private Double validationRmse;

  @Column(name = "validation_bias")
  private Double validationBias;

  @Column(name = "validation_median_ae")
  private Double validationMedianAe;

  @Column(name = "validation_max_ae")
  private Double validationMaxAe;

  @Column(name = "validation_corr")
  private Double validationCorr;

  @Column(name = "validation_n")
  private Integer validationN;

  @Column(name = "test_mae")
  private Double testMae;

  @Column(name = "test_rmse")
  private Double testRmse;

  @Column(name = "test_bias")
  private Double testBias;

  @Column(name = "test_median_ae")
  private Double testMedianAe;

  @Column(name = "test_max_ae")
  private Double testMaxAe;

  @Column(name = "test_corr")
  private Double testCorr;

  @Column(name = "test_n")
  private Integer testN;

  @Lob
  @Column(name = "description_text", nullable = false)
  private String descriptionText;

  @Column(name = "raw_payload_hash", length = 64)
  private String rawPayloadHash;

  @Column(name = "retrieved_at_utc", nullable = false)
  private Instant retrievedAtUtc;

  @Column(name = "created_at_utc", nullable = false)
  private Instant createdAtUtc;

  @Column(name = "updated_at_utc", nullable = false)
  private Instant updatedAtUtc;

  public Long getId() {
    return id;
  }

  public void setId(Long id) {
    this.id = id;
  }

  public String getExperimentKey() {
    return experimentKey;
  }

  public void setExperimentKey(String experimentKey) {
    this.experimentKey = experimentKey;
  }

  public String getExperimentName() {
    return experimentName;
  }

  public void setExperimentName(String experimentName) {
    this.experimentName = experimentName;
  }

  public String getStationId() {
    return stationId;
  }

  public void setStationId(String stationId) {
    this.stationId = stationId;
  }

  public String getModelName() {
    return modelName;
  }

  public void setModelName(String modelName) {
    this.modelName = modelName;
  }

  public String getModelFamily() {
    return modelFamily;
  }

  public void setModelFamily(String modelFamily) {
    this.modelFamily = modelFamily;
  }

  public String getSourcePath() {
    return sourcePath;
  }

  public void setSourcePath(String sourcePath) {
    this.sourcePath = sourcePath;
  }

  public String getMetadataJson() {
    return metadataJson;
  }

  public void setMetadataJson(String metadataJson) {
    this.metadataJson = metadataJson;
  }

  public String getMetricsTrainJson() {
    return metricsTrainJson;
  }

  public void setMetricsTrainJson(String metricsTrainJson) {
    this.metricsTrainJson = metricsTrainJson;
  }

  public String getMetricsValidationJson() {
    return metricsValidationJson;
  }

  public void setMetricsValidationJson(String metricsValidationJson) {
    this.metricsValidationJson = metricsValidationJson;
  }

  public String getMetricsTestJson() {
    return metricsTestJson;
  }

  public void setMetricsTestJson(String metricsTestJson) {
    this.metricsTestJson = metricsTestJson;
  }

  public Double getTrainMae() {
    return trainMae;
  }

  public void setTrainMae(Double trainMae) {
    this.trainMae = trainMae;
  }

  public Double getTrainRmse() {
    return trainRmse;
  }

  public void setTrainRmse(Double trainRmse) {
    this.trainRmse = trainRmse;
  }

  public Double getTrainBias() {
    return trainBias;
  }

  public void setTrainBias(Double trainBias) {
    this.trainBias = trainBias;
  }

  public Double getTrainMedianAe() {
    return trainMedianAe;
  }

  public void setTrainMedianAe(Double trainMedianAe) {
    this.trainMedianAe = trainMedianAe;
  }

  public Double getTrainMaxAe() {
    return trainMaxAe;
  }

  public void setTrainMaxAe(Double trainMaxAe) {
    this.trainMaxAe = trainMaxAe;
  }

  public Double getTrainCorr() {
    return trainCorr;
  }

  public void setTrainCorr(Double trainCorr) {
    this.trainCorr = trainCorr;
  }

  public Integer getTrainN() {
    return trainN;
  }

  public void setTrainN(Integer trainN) {
    this.trainN = trainN;
  }

  public Double getValidationMae() {
    return validationMae;
  }

  public void setValidationMae(Double validationMae) {
    this.validationMae = validationMae;
  }

  public Double getValidationRmse() {
    return validationRmse;
  }

  public void setValidationRmse(Double validationRmse) {
    this.validationRmse = validationRmse;
  }

  public Double getValidationBias() {
    return validationBias;
  }

  public void setValidationBias(Double validationBias) {
    this.validationBias = validationBias;
  }

  public Double getValidationMedianAe() {
    return validationMedianAe;
  }

  public void setValidationMedianAe(Double validationMedianAe) {
    this.validationMedianAe = validationMedianAe;
  }

  public Double getValidationMaxAe() {
    return validationMaxAe;
  }

  public void setValidationMaxAe(Double validationMaxAe) {
    this.validationMaxAe = validationMaxAe;
  }

  public Double getValidationCorr() {
    return validationCorr;
  }

  public void setValidationCorr(Double validationCorr) {
    this.validationCorr = validationCorr;
  }

  public Integer getValidationN() {
    return validationN;
  }

  public void setValidationN(Integer validationN) {
    this.validationN = validationN;
  }

  public Double getTestMae() {
    return testMae;
  }

  public void setTestMae(Double testMae) {
    this.testMae = testMae;
  }

  public Double getTestRmse() {
    return testRmse;
  }

  public void setTestRmse(Double testRmse) {
    this.testRmse = testRmse;
  }

  public Double getTestBias() {
    return testBias;
  }

  public void setTestBias(Double testBias) {
    this.testBias = testBias;
  }

  public Double getTestMedianAe() {
    return testMedianAe;
  }

  public void setTestMedianAe(Double testMedianAe) {
    this.testMedianAe = testMedianAe;
  }

  public Double getTestMaxAe() {
    return testMaxAe;
  }

  public void setTestMaxAe(Double testMaxAe) {
    this.testMaxAe = testMaxAe;
  }

  public Double getTestCorr() {
    return testCorr;
  }

  public void setTestCorr(Double testCorr) {
    this.testCorr = testCorr;
  }

  public Integer getTestN() {
    return testN;
  }

  public void setTestN(Integer testN) {
    this.testN = testN;
  }

  public String getDescriptionText() {
    return descriptionText;
  }

  public void setDescriptionText(String descriptionText) {
    this.descriptionText = descriptionText;
  }

  public String getRawPayloadHash() {
    return rawPayloadHash;
  }

  public void setRawPayloadHash(String rawPayloadHash) {
    this.rawPayloadHash = rawPayloadHash;
  }

  public Instant getRetrievedAtUtc() {
    return retrievedAtUtc;
  }

  public void setRetrievedAtUtc(Instant retrievedAtUtc) {
    this.retrievedAtUtc = retrievedAtUtc;
  }

  public Instant getCreatedAtUtc() {
    return createdAtUtc;
  }

  public void setCreatedAtUtc(Instant createdAtUtc) {
    this.createdAtUtc = createdAtUtc;
  }

  public Instant getUpdatedAtUtc() {
    return updatedAtUtc;
  }

  public void setUpdatedAtUtc(Instant updatedAtUtc) {
    this.updatedAtUtc = updatedAtUtc;
  }
}
