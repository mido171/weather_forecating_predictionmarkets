package com.predictionmarkets.weather.gribstream;

import java.util.ArrayList;
import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "gribstream.variable-ingest", ignoreInvalidFields = true)
public class GribstreamVariableIngestProperties {
  private boolean enabled = false;
  private String catalogResource = "classpath:gribstream_data";
  private String whitelistResource = "classpath:gribstream_variable_whitelist.csv";
  private boolean whitelistRequired = true;
  private boolean storeRaw = false;
  private boolean storeSummary = true;
  private boolean collapseEnsembleMembers = true;
  private double minCoverageRatio = 0.5;
  private int batchSize = 20;
  private int maxVariablesPerModel = 0;
  private List<String> models = new ArrayList<>();

  public boolean isEnabled() {
    return enabled;
  }

  public void setEnabled(boolean enabled) {
    this.enabled = enabled;
  }

  public String getCatalogResource() {
    return catalogResource;
  }

  public void setCatalogResource(String catalogResource) {
    this.catalogResource = catalogResource;
  }

  public String getWhitelistResource() {
    return whitelistResource;
  }

  public void setWhitelistResource(String whitelistResource) {
    this.whitelistResource = whitelistResource;
  }

  public boolean isWhitelistRequired() {
    return whitelistRequired;
  }

  public void setWhitelistRequired(boolean whitelistRequired) {
    this.whitelistRequired = whitelistRequired;
  }

  public boolean isStoreRaw() {
    return storeRaw;
  }

  public void setStoreRaw(boolean storeRaw) {
    this.storeRaw = storeRaw;
  }

  public boolean isStoreSummary() {
    return storeSummary;
  }

  public void setStoreSummary(boolean storeSummary) {
    this.storeSummary = storeSummary;
  }

  public boolean isCollapseEnsembleMembers() {
    return collapseEnsembleMembers;
  }

  public void setCollapseEnsembleMembers(boolean collapseEnsembleMembers) {
    this.collapseEnsembleMembers = collapseEnsembleMembers;
  }

  public double getMinCoverageRatio() {
    return minCoverageRatio;
  }

  public void setMinCoverageRatio(double minCoverageRatio) {
    this.minCoverageRatio = minCoverageRatio;
  }

  public int getBatchSize() {
    return batchSize;
  }

  public void setBatchSize(int batchSize) {
    this.batchSize = batchSize;
  }

  public int getMaxVariablesPerModel() {
    return maxVariablesPerModel;
  }

  public void setMaxVariablesPerModel(int maxVariablesPerModel) {
    this.maxVariablesPerModel = maxVariablesPerModel;
  }

  public List<String> getModels() {
    return models;
  }

  public void setModels(List<String> models) {
    this.models = models;
  }
}
