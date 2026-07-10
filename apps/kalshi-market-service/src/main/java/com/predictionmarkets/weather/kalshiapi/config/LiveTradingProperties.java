package com.predictionmarkets.weather.kalshiapi.config;

import jakarta.validation.Valid;
import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import java.util.ArrayList;
import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@ConfigurationProperties(prefix = "kalshi.live-trading")
@Validated
public class LiveTradingProperties {

  private boolean enabled;

  @Min(5)
  private int marketResolveIntervalSeconds = 60;

  @Min(100)
  private int publishIntervalMillis = 750;

  @Min(1)
  private int topLevelsPerSide = 8;

  @NotBlank
  private String frontendWsPath = "/ws/live-orderbooks";

  @NotBlank
  private String inferenceRootDir = "D:\\Ahmed\\data\\live\\mos_quantile_live_inference";

  @NotBlank
  private String inferenceReportFileName = "inference_report.json";

  @Min(5)
  private int inferenceRefreshIntervalSeconds = 15;

  private boolean inferenceInvokeEnabled;

  @NotBlank
  private String inferenceInvokePythonExecutable = "python";

  @NotBlank
  private String inferenceInvokeScriptPath = "tools/live/mos_quantile_live_inference.py";

  @Min(10)
  private int inferenceInvokeTimeoutSeconds = 180;

  @NotBlank
  private String nextTargetDateCutoffLocalTime = "17:45";

  @DecimalMin("0.0")
  @DecimalMax("1.0")
  private double opportunitiesMinWinProbability = 0.70;

  @DecimalMin("0.0")
  @DecimalMax("1.0")
  private double opportunitiesMinEv = 0.30;

  @DecimalMin("0.0")
  @DecimalMax("1.0")
  private double opportunitiesMinSidePriceProbability = 0.25;

  @Min(1)
  private int opportunitiesMaxCount = 100;

  @NotBlank
  private String strategyReferenceLabel = "2024-2025 | Top #3";

  @NotBlank
  private String strategyPeriodLabel = "2024-10-01 -> 2025-12-31";

  @NotBlank
  private String strategySizingMode = "fractional_kelly";

  @DecimalMin("0.0")
  @DecimalMax("1.0")
  private double strategyKellyFraction = 0.20;

  @DecimalMin("0.0")
  private double strategyStakeCapUsd = 700.0;

  @NotBlank
  private String strategyEntryRule = "Entry >= max(T-1 12:00Z, open+30m)";

  @NotBlank
  private String strategyPredictionSource = "live-script replay";

  @Valid
  @NotEmpty
  private List<Station> stations = defaultStations();

  public boolean isEnabled() {
    return enabled;
  }

  public void setEnabled(boolean enabled) {
    this.enabled = enabled;
  }

  public int getMarketResolveIntervalSeconds() {
    return marketResolveIntervalSeconds;
  }

  public void setMarketResolveIntervalSeconds(int marketResolveIntervalSeconds) {
    this.marketResolveIntervalSeconds = marketResolveIntervalSeconds;
  }

  public int getPublishIntervalMillis() {
    return publishIntervalMillis;
  }

  public void setPublishIntervalMillis(int publishIntervalMillis) {
    this.publishIntervalMillis = publishIntervalMillis;
  }

  public int getTopLevelsPerSide() {
    return topLevelsPerSide;
  }

  public void setTopLevelsPerSide(int topLevelsPerSide) {
    this.topLevelsPerSide = topLevelsPerSide;
  }

  public String getFrontendWsPath() {
    return frontendWsPath;
  }

  public void setFrontendWsPath(String frontendWsPath) {
    this.frontendWsPath = frontendWsPath;
  }

  public String getInferenceRootDir() {
    return inferenceRootDir;
  }

  public void setInferenceRootDir(String inferenceRootDir) {
    this.inferenceRootDir = inferenceRootDir;
  }

  public String getInferenceReportFileName() {
    return inferenceReportFileName;
  }

  public void setInferenceReportFileName(String inferenceReportFileName) {
    this.inferenceReportFileName = inferenceReportFileName;
  }

  public int getInferenceRefreshIntervalSeconds() {
    return inferenceRefreshIntervalSeconds;
  }

  public void setInferenceRefreshIntervalSeconds(int inferenceRefreshIntervalSeconds) {
    this.inferenceRefreshIntervalSeconds = inferenceRefreshIntervalSeconds;
  }

  public boolean isInferenceInvokeEnabled() {
    return inferenceInvokeEnabled;
  }

  public void setInferenceInvokeEnabled(boolean inferenceInvokeEnabled) {
    this.inferenceInvokeEnabled = inferenceInvokeEnabled;
  }

  public String getInferenceInvokePythonExecutable() {
    return inferenceInvokePythonExecutable;
  }

  public void setInferenceInvokePythonExecutable(String inferenceInvokePythonExecutable) {
    this.inferenceInvokePythonExecutable = inferenceInvokePythonExecutable;
  }

  public String getInferenceInvokeScriptPath() {
    return inferenceInvokeScriptPath;
  }

  public void setInferenceInvokeScriptPath(String inferenceInvokeScriptPath) {
    this.inferenceInvokeScriptPath = inferenceInvokeScriptPath;
  }

  public int getInferenceInvokeTimeoutSeconds() {
    return inferenceInvokeTimeoutSeconds;
  }

  public void setInferenceInvokeTimeoutSeconds(int inferenceInvokeTimeoutSeconds) {
    this.inferenceInvokeTimeoutSeconds = inferenceInvokeTimeoutSeconds;
  }

  public String getNextTargetDateCutoffLocalTime() {
    return nextTargetDateCutoffLocalTime;
  }

  public void setNextTargetDateCutoffLocalTime(String nextTargetDateCutoffLocalTime) {
    this.nextTargetDateCutoffLocalTime = nextTargetDateCutoffLocalTime;
  }

  public double getOpportunitiesMinWinProbability() {
    return opportunitiesMinWinProbability;
  }

  public void setOpportunitiesMinWinProbability(double opportunitiesMinWinProbability) {
    this.opportunitiesMinWinProbability = opportunitiesMinWinProbability;
  }

  public int getOpportunitiesMaxCount() {
    return opportunitiesMaxCount;
  }

  public void setOpportunitiesMaxCount(int opportunitiesMaxCount) {
    this.opportunitiesMaxCount = opportunitiesMaxCount;
  }

  public double getOpportunitiesMinEv() {
    return opportunitiesMinEv;
  }

  public void setOpportunitiesMinEv(double opportunitiesMinEv) {
    this.opportunitiesMinEv = opportunitiesMinEv;
  }

  public double getOpportunitiesMinSidePriceProbability() {
    return opportunitiesMinSidePriceProbability;
  }

  public void setOpportunitiesMinSidePriceProbability(double opportunitiesMinSidePriceProbability) {
    this.opportunitiesMinSidePriceProbability = opportunitiesMinSidePriceProbability;
  }

  public String getStrategyReferenceLabel() {
    return strategyReferenceLabel;
  }

  public void setStrategyReferenceLabel(String strategyReferenceLabel) {
    this.strategyReferenceLabel = strategyReferenceLabel;
  }

  public String getStrategyPeriodLabel() {
    return strategyPeriodLabel;
  }

  public void setStrategyPeriodLabel(String strategyPeriodLabel) {
    this.strategyPeriodLabel = strategyPeriodLabel;
  }

  public String getStrategySizingMode() {
    return strategySizingMode;
  }

  public void setStrategySizingMode(String strategySizingMode) {
    this.strategySizingMode = strategySizingMode;
  }

  public double getStrategyKellyFraction() {
    return strategyKellyFraction;
  }

  public void setStrategyKellyFraction(double strategyKellyFraction) {
    this.strategyKellyFraction = strategyKellyFraction;
  }

  public double getStrategyStakeCapUsd() {
    return strategyStakeCapUsd;
  }

  public void setStrategyStakeCapUsd(double strategyStakeCapUsd) {
    this.strategyStakeCapUsd = strategyStakeCapUsd;
  }

  public String getStrategyEntryRule() {
    return strategyEntryRule;
  }

  public void setStrategyEntryRule(String strategyEntryRule) {
    this.strategyEntryRule = strategyEntryRule;
  }

  public String getStrategyPredictionSource() {
    return strategyPredictionSource;
  }

  public void setStrategyPredictionSource(String strategyPredictionSource) {
    this.strategyPredictionSource = strategyPredictionSource;
  }

  public List<Station> getStations() {
    return stations;
  }

  public void setStations(List<Station> stations) {
    this.stations = stations;
  }

  private static List<Station> defaultStations() {
    List<Station> defaults = new ArrayList<>();
    defaults.add(new Station("KNYC", "KXHIGHNY", "America/New_York", "New York"));
    defaults.add(new Station("KMIA", "KXHIGHMIA", "America/New_York", "Miami"));
    defaults.add(new Station("KMDW", "KXHIGHCHI", "America/Chicago", "Chicago Midway"));
    defaults.add(new Station("KLAX", "KXHIGHLAX", "America/Los_Angeles", "Los Angeles"));
    return defaults;
  }

  public static class Station {
    @NotBlank
    private String stationId;
    @NotBlank
    private String seriesTicker;
    @NotBlank
    private String zoneId;
    @NotBlank
    private String displayName;

    public Station() {
    }

    public Station(String stationId, String seriesTicker, String zoneId, String displayName) {
      this.stationId = stationId;
      this.seriesTicker = seriesTicker;
      this.zoneId = zoneId;
      this.displayName = displayName;
    }

    public String getStationId() {
      return stationId;
    }

    public void setStationId(String stationId) {
      this.stationId = stationId;
    }

    public String getSeriesTicker() {
      return seriesTicker;
    }

    public void setSeriesTicker(String seriesTicker) {
      this.seriesTicker = seriesTicker;
    }

    public String getZoneId() {
      return zoneId;
    }

    public void setZoneId(String zoneId) {
      this.zoneId = zoneId;
    }

    public String getDisplayName() {
      return displayName;
    }

    public void setDisplayName(String displayName) {
      this.displayName = displayName;
    }
  }
}
