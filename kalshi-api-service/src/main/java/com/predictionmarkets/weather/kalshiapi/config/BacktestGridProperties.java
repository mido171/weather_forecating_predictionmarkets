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

@ConfigurationProperties(prefix = "kalshi.backtest-grid")
@Validated
public class BacktestGridProperties {

  private boolean enabled = false;

  @NotBlank
  private String startDate = "2024-10-01";

  @NotBlank
  private String endDate = "2025-12-31";

  @Min(0)
  private int entryHourZ = 12;

  @Min(0)
  private int entryMinuteZ = 0;

  @Min(0)
  private int minEntryMinutesAfterOpen = 30;

  @DecimalMin("0.0")
  @DecimalMax("1.0")
  private double minMarketPrice = 0.25;

  @DecimalMin("0.0")
  private double startBalance = 2700.0;

  @DecimalMin("0.0")
  private double stakeCapUsd = 700.0;

  @DecimalMin("0.0")
  @DecimalMax("1.0")
  private double selectionRiskFraction = 0.075;

  @DecimalMin("0.0")
  @DecimalMax("1.0")
  private double evStart = 0.15;

  @DecimalMin("0.0")
  @DecimalMax("1.0")
  private double evEnd = 0.55;

  @DecimalMin("0.0001")
  private double evStep = 0.05;

  @DecimalMin("0.0")
  @DecimalMax("1.0")
  private double winStart = 0.65;

  @DecimalMin("0.0")
  @DecimalMax("1.0")
  private double winEnd = 0.90;

  @DecimalMin("0.0001")
  private double winStep = 0.05;

  @DecimalMin("0.0")
  @DecimalMax("1.0")
  private double fixedRiskStart = 0.045;

  @DecimalMin("0.0")
  @DecimalMax("1.0")
  private double fixedRiskEnd = 0.085;

  @DecimalMin("0.0001")
  private double fixedRiskStep = 0.01;

  @DecimalMin("0.0")
  @DecimalMax("1.0")
  private double kellyStart = 0.10;

  @DecimalMin("0.0")
  @DecimalMax("1.0")
  private double kellyEnd = 0.25;

  @DecimalMin("0.0001")
  private double kellyStep = 0.01;

  @Min(1)
  private int threadCount = 40;

  @NotBlank
  private String sqlitePath = "D:\\Ahmed\\data\\sqlite\\mos_live_script_4station_grid_2024_2025_spring.sqlite";

  @NotBlank
  private String outDir = "D:\\Ahmed\\data\\kalshi\\Experiments\\MOS\\05_backtest\\live_replay_grid_4station_2024_2025_spring";

  @NotBlank
  private String runPrefix = "cojoined_blend12_knyc_kmia_kmdw_klax_tminus1_1200z_openplus30m_live_script_grid_2024_2025_spring";

  @NotBlank
  private String liveInferenceRoot = "D:\\Ahmed\\data\\live\\mos_quantile_live_inference\\backtest_replay_knyc_kmia_kmdw_klax_2024_2025";

  @NotBlank
  private String liveScriptPath = "tools/live/mos_quantile_live_inference.py";

  @NotBlank
  private String liveScriptPython = "python";

  @NotBlank
  private String liveScriptLogLevel = "ERROR";

  private boolean overwriteSqlite = true;

  @Valid
  @NotEmpty
  private List<StationSpec> stations = defaultStations();

  public boolean isEnabled() {
    return enabled;
  }

  public void setEnabled(boolean enabled) {
    this.enabled = enabled;
  }

  public String getStartDate() {
    return startDate;
  }

  public void setStartDate(String startDate) {
    this.startDate = startDate;
  }

  public String getEndDate() {
    return endDate;
  }

  public void setEndDate(String endDate) {
    this.endDate = endDate;
  }

  public int getEntryHourZ() {
    return entryHourZ;
  }

  public void setEntryHourZ(int entryHourZ) {
    this.entryHourZ = entryHourZ;
  }

  public int getEntryMinuteZ() {
    return entryMinuteZ;
  }

  public void setEntryMinuteZ(int entryMinuteZ) {
    this.entryMinuteZ = entryMinuteZ;
  }

  public int getMinEntryMinutesAfterOpen() {
    return minEntryMinutesAfterOpen;
  }

  public void setMinEntryMinutesAfterOpen(int minEntryMinutesAfterOpen) {
    this.minEntryMinutesAfterOpen = minEntryMinutesAfterOpen;
  }

  public double getMinMarketPrice() {
    return minMarketPrice;
  }

  public void setMinMarketPrice(double minMarketPrice) {
    this.minMarketPrice = minMarketPrice;
  }

  public double getStartBalance() {
    return startBalance;
  }

  public void setStartBalance(double startBalance) {
    this.startBalance = startBalance;
  }

  public double getStakeCapUsd() {
    return stakeCapUsd;
  }

  public void setStakeCapUsd(double stakeCapUsd) {
    this.stakeCapUsd = stakeCapUsd;
  }

  public double getSelectionRiskFraction() {
    return selectionRiskFraction;
  }

  public void setSelectionRiskFraction(double selectionRiskFraction) {
    this.selectionRiskFraction = selectionRiskFraction;
  }

  public double getEvStart() {
    return evStart;
  }

  public void setEvStart(double evStart) {
    this.evStart = evStart;
  }

  public double getEvEnd() {
    return evEnd;
  }

  public void setEvEnd(double evEnd) {
    this.evEnd = evEnd;
  }

  public double getEvStep() {
    return evStep;
  }

  public void setEvStep(double evStep) {
    this.evStep = evStep;
  }

  public double getWinStart() {
    return winStart;
  }

  public void setWinStart(double winStart) {
    this.winStart = winStart;
  }

  public double getWinEnd() {
    return winEnd;
  }

  public void setWinEnd(double winEnd) {
    this.winEnd = winEnd;
  }

  public double getWinStep() {
    return winStep;
  }

  public void setWinStep(double winStep) {
    this.winStep = winStep;
  }

  public double getFixedRiskStart() {
    return fixedRiskStart;
  }

  public void setFixedRiskStart(double fixedRiskStart) {
    this.fixedRiskStart = fixedRiskStart;
  }

  public double getFixedRiskEnd() {
    return fixedRiskEnd;
  }

  public void setFixedRiskEnd(double fixedRiskEnd) {
    this.fixedRiskEnd = fixedRiskEnd;
  }

  public double getFixedRiskStep() {
    return fixedRiskStep;
  }

  public void setFixedRiskStep(double fixedRiskStep) {
    this.fixedRiskStep = fixedRiskStep;
  }

  public double getKellyStart() {
    return kellyStart;
  }

  public void setKellyStart(double kellyStart) {
    this.kellyStart = kellyStart;
  }

  public double getKellyEnd() {
    return kellyEnd;
  }

  public void setKellyEnd(double kellyEnd) {
    this.kellyEnd = kellyEnd;
  }

  public double getKellyStep() {
    return kellyStep;
  }

  public void setKellyStep(double kellyStep) {
    this.kellyStep = kellyStep;
  }

  public int getThreadCount() {
    return threadCount;
  }

  public void setThreadCount(int threadCount) {
    this.threadCount = threadCount;
  }

  public String getSqlitePath() {
    return sqlitePath;
  }

  public void setSqlitePath(String sqlitePath) {
    this.sqlitePath = sqlitePath;
  }

  public String getOutDir() {
    return outDir;
  }

  public void setOutDir(String outDir) {
    this.outDir = outDir;
  }

  public String getRunPrefix() {
    return runPrefix;
  }

  public void setRunPrefix(String runPrefix) {
    this.runPrefix = runPrefix;
  }

  public String getLiveInferenceRoot() {
    return liveInferenceRoot;
  }

  public void setLiveInferenceRoot(String liveInferenceRoot) {
    this.liveInferenceRoot = liveInferenceRoot;
  }

  public String getLiveScriptPath() {
    return liveScriptPath;
  }

  public void setLiveScriptPath(String liveScriptPath) {
    this.liveScriptPath = liveScriptPath;
  }

  public String getLiveScriptPython() {
    return liveScriptPython;
  }

  public void setLiveScriptPython(String liveScriptPython) {
    this.liveScriptPython = liveScriptPython;
  }

  public String getLiveScriptLogLevel() {
    return liveScriptLogLevel;
  }

  public void setLiveScriptLogLevel(String liveScriptLogLevel) {
    this.liveScriptLogLevel = liveScriptLogLevel;
  }

  public boolean isOverwriteSqlite() {
    return overwriteSqlite;
  }

  public void setOverwriteSqlite(boolean overwriteSqlite) {
    this.overwriteSqlite = overwriteSqlite;
  }

  public List<StationSpec> getStations() {
    return stations;
  }

  public void setStations(List<StationSpec> stations) {
    this.stations = stations;
  }

  private static List<StationSpec> defaultStations() {
    List<StationSpec> defaults = new ArrayList<>();
    defaults.add(new StationSpec("KNYC",
        "D:\\Ahmed\\data\\kalshi\\training_data\\02_truth\\KNYC_settled_tmax.csv",
        "D:\\Ahmed\\data\\kalshi\\kalshi_history\\kxhighny_2024_10_01_to_2025_12_31",
        "KNYC"));
    defaults.add(new StationSpec("KMIA",
        "D:\\Ahmed\\data\\kalshi\\training_data\\02_truth\\KMIA_settled_tmax.csv",
        "D:\\Ahmed\\data\\kalshi\\kalshi_history\\kxhighmia_2024_10_01_to_2025_12_31",
        "KMIA"));
    defaults.add(new StationSpec("KMDW",
        "D:\\Ahmed\\data\\kalshi\\training_data\\02_truth\\KMDW_settled_tmax_2002_2026.csv",
        "D:\\Ahmed\\data\\kalshi\\kalshi_history\\kxhighchi_2024_10_01_to_2026_03_03",
        "KMDW"));
    defaults.add(new StationSpec("KLAX",
        "D:\\Ahmed\\data\\kalshi\\training_data\\02_truth\\KLAX_settled_tmax_2002_2026.csv",
        "D:\\Ahmed\\data\\kalshi\\kalshi_history\\kxhighlax_2025_01_01_to_2026_03_05",
        "KLAX"));
    return defaults;
  }

  public static class StationSpec {
    @NotBlank
    private String stationId;
    @NotBlank
    private String truthCsvPath;
    @NotBlank
    private String kalshiRoot;
    @NotBlank
    private String filePrefix;

    public StationSpec() {
    }

    public StationSpec(String stationId, String truthCsvPath, String kalshiRoot, String filePrefix) {
      this.stationId = stationId;
      this.truthCsvPath = truthCsvPath;
      this.kalshiRoot = kalshiRoot;
      this.filePrefix = filePrefix;
    }

    public String getStationId() {
      return stationId;
    }

    public void setStationId(String stationId) {
      this.stationId = stationId;
    }

    public String getTruthCsvPath() {
      return truthCsvPath;
    }

    public void setTruthCsvPath(String truthCsvPath) {
      this.truthCsvPath = truthCsvPath;
    }

    public String getKalshiRoot() {
      return kalshiRoot;
    }

    public void setKalshiRoot(String kalshiRoot) {
      this.kalshiRoot = kalshiRoot;
    }

    public String getFilePrefix() {
      return filePrefix;
    }

    public void setFilePrefix(String filePrefix) {
      this.filePrefix = filePrefix;
    }
  }
}
