package com.predictionmarkets.weather.pilot.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Component
@ConfigurationProperties(prefix = "pilot.knyc")
public class PilotIngestionProperties {
  private boolean enabled;
  private String sqliteRoot;
  private String sqliteFileName;
  private String configDir;
  private String dataDir;
  private String parserVersion;
  private String defaultStationKey;
  private int maxHttpRetries = 1;
  private int connectTimeoutMs = 10000;
  private int readTimeoutMs = 60000;
  private Jobs jobs = new Jobs();

  public boolean isEnabled() {
    return enabled;
  }

  public void setEnabled(boolean enabled) {
    this.enabled = enabled;
  }

  public String getSqliteRoot() {
    return sqliteRoot;
  }

  public void setSqliteRoot(String sqliteRoot) {
    this.sqliteRoot = sqliteRoot;
  }

  public String getSqliteFileName() {
    return sqliteFileName;
  }

  public void setSqliteFileName(String sqliteFileName) {
    this.sqliteFileName = sqliteFileName;
  }

  public String getConfigDir() {
    return configDir;
  }

  public void setConfigDir(String configDir) {
    this.configDir = configDir;
  }

  public String getDataDir() {
    return dataDir;
  }

  public void setDataDir(String dataDir) {
    this.dataDir = dataDir;
  }

  public String getParserVersion() {
    return parserVersion;
  }

  public void setParserVersion(String parserVersion) {
    this.parserVersion = parserVersion;
  }

  public String getDefaultStationKey() {
    return defaultStationKey;
  }

  public void setDefaultStationKey(String defaultStationKey) {
    this.defaultStationKey = defaultStationKey;
  }

  public int getMaxHttpRetries() {
    return maxHttpRetries;
  }

  public void setMaxHttpRetries(int maxHttpRetries) {
    this.maxHttpRetries = maxHttpRetries;
  }

  public int getConnectTimeoutMs() {
    return connectTimeoutMs;
  }

  public void setConnectTimeoutMs(int connectTimeoutMs) {
    this.connectTimeoutMs = connectTimeoutMs;
  }

  public int getReadTimeoutMs() {
    return readTimeoutMs;
  }

  public void setReadTimeoutMs(int readTimeoutMs) {
    this.readTimeoutMs = readTimeoutMs;
  }

  public Jobs getJobs() {
    return jobs;
  }

  public void setJobs(Jobs jobs) {
    this.jobs = jobs == null ? new Jobs() : jobs;
  }

  public static class Jobs {
    private boolean bootstrapEnabled;
    private boolean lightweightEnabled;
    private boolean heavyEnabled;
    private boolean snapshotEnabled;
    private boolean smokeEnabled;
    private String smokeTargetDateLocal;
    private String smokeNdfdHistoricalDateLocal;
    private String smokeThreadexStartDateLocal;
    private String smokeClimodatStartDateLocal;
    private int smokeCliStartYear = 2025;
    private int smokeCliEndYear = 2026;
    private int smokeMosLookbackDays = 30;
    private int smokeHourlyLookbackDays = 14;
    private int smokeOneMinuteLookbackDays = 7;

    public boolean isBootstrapEnabled() {
      return bootstrapEnabled;
    }

    public void setBootstrapEnabled(boolean bootstrapEnabled) {
      this.bootstrapEnabled = bootstrapEnabled;
    }

    public boolean isLightweightEnabled() {
      return lightweightEnabled;
    }

    public void setLightweightEnabled(boolean lightweightEnabled) {
      this.lightweightEnabled = lightweightEnabled;
    }

    public boolean isHeavyEnabled() {
      return heavyEnabled;
    }

    public void setHeavyEnabled(boolean heavyEnabled) {
      this.heavyEnabled = heavyEnabled;
    }

    public boolean isSnapshotEnabled() {
      return snapshotEnabled;
    }

    public void setSnapshotEnabled(boolean snapshotEnabled) {
      this.snapshotEnabled = snapshotEnabled;
    }

    public boolean isSmokeEnabled() {
      return smokeEnabled;
    }

    public void setSmokeEnabled(boolean smokeEnabled) {
      this.smokeEnabled = smokeEnabled;
    }

    public String getSmokeTargetDateLocal() {
      return smokeTargetDateLocal;
    }

    public void setSmokeTargetDateLocal(String smokeTargetDateLocal) {
      this.smokeTargetDateLocal = smokeTargetDateLocal;
    }

    public String getSmokeNdfdHistoricalDateLocal() {
      return smokeNdfdHistoricalDateLocal;
    }

    public void setSmokeNdfdHistoricalDateLocal(String smokeNdfdHistoricalDateLocal) {
      this.smokeNdfdHistoricalDateLocal = smokeNdfdHistoricalDateLocal;
    }

    public String getSmokeThreadexStartDateLocal() {
      return smokeThreadexStartDateLocal;
    }

    public void setSmokeThreadexStartDateLocal(String smokeThreadexStartDateLocal) {
      this.smokeThreadexStartDateLocal = smokeThreadexStartDateLocal;
    }

    public String getSmokeClimodatStartDateLocal() {
      return smokeClimodatStartDateLocal;
    }

    public void setSmokeClimodatStartDateLocal(String smokeClimodatStartDateLocal) {
      this.smokeClimodatStartDateLocal = smokeClimodatStartDateLocal;
    }

    public int getSmokeCliStartYear() {
      return smokeCliStartYear;
    }

    public void setSmokeCliStartYear(int smokeCliStartYear) {
      this.smokeCliStartYear = smokeCliStartYear;
    }

    public int getSmokeCliEndYear() {
      return smokeCliEndYear;
    }

    public void setSmokeCliEndYear(int smokeCliEndYear) {
      this.smokeCliEndYear = smokeCliEndYear;
    }

    public int getSmokeMosLookbackDays() {
      return smokeMosLookbackDays;
    }

    public void setSmokeMosLookbackDays(int smokeMosLookbackDays) {
      this.smokeMosLookbackDays = smokeMosLookbackDays;
    }

    public int getSmokeHourlyLookbackDays() {
      return smokeHourlyLookbackDays;
    }

    public void setSmokeHourlyLookbackDays(int smokeHourlyLookbackDays) {
      this.smokeHourlyLookbackDays = smokeHourlyLookbackDays;
    }

    public int getSmokeOneMinuteLookbackDays() {
      return smokeOneMinuteLookbackDays;
    }

    public void setSmokeOneMinuteLookbackDays(int smokeOneMinuteLookbackDays) {
      this.smokeOneMinuteLookbackDays = smokeOneMinuteLookbackDays;
    }
  }
}
