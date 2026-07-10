package com.predictionmarkets.weather.weathercom.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "weathercom")
public class WeatherComProperties {
  private final Api api = new Api();
  private final Ingestion ingestion = new Ingestion();

  public Api getApi() {
    return api;
  }

  public Ingestion getIngestion() {
    return ingestion;
  }

  public static class Api {
    private String baseUrl = "https://api.weather.com";
    private String apiKey;
    private int connectTimeoutMs = 5000;
    private int readTimeoutMs = 20000;
    private String userAgent = "weather-forecasting-predictionmarkets/1.0";

    public String getBaseUrl() {
      return baseUrl;
    }

    public void setBaseUrl(String baseUrl) {
      this.baseUrl = baseUrl;
    }

    public String getApiKey() {
      return apiKey;
    }

    public void setApiKey(String apiKey) {
      this.apiKey = apiKey;
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

    public String getUserAgent() {
      return userAgent;
    }

    public void setUserAgent(String userAgent) {
      this.userAgent = userAgent;
    }
  }

  public static class Ingestion {
    private boolean enabled;
    private int threadPoolSize = 1;
    private int queueCapacity = 10;
    private int chunkDays = 1;
    private int maxRetries = 1;
    private long retryBackoffMs = 500L;
    private long maxBackoffMs = 10000L;
    private long retryJitterMs = 250L;
    private int upsertBatchSize = 100;
    private boolean storeResponseBody;
    private int maxResponseBodyChars;
    private int maxLocationsPerRun = 5;
    private int maxDaysPerRun = 31;
    private int maxTasksPerRun = 200;
    private final RateLimit rateLimit = new RateLimit();

    public boolean isEnabled() {
      return enabled;
    }

    public void setEnabled(boolean enabled) {
      this.enabled = enabled;
    }

    public int getThreadPoolSize() {
      return threadPoolSize;
    }

    public void setThreadPoolSize(int threadPoolSize) {
      this.threadPoolSize = threadPoolSize;
    }

    public int getQueueCapacity() {
      return queueCapacity;
    }

    public void setQueueCapacity(int queueCapacity) {
      this.queueCapacity = queueCapacity;
    }

    public int getChunkDays() {
      return chunkDays;
    }

    public void setChunkDays(int chunkDays) {
      this.chunkDays = chunkDays;
    }

    public int getMaxRetries() {
      return maxRetries;
    }

    public void setMaxRetries(int maxRetries) {
      this.maxRetries = maxRetries;
    }

    public long getRetryBackoffMs() {
      return retryBackoffMs;
    }

    public void setRetryBackoffMs(long retryBackoffMs) {
      this.retryBackoffMs = retryBackoffMs;
    }

    public long getMaxBackoffMs() {
      return maxBackoffMs;
    }

    public void setMaxBackoffMs(long maxBackoffMs) {
      this.maxBackoffMs = maxBackoffMs;
    }

    public long getRetryJitterMs() {
      return retryJitterMs;
    }

    public void setRetryJitterMs(long retryJitterMs) {
      this.retryJitterMs = retryJitterMs;
    }

    public int getUpsertBatchSize() {
      return upsertBatchSize;
    }

    public void setUpsertBatchSize(int upsertBatchSize) {
      this.upsertBatchSize = upsertBatchSize;
    }

    public boolean isStoreResponseBody() {
      return storeResponseBody;
    }

    public void setStoreResponseBody(boolean storeResponseBody) {
      this.storeResponseBody = storeResponseBody;
    }

    public int getMaxResponseBodyChars() {
      return maxResponseBodyChars;
    }

    public void setMaxResponseBodyChars(int maxResponseBodyChars) {
      this.maxResponseBodyChars = maxResponseBodyChars;
    }

    public int getMaxLocationsPerRun() {
      return maxLocationsPerRun;
    }

    public void setMaxLocationsPerRun(int maxLocationsPerRun) {
      this.maxLocationsPerRun = maxLocationsPerRun;
    }

    public int getMaxDaysPerRun() {
      return maxDaysPerRun;
    }

    public void setMaxDaysPerRun(int maxDaysPerRun) {
      this.maxDaysPerRun = maxDaysPerRun;
    }

    public int getMaxTasksPerRun() {
      return maxTasksPerRun;
    }

    public void setMaxTasksPerRun(int maxTasksPerRun) {
      this.maxTasksPerRun = maxTasksPerRun;
    }

    public RateLimit getRateLimit() {
      return rateLimit;
    }
  }

  public static class RateLimit {
    private double permitsPerSecond = 1.0d;

    public double getPermitsPerSecond() {
      return permitsPerSecond;
    }

    public void setPermitsPerSecond(double permitsPerSecond) {
      this.permitsPerSecond = permitsPerSecond;
    }
  }
}

