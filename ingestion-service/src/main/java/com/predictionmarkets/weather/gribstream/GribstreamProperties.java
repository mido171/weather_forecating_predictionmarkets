package com.predictionmarkets.weather.gribstream;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "gribstream", ignoreInvalidFields = true)
public class GribstreamProperties {
  private String baseUrl = "https://gribstream.com";
  private String apiToken;
  private String authScheme = "Bearer";
  private int connectTimeoutMillis = 5000;
  private int readTimeoutMillis = 30000;
  private String defaultAccept = "application/ndjson";
  private boolean gzip = true;
  private Map<String, ModelProperties> models = new HashMap<>();
  private GefsProperties gefs = new GefsProperties();

  @Value("${gribstream.models.defaultMinHorizonHours:0}")
  private int defaultMinHorizonHours;

  public String getBaseUrl() {
    return baseUrl;
  }

  public void setBaseUrl(String baseUrl) {
    this.baseUrl = baseUrl;
  }

  public String getApiToken() {
    return apiToken;
  }

  public void setApiToken(String apiToken) {
    this.apiToken = apiToken;
  }

  public String getAuthScheme() {
    return authScheme;
  }

  public void setAuthScheme(String authScheme) {
    this.authScheme = authScheme;
  }

  public int getConnectTimeoutMillis() {
    return connectTimeoutMillis;
  }

  public void setConnectTimeoutMillis(int connectTimeoutMillis) {
    this.connectTimeoutMillis = connectTimeoutMillis;
  }

  public int getReadTimeoutMillis() {
    return readTimeoutMillis;
  }

  public void setReadTimeoutMillis(int readTimeoutMillis) {
    this.readTimeoutMillis = readTimeoutMillis;
  }

  public String getDefaultAccept() {
    return defaultAccept;
  }

  public void setDefaultAccept(String defaultAccept) {
    this.defaultAccept = defaultAccept;
  }

  public boolean isGzip() {
    return gzip;
  }

  public void setGzip(boolean gzip) {
    this.gzip = gzip;
  }

  public Map<String, ModelProperties> getModels() {
    return models;
  }

  public void setModels(Map<String, ModelProperties> models) {
    this.models = models;
  }

  public int getDefaultMinHorizonHours() {
    return defaultMinHorizonHours;
  }

  public GefsProperties getGefs() {
    return gefs;
  }

  public void setGefs(GefsProperties gefs) {
    this.gefs = gefs;
  }

  public static class ModelProperties {
    private int maxHorizonHours;

    public int getMaxHorizonHours() {
      return maxHorizonHours;
    }

    public void setMaxHorizonHours(int maxHorizonHours) {
      this.maxHorizonHours = maxHorizonHours;
    }
  }

  public static class GefsProperties {
    private List<Integer> members = defaultMembers();

    public List<Integer> getMembers() {
      return members;
    }

    public void setMembers(List<Integer> members) {
      this.members = members;
    }

    private static List<Integer> defaultMembers() {
      List<Integer> defaults = new ArrayList<>(31);
      for (int i = 0; i <= 30; i++) {
        defaults.add(i);
      }
      return defaults;
    }
  }
}
