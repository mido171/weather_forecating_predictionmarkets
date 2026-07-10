package com.predictionmarkets.weather.iem;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "iem")
public class IemProperties {
  private String baseUrl = "https://mesonet.agron.iastate.edu";

  public String getBaseUrl() {
    return baseUrl;
  }

  public void setBaseUrl(String baseUrl) {
    this.baseUrl = baseUrl;
  }
}
