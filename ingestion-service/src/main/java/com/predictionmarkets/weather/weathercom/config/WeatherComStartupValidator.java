package com.predictionmarkets.weather.weathercom.config;

import org.springframework.stereotype.Component;

@Component
public class WeatherComStartupValidator {
  public WeatherComStartupValidator(WeatherComProperties properties) {
    if (!properties.getIngestion().isEnabled()) {
      return;
    }
    String apiKey = properties.getApi().getApiKey();
    if (apiKey == null || apiKey.isBlank()) {
      throw new IllegalStateException(
          "weathercom.ingestion.enabled=true requires weathercom.api.api-key "
              + "(set WEATHERCOM_API_KEY)");
    }
  }
}

