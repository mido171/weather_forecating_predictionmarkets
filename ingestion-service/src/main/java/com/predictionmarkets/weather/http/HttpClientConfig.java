package com.predictionmarkets.weather.http;

import com.predictionmarkets.weather.common.http.HttpClientSettings;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class HttpClientConfig {
  @Bean
  public HttpClientSettings httpClientSettings() {
    return HttpClientSettings.defaultSettings();
  }
}
