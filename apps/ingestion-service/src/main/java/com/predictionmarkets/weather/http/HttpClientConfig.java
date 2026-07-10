package com.predictionmarkets.weather.http;

import com.predictionmarkets.weather.common.http.HttpClientSettings;
import com.predictionmarkets.weather.common.http.HttpRetryPolicy;
import java.time.Duration;
import java.util.Set;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class HttpClientConfig {
  @Bean
  public HttpClientSettings httpClientSettings() {
    return new HttpClientSettings(
        Duration.ofSeconds(10),
        Duration.ofSeconds(30),
        new HttpRetryPolicy(
            2,
            Duration.ofMillis(250),
            Duration.ofSeconds(2),
            Duration.ofMillis(250),
            Set.of(429, 502, 503, 504)));
  }
}
