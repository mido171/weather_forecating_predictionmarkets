package com.predictionmarkets.weather.kalshiapi.config;

import java.time.Clock;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class KalshiCoreConfig {
  @Bean
  public Clock kalshiClock() {
    return Clock.systemUTC();
  }
}
