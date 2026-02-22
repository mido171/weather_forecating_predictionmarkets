package com.predictionmarkets.weather.executors.datafetcher;

import com.predictionmarkets.weather.IngestionServiceApplication;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.ConfigurableApplicationContext;

public final class GribstreamTmaxForecastExecutor {
  private GribstreamTmaxForecastExecutor() {
  }

  public static void main(String[] args) {
    try (ConfigurableApplicationContext context = new SpringApplicationBuilder(
        IngestionServiceApplication.class)
        .web(WebApplicationType.NONE)
        .run(args)) {
      GribstreamTmaxForecastFetcher fetcher = context.getBean(GribstreamTmaxForecastFetcher.class);
      fetcher.run();
    }
  }
}

