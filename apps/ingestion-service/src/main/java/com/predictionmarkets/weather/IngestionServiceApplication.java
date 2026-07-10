package com.predictionmarkets.weather;

import com.predictionmarkets.weather.klga.iemmos.IemMosBackfillApplication;
import com.predictionmarkets.weather.pilot.PilotIngestionApplication;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.FilterType;

@SpringBootApplication
@ComponentScan(
    basePackages = "com.predictionmarkets.weather",
    excludeFilters = @ComponentScan.Filter(
        type = FilterType.ASSIGNABLE_TYPE,
        classes = { PilotIngestionApplication.class, IemMosBackfillApplication.class }))
@ConfigurationPropertiesScan
public class IngestionServiceApplication {
  public static void main(String[] args) {
    SpringApplication.run(IngestionServiceApplication.class, args);
  }
}
