package com.predictionmarkets.weather;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication
@ConfigurationPropertiesScan
public class IngestionServiceApplication {
  public static void main(String[] args) {
    SpringApplication.run(IngestionServiceApplication.class, args);
  }
}
