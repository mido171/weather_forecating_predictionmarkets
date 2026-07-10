package com.predictionmarkets.weather.pilot;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.data.jpa.JpaRepositoriesAutoConfiguration;
import org.springframework.boot.autoconfigure.flyway.FlywayAutoConfiguration;
import org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration;
import org.springframework.boot.autoconfigure.orm.jpa.HibernateJpaAutoConfiguration;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication(
    scanBasePackages = "com.predictionmarkets.weather.pilot",
    exclude = {
      DataSourceAutoConfiguration.class,
      HibernateJpaAutoConfiguration.class,
      JpaRepositoriesAutoConfiguration.class,
      FlywayAutoConfiguration.class
    })
@ConfigurationPropertiesScan(basePackages = "com.predictionmarkets.weather.pilot")
public class PilotIngestionApplication {
  public static void main(String[] args) {
    SpringApplication.run(PilotIngestionApplication.class, args);
  }
}
