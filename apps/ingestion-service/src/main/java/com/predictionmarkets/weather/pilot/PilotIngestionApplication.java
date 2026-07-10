package com.predictionmarkets.weather.pilot;

import com.predictionmarkets.weather.pilot.config.PilotIngestionProperties;
import com.predictionmarkets.weather.security.AdminApiControlTokenFilter;
import com.predictionmarkets.weather.security.AdminApiProperties;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.data.jpa.JpaRepositoriesAutoConfiguration;
import org.springframework.boot.autoconfigure.flyway.FlywayAutoConfiguration;
import org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration;
import org.springframework.boot.autoconfigure.orm.jpa.HibernateJpaAutoConfiguration;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication(
    scanBasePackageClasses = {
      PilotIngestionApplication.class,
      AdminApiControlTokenFilter.class
    },
    exclude = {
      DataSourceAutoConfiguration.class,
      HibernateJpaAutoConfiguration.class,
      JpaRepositoriesAutoConfiguration.class,
      FlywayAutoConfiguration.class
    })
@ConfigurationPropertiesScan(basePackageClasses = {
  PilotIngestionProperties.class,
  AdminApiProperties.class
})
public class PilotIngestionApplication {
  public static void main(String[] args) {
    SpringApplication.run(PilotIngestionApplication.class, args);
  }
}
