package com.predictionmarkets.weather;

import static org.assertj.core.api.Assertions.assertThat;

import com.predictionmarkets.weather.backfill.BackfillCommandLineRunner;
import com.predictionmarkets.weather.gribstream.GribstreamExampleRunner;
import com.predictionmarkets.weather.pilot.api.IngestAdminController;
import com.predictionmarkets.weather.pilot.api.JobController;
import com.predictionmarkets.weather.pilot.api.StationController;
import com.predictionmarkets.weather.gribstream.GribstreamProperties;
import com.predictionmarkets.weather.weathercom.service.WeatherComBackfillCommandLineRunner;
import com.predictionmarkets.weather.weathercom.web.WeatherComController;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.ApplicationContext;
import org.springframework.test.context.ActiveProfiles;

@SpringBootTest(properties = "gribstream.api-token=")
@ActiveProfiles("test")
class IngestionServiceApplicationTests {
  @Autowired
  private ApplicationContext applicationContext;

  @Autowired
  private GribstreamProperties gribstreamProperties;

  @Test
  void contextLoads() {
    assertThat(applicationContext.getBeansOfType(WeatherComController.class)).isEmpty();
    assertThat(applicationContext.getBeansOfType(JobController.class)).isEmpty();
    assertThat(applicationContext.getBeansOfType(StationController.class)).isEmpty();
    assertThat(applicationContext.getBeansOfType(IngestAdminController.class)).isEmpty();
    assertThat(applicationContext.getBeansOfType(BackfillCommandLineRunner.class)).isEmpty();
    assertThat(applicationContext.getBeansOfType(GribstreamExampleRunner.class)).isEmpty();
    assertThat(applicationContext.getBeansOfType(WeatherComBackfillCommandLineRunner.class)).isEmpty();
    assertThat(gribstreamProperties.getApiToken()).isBlank();
  }
}
