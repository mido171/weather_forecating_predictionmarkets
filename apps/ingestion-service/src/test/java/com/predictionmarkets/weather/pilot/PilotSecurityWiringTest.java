package com.predictionmarkets.weather.pilot;

import static org.assertj.core.api.Assertions.assertThat;

import com.predictionmarkets.weather.pilot.api.IngestAdminController;
import com.predictionmarkets.weather.pilot.api.JobController;
import com.predictionmarkets.weather.pilot.api.StationController;
import com.predictionmarkets.weather.security.AdminApiControlTokenFilter;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.ApplicationContext;

@SpringBootTest(
    classes = PilotIngestionApplication.class,
    properties = {
      "spring.main.web-application-type=none",
      "pilot.knyc.enabled=false",
      "ingestion.admin-api.enabled=true",
      "ingestion.admin-api.control-token=test-control-token"
    })
class PilotSecurityWiringTest {
  @Autowired
  private ApplicationContext applicationContext;

  @Test
  void enabledPilotAdminControllersAlwaysIncludeControlTokenFilter() {
    assertThat(applicationContext.getBeansOfType(AdminApiControlTokenFilter.class)).hasSize(1);
    assertThat(applicationContext.getBeansOfType(JobController.class)).hasSize(1);
    assertThat(applicationContext.getBeansOfType(StationController.class)).hasSize(1);
    assertThat(applicationContext.getBeansOfType(IngestAdminController.class)).hasSize(1);
  }
}
