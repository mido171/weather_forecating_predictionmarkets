package com.predictionmarkets.weather.klga.iemmos;

import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

@Component
public class IemMosBackfillCommandLineRunner implements CommandLineRunner {
  private final IemMosBackfillProperties properties;
  private final IemMosBackfillService service;

  public IemMosBackfillCommandLineRunner(IemMosBackfillProperties properties,
                                         IemMosBackfillService service) {
    this.properties = properties;
    this.service = service;
  }

  @Override
  public void run(String... args) {
    if (!properties.isEnabled()) {
      return;
    }
    service.run(properties);
  }
}
