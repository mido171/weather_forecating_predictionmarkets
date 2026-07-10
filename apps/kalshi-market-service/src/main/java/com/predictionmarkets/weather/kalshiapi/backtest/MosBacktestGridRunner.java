package com.predictionmarkets.weather.kalshiapi.backtest;

import com.predictionmarkets.weather.kalshiapi.config.BacktestGridProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

@Component
@ConditionalOnProperty(prefix = "kalshi.backtest-grid", name = "enabled", havingValue = "true")
public class MosBacktestGridRunner implements CommandLineRunner {

  private static final Logger log = LoggerFactory.getLogger(MosBacktestGridRunner.class);

  private final BacktestGridProperties properties;
  private final MosBacktestGridService service;

  public MosBacktestGridRunner(BacktestGridProperties properties, MosBacktestGridService service) {
    this.properties = properties;
    this.service = service;
  }

  @Override
  public void run(String... args) throws Exception {
    if (!properties.isEnabled()) {
      return;
    }
    log.info("Starting MOS backtest grid run for {}..{} with {} threads",
        properties.getStartDate(), properties.getEndDate(), properties.getThreadCount());
    service.run();
  }
}
