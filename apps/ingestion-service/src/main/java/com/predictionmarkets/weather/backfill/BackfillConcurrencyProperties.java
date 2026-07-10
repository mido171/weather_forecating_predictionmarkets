package com.predictionmarkets.weather.backfill;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "backfill.concurrency")
public class BackfillConcurrencyProperties {
  private int mosWindowThreads = 1;
  private int mosAsofThreads = 1;

  public int getMosWindowThreads() {
    return mosWindowThreads;
  }

  public void setMosWindowThreads(int mosWindowThreads) {
    this.mosWindowThreads = mosWindowThreads;
  }

  public int getMosAsofThreads() {
    return mosAsofThreads;
  }

  public void setMosAsofThreads(int mosAsofThreads) {
    this.mosAsofThreads = mosAsofThreads;
  }
}
