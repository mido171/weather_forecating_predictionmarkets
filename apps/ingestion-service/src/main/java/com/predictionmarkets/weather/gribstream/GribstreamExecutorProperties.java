package com.predictionmarkets.weather.gribstream;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "gribstream.executor")
public class GribstreamExecutorProperties {
  private int threadCount = 1;

  public int getThreadCount() {
    return threadCount;
  }

  public void setThreadCount(int threadCount) {
    this.threadCount = threadCount;
  }
}
