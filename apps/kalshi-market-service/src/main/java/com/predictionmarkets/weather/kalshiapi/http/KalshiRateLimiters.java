package com.predictionmarkets.weather.kalshiapi.http;

import com.predictionmarkets.weather.kalshiapi.config.KalshiExecutionProperties;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.stereotype.Component;

@Component
public class KalshiRateLimiters implements DisposableBean {

  private static final int WRITE_SCALE = 10;

  private final SimpleRateLimiter readLimiter;
  private final SimpleRateLimiter writeLimiter;

  public KalshiRateLimiters(KalshiExecutionProperties properties) {
    this.readLimiter = new SimpleRateLimiter(
        properties.getRateLimiting().getMaxReadRequestsPerSecond(),
        "read");
    int writePermitsPerSecond = Math.max(
        WRITE_SCALE,
        properties.getRateLimiting().getMaxWriteRequestsPerSecond() * WRITE_SCALE);
    this.writeLimiter = new SimpleRateLimiter(writePermitsPerSecond, "write");
  }

  public SimpleRateLimiter limiterFor(RequestType requestType) {
    return requestType == RequestType.WRITE ? writeLimiter : readLimiter;
  }

  public int writeScale() {
    return WRITE_SCALE;
  }

  @Override
  public void destroy() {
    readLimiter.destroy();
    writeLimiter.destroy();
  }
}
