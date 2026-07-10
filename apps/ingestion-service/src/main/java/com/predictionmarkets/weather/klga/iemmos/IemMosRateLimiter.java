package com.predictionmarkets.weather.klga.iemmos;

import java.time.Clock;
import java.time.Duration;
import java.time.Instant;

public class IemMosRateLimiter {
  private final Clock clock;
  private final long spacingMillis;
  private Instant nextAllowedAt;

  public IemMosRateLimiter(long spacingMillis) {
    this(spacingMillis, Clock.systemUTC());
  }

  IemMosRateLimiter(long spacingMillis, Clock clock) {
    if (spacingMillis < 0) {
      throw new IllegalArgumentException("spacingMillis must be >= 0");
    }
    this.spacingMillis = spacingMillis;
    this.clock = clock;
    this.nextAllowedAt = Instant.EPOCH;
  }

  public synchronized void awaitTurn() {
    Instant now = clock.instant();
    if (now.isBefore(nextAllowedAt)) {
      long sleepMillis = Duration.between(now, nextAllowedAt).toMillis();
      sleep(sleepMillis);
      now = clock.instant();
    }
    Instant base = now.isAfter(nextAllowedAt) ? now : nextAllowedAt;
    nextAllowedAt = base.plusMillis(spacingMillis);
  }

  private void sleep(long sleepMillis) {
    if (sleepMillis <= 0) {
      return;
    }
    try {
      Thread.sleep(sleepMillis);
    } catch (InterruptedException ex) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException("IEM MOS request limiter interrupted", ex);
    }
  }
}
