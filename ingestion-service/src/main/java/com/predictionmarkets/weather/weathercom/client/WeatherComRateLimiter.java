package com.predictionmarkets.weather.weathercom.client;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;

public class WeatherComRateLimiter {
  private final boolean enabled;
  private final long intervalNanos;
  private final AtomicLong nextAvailableNanos = new AtomicLong(0L);

  public WeatherComRateLimiter(double permitsPerSecond) {
    if (permitsPerSecond <= 0.0d) {
      this.enabled = false;
      this.intervalNanos = 0L;
      return;
    }
    this.enabled = true;
    this.intervalNanos = Math.max(1L, (long) (1_000_000_000d / permitsPerSecond));
  }

  public void acquire() {
    if (!enabled) {
      return;
    }
    while (true) {
      long now = System.nanoTime();
      long current = nextAvailableNanos.get();
      long target = Math.max(now, current);
      long updated = target + intervalNanos;
      if (nextAvailableNanos.compareAndSet(current, updated)) {
        long waitNanos = target - now;
        if (waitNanos > 0) {
          LockSupport.parkNanos(waitNanos);
        }
        return;
      }
    }
  }
}

