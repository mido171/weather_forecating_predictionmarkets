package com.predictionmarkets.weather.common.http;

import java.time.Duration;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.LongSupplier;

public record HttpRetryPolicy(int maxAttempts,
                              Duration baseBackoff,
                              Duration maxBackoff,
                              Duration jitter,
                              Set<Integer> retryableStatusCodes) {

  public HttpRetryPolicy {
    if (maxAttempts < 1) {
      throw new IllegalArgumentException("maxAttempts must be >= 1");
    }
    Objects.requireNonNull(baseBackoff, "baseBackoff must not be null");
    Objects.requireNonNull(maxBackoff, "maxBackoff must not be null");
    Objects.requireNonNull(jitter, "jitter must not be null");
    Objects.requireNonNull(retryableStatusCodes, "retryableStatusCodes must not be null");
    if (baseBackoff.isNegative()) {
      throw new IllegalArgumentException("baseBackoff must be >= 0");
    }
    if (maxBackoff.isNegative()) {
      throw new IllegalArgumentException("maxBackoff must be >= 0");
    }
    if (jitter.isNegative()) {
      throw new IllegalArgumentException("jitter must be >= 0");
    }
    if (maxBackoff.compareTo(baseBackoff) < 0) {
      throw new IllegalArgumentException("maxBackoff must be >= baseBackoff");
    }
  }

  public static HttpRetryPolicy defaultPolicy() {
    return new HttpRetryPolicy(
        5,
        Duration.ofMillis(250),
        Duration.ofSeconds(2),
        Duration.ofMillis(250),
        Set.of(429, 502, 503, 504));
  }

  public boolean isRetryableStatus(int statusCode) {
    return retryableStatusCodes.contains(statusCode);
  }

  public long computeDelayMillis(int attempt) {
    return computeDelayMillis(attempt, this::defaultJitterMillis);
  }

  public long computeDelayMillis(int attempt, LongSupplier jitterSupplier) {
    if (attempt < 1) {
      throw new IllegalArgumentException("attempt must be >= 1");
    }
    Objects.requireNonNull(jitterSupplier, "jitterSupplier must not be null");
    long baseMillis = baseBackoff.toMillis();
    long maxMillis = maxBackoff.toMillis();
    int exponent = Math.min(attempt - 1, 30);
    long exponential = baseMillis * (1L << exponent);
    long capped = Math.min(exponential, maxMillis);
    long jitterMillis = jitter.toMillis();
    long jitterValue = jitterMillis > 0 ? clampJitter(jitterSupplier.getAsLong(), jitterMillis) : 0L;
    long delay = capped + jitterValue;
    return Math.min(delay, maxMillis);
  }

  private long defaultJitterMillis() {
    long jitterMillis = jitter.toMillis();
    if (jitterMillis <= 0) {
      return 0L;
    }
    return ThreadLocalRandom.current().nextLong(0, jitterMillis + 1);
  }

  private long clampJitter(long candidate, long maxJitter) {
    if (candidate < 0) {
      return 0L;
    }
    return Math.min(candidate, maxJitter);
  }
}
