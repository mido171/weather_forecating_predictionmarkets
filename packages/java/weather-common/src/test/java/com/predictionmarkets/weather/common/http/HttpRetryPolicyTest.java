package com.predictionmarkets.weather.common.http;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.time.Duration;
import java.util.Set;
import org.junit.jupiter.api.Test;

class HttpRetryPolicyTest {
  @Test
  void computeDelayMillis_appliesExponentialBackoffAndJitter() {
    HttpRetryPolicy policy = new HttpRetryPolicy(
        5,
        Duration.ofMillis(100),
        Duration.ofMillis(500),
        Duration.ofMillis(50),
        Set.of(429, 503));

    long delay = policy.computeDelayMillis(2, () -> 25L);
    assertEquals(225L, delay);

    long capped = policy.computeDelayMillis(5, () -> 50L);
    assertEquals(500L, capped);
  }
}
