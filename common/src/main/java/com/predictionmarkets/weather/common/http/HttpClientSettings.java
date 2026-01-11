package com.predictionmarkets.weather.common.http;

import java.time.Duration;
import java.util.Objects;

public record HttpClientSettings(Duration connectTimeout,
                                 Duration readTimeout,
                                 HttpRetryPolicy retryPolicy) {

  public HttpClientSettings {
    Objects.requireNonNull(connectTimeout, "connectTimeout must not be null");
    Objects.requireNonNull(readTimeout, "readTimeout must not be null");
    Objects.requireNonNull(retryPolicy, "retryPolicy must not be null");
    if (connectTimeout.isNegative() || connectTimeout.isZero()) {
      throw new IllegalArgumentException("connectTimeout must be > 0");
    }
    if (readTimeout.isNegative() || readTimeout.isZero()) {
      throw new IllegalArgumentException("readTimeout must be > 0");
    }
  }

  public static HttpClientSettings defaultSettings() {
    return new HttpClientSettings(
        Duration.ofSeconds(10),
        Duration.ofSeconds(30),
        HttpRetryPolicy.defaultPolicy());
  }
}
