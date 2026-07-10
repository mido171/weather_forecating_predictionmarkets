package com.predictionmarkets.weather.pilot.source;

import com.predictionmarkets.weather.pilot.config.PilotIngestionProperties;
import java.util.Locale;
import org.springframework.stereotype.Component;

@Component
public class RetryPolicyFactory {
  private final PilotIngestionProperties properties;

  public RetryPolicyFactory(PilotIngestionProperties properties) {
    this.properties = properties;
  }

  public RetryPolicy forSourceFamily(String sourceFamily) {
    String normalized = sourceFamily == null ? "" : sourceFamily.trim().toLowerCase(Locale.ROOT);
    if ("iem".equals(normalized)) {
      return new RetryPolicy(properties.getMaxHttpRetries(), 300L, 4000L);
    }
    return new RetryPolicy(properties.getMaxHttpRetries(), 500L, 8000L);
  }

  public record RetryPolicy(int maxAttempts, long baseBackoffMs, long maxBackoffMs) {
    public long computeDelayMs(int attempt) {
      long backoff = baseBackoffMs * (1L << Math.max(0, attempt - 1));
      return Math.min(backoff, maxBackoffMs);
    }
  }
}
