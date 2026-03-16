package com.predictionmarkets.weather.pilot.source;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;
import org.springframework.stereotype.Component;

@Component
public class RateLimiterRegistry {
  private final Map<String, Semaphore> semaphores = new ConcurrentHashMap<>();

  public <T> T withPermit(String sourceFamily, CheckedSupplier<T> callback) throws Exception {
    Semaphore semaphore = semaphores.computeIfAbsent(sourceFamily == null ? "default" : sourceFamily,
        ignored -> new Semaphore("iem".equalsIgnoreCase(sourceFamily) ? 2 : 4, true));
    semaphore.acquire();
    try {
      return callback.get();
    } finally {
      semaphore.release();
    }
  }

  @FunctionalInterface
  public interface CheckedSupplier<T> {
    T get() throws Exception;
  }
}
