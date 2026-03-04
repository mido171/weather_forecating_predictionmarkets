package com.predictionmarkets.weather.kalshiapi.http;

import java.time.Duration;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

public class SimpleRateLimiter {

  private static final int MAX_BURST_SECONDS = 60;

  private final int permitsPerSecond;
  private final int maxPermits;
  private final Semaphore semaphore;
  private final ScheduledExecutorService scheduler;

  public SimpleRateLimiter(int permitsPerSecond, String limiterName) {
    if (permitsPerSecond <= 0) {
      throw new IllegalArgumentException("permitsPerSecond must be positive");
    }
    this.permitsPerSecond = permitsPerSecond;
    this.maxPermits = Math.max(permitsPerSecond, permitsPerSecond * MAX_BURST_SECONDS);
    this.semaphore = new Semaphore(permitsPerSecond, true);
    this.scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
      Thread thread = new Thread(r, "kalshi-api-rate-limiter-" + limiterName);
      thread.setDaemon(true);
      return thread;
    });
    this.scheduler.scheduleAtFixedRate(this::refill, 1, 1, TimeUnit.SECONDS);
  }

  public Mono<Void> acquire() {
    return acquire(1);
  }

  public Mono<Void> acquire(int permits) {
    if (permits <= 0) {
      return Mono.error(new IllegalArgumentException("permits must be positive"));
    }
    return Mono.fromCallable(() -> {
          semaphore.acquire(permits);
          return 0;
        })
        .subscribeOn(Schedulers.boundedElastic())
        .timeout(Duration.ofSeconds(MAX_BURST_SECONDS + 5L))
        .then();
  }

  private void refill() {
    int available = semaphore.availablePermits();
    int target = Math.min(maxPermits, available + permitsPerSecond);
    int toRelease = target - available;
    if (toRelease > 0) {
      semaphore.release(toRelease);
    }
  }

  public void destroy() {
    scheduler.shutdownNow();
  }
}
