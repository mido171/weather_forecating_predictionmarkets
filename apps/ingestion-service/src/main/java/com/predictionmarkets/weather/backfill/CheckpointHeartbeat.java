package com.predictionmarkets.weather.backfill;

import java.time.Duration;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public final class CheckpointHeartbeat implements AutoCloseable {
  private static final Duration HEARTBEAT_INTERVAL = Duration.ofSeconds(30);
  private static final AtomicInteger HEARTBEAT_COUNTER = new AtomicInteger(1);

  private final ScheduledExecutorService scheduler;

  private CheckpointHeartbeat(Runnable task) {
    ThreadFactory factory = runnable -> {
      Thread thread = new Thread(runnable);
      thread.setDaemon(true);
      thread.setName("checkpoint-heartbeat-" + HEARTBEAT_COUNTER.getAndIncrement());
      return thread;
    };
    this.scheduler = Executors.newSingleThreadScheduledExecutor(factory);
    Runnable guarded = () -> {
      try {
        task.run();
      } catch (RuntimeException ignored) {
        // heartbeat failures should not terminate the backfill loop
      }
    };
    long intervalMillis = HEARTBEAT_INTERVAL.toMillis();
    scheduler.scheduleAtFixedRate(guarded, intervalMillis, intervalMillis, TimeUnit.MILLISECONDS);
  }

  public static CheckpointHeartbeat start(Runnable task) {
    return new CheckpointHeartbeat(task);
  }

  @Override
  public void close() {
    scheduler.shutdownNow();
    try {
      scheduler.awaitTermination(5, TimeUnit.SECONDS);
    } catch (InterruptedException ex) {
      Thread.currentThread().interrupt();
    }
  }
}
