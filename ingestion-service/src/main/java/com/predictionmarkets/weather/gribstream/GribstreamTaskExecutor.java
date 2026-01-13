package com.predictionmarkets.weather.gribstream;

import jakarta.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.springframework.stereotype.Component;

@Component
public class GribstreamTaskExecutor {
  private final ExecutorService executor;

  public GribstreamTaskExecutor(GribstreamExecutorProperties properties) {
    int threadCount = Math.max(1, properties.getThreadCount());
    this.executor = Executors.newFixedThreadPool(threadCount, namedThreadFactory());
  }

  public <T> List<T> invokeAllOrFail(List<Callable<T>> tasks) {
    List<Future<T>> futures = new ArrayList<>(tasks.size());
    for (Callable<T> task : tasks) {
      futures.add(executor.submit(task));
    }
    List<T> results = new ArrayList<>(tasks.size());
    for (Future<T> future : futures) {
      try {
        results.add(future.get());
      } catch (InterruptedException ex) {
        Thread.currentThread().interrupt();
        cancelRemaining(futures);
        throw new IllegalStateException("Gribstream task interrupted", ex);
      } catch (ExecutionException ex) {
        cancelRemaining(futures);
        Throwable cause = ex.getCause() == null ? ex : ex.getCause();
        if (cause instanceof RuntimeException runtimeException) {
          throw runtimeException;
        }
        throw new IllegalStateException("Gribstream task failed", cause);
      }
    }
    return results;
  }

  private void cancelRemaining(List<? extends Future<?>> futures) {
    for (Future<?> future : futures) {
      if (!future.isDone()) {
        future.cancel(true);
      }
    }
  }

  private static ThreadFactory namedThreadFactory() {
    AtomicInteger counter = new AtomicInteger(1);
    return runnable -> {
      Thread thread = new Thread(runnable);
      thread.setName("gribstream-worker-" + counter.getAndIncrement());
      thread.setDaemon(false);
      return thread;
    };
  }

  @PreDestroy
  public void shutdown() {
    executor.shutdown();
    try {
      if (!executor.awaitTermination(30, TimeUnit.SECONDS)) {
        executor.shutdownNow();
      }
    } catch (InterruptedException ex) {
      Thread.currentThread().interrupt();
      executor.shutdownNow();
    }
  }
}
