package com.predictionmarkets.weather.weathercom.config;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class WeatherComExecutorConfig {
  @Bean(destroyMethod = "shutdown")
  public ThreadPoolExecutor weatherComTaskExecutor(WeatherComProperties properties) {
    WeatherComProperties.Ingestion ingestion = properties.getIngestion();
    int threads = Math.max(1, ingestion.getThreadPoolSize());
    int queueCapacity = Math.max(1, ingestion.getQueueCapacity());
    // CallerRunsPolicy keeps ingestion bounded under pressure without silently dropping tasks.
    return new ThreadPoolExecutor(
        threads,
        threads,
        0L,
        TimeUnit.MILLISECONDS,
        new ArrayBlockingQueue<>(queueCapacity),
        namedThreadFactory("weathercom-task"),
        new ThreadPoolExecutor.CallerRunsPolicy());
  }

  @Bean(destroyMethod = "shutdown")
  public ExecutorService weatherComRunExecutor() {
    return Executors.newSingleThreadExecutor(namedThreadFactory("weathercom-run"));
  }

  private ThreadFactory namedThreadFactory(String prefix) {
    AtomicInteger counter = new AtomicInteger(1);
    return runnable -> {
      Thread thread = new Thread(runnable);
      thread.setName(prefix + "-" + counter.getAndIncrement());
      thread.setDaemon(false);
      return thread;
    };
  }
}
