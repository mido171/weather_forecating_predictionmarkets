package com.predictionmarkets.weather.kalshiapi.http;

import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Tag;
import java.time.Duration;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.ClientRequest;
import org.springframework.web.reactive.function.client.ClientResponse;
import org.springframework.web.reactive.function.client.ExchangeFilterFunction;
import org.springframework.web.reactive.function.client.ExchangeFunction;
import reactor.core.publisher.Mono;

@Component
public class KalshiObservabilityFilter implements ExchangeFilterFunction {

  private static final Logger log = LoggerFactory.getLogger(KalshiObservabilityFilter.class);

  private final MeterRegistry meterRegistry;

  public KalshiObservabilityFilter(MeterRegistry meterRegistry) {
    this.meterRegistry = meterRegistry;
  }

  @Override
  public Mono<ClientResponse> filter(ClientRequest request, ExchangeFunction next) {
    long startNanos = System.nanoTime();
    String endpoint = request.url().getRawPath();
    String method = request.method().name();

    return next.exchange(request)
        .doOnNext(response -> recordSuccess(method, endpoint, response, startNanos))
        .doOnError(error -> recordError(method, endpoint, startNanos, error));
  }

  private void recordSuccess(String method, String endpoint, ClientResponse response, long startNanos) {
    int status = response.statusCode().value();
    Duration latency = Duration.ofNanos(System.nanoTime() - startNanos);
    recordMetrics(method, endpoint, status, latency);
    log.info("Kalshi REST {} {} -> {} ({} ms)", method, endpoint, status, latency.toMillis());
  }

  private void recordError(String method, String endpoint, long startNanos, Throwable error) {
    Duration latency = Duration.ofNanos(System.nanoTime() - startNanos);
    recordMetrics(method, endpoint, -1, latency);
    log.warn("Kalshi REST {} {} failed after {} ms: {}", method, endpoint, latency.toMillis(), error.toString());
  }

  private void recordMetrics(String method, String endpoint, int status, Duration latency) {
    List<Tag> tags = List.of(
        Tag.of("method", method),
        Tag.of("endpoint", endpoint),
        Tag.of("status", Integer.toString(status))
    );

    meterRegistry.counter("kalshi.api.http.requests", tags).increment();
    meterRegistry.timer("kalshi.api.http.latency", tags).record(latency);
  }
}
