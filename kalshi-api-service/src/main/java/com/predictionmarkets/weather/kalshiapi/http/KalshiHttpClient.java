package com.predictionmarkets.weather.kalshiapi.http;

import com.predictionmarkets.weather.kalshiapi.config.KalshiExecutionProperties;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.BatchCancelOrdersRequest;
import java.net.URI;
import java.time.Duration;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.HttpMethod;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.util.UriBuilder;
import reactor.core.publisher.Mono;
import reactor.util.retry.Retry;

@Component
public class KalshiHttpClient {

  private static final Logger log = LoggerFactory.getLogger(KalshiHttpClient.class);
  private static final String BATCHED_ORDERS_PATH = "/portfolio/orders/batched";
  private static final double BATCH_CANCEL_WEIGHT = 0.2d;

  private final WebClient publicWebClient;
  private final WebClient authWebClient;
  private final KalshiExecutionProperties properties;
  private final KalshiRateLimiters rateLimiters;
  private final KalshiErrorParser errorParser;
  private final Duration requestTimeout;
  private final String restBasePathNormalized;

  public KalshiHttpClient(@Qualifier("kalshiPublicWebClient") WebClient publicWebClient,
                          @Qualifier("kalshiAuthWebClient") WebClient authWebClient,
                          KalshiExecutionProperties properties,
                          KalshiRateLimiters rateLimiters,
                          KalshiErrorParser errorParser) {
    this.publicWebClient = publicWebClient;
    this.authWebClient = authWebClient;
    this.properties = properties;
    this.rateLimiters = rateLimiters;
    this.errorParser = errorParser;
    this.requestTimeout = Duration.ofMillis(properties.getTimeouts().getRequestTimeoutMs());
    this.restBasePathNormalized = normalizeBasePath(properties.resolvedRestBaseUrl());
  }

  public <T> T getPublic(String path, Consumer<UriBuilder> uriCustomizer, Class<T> responseType) {
    return exchangeBlocking(RequestType.READ, false, HttpMethod.GET, path, uriCustomizer, null, responseType);
  }

  public <T> T getAuth(String path, Consumer<UriBuilder> uriCustomizer, Class<T> responseType) {
    return exchangeBlocking(RequestType.READ, true, HttpMethod.GET, path, uriCustomizer, null, responseType);
  }

  public <T, B> T postAuth(String path, B body, Class<T> responseType) {
    return exchangeBlocking(RequestType.WRITE, true, HttpMethod.POST, path, null, body, responseType);
  }

  public <T, B> T postAuth(String path, Consumer<UriBuilder> uriCustomizer, B body, Class<T> responseType) {
    return exchangeBlocking(RequestType.WRITE, true, HttpMethod.POST, path, uriCustomizer, body, responseType);
  }

  public <T> T deleteAuth(String path, Class<T> responseType) {
    return exchangeBlocking(RequestType.WRITE, true, HttpMethod.DELETE, path, null, null, responseType);
  }

  public <T, B> T deleteAuth(String path, B body, Class<T> responseType) {
    return exchangeBlocking(RequestType.WRITE, true, HttpMethod.DELETE, path, null, body, responseType);
  }

  public <T> Mono<T> exchangeMono(RequestType requestType,
                                  boolean authenticated,
                                  HttpMethod method,
                                  String path,
                                  Consumer<UriBuilder> uriCustomizer,
                                  Object body,
                                  Class<T> responseType) {
    if (requestType == RequestType.WRITE && !properties.isTradingEnabled()) {
      return Mono.error(new TradingDisabledException("Kalshi trading is disabled via kalshi.trading-enabled=false"));
    }

    SimpleRateLimiter limiter = rateLimiters.limiterFor(requestType);
    WebClient client = authenticated ? authWebClient : publicWebClient;
    String sanitizedPath = sanitizePath(path);
    int permitsNeeded = permitsNeeded(requestType, method, sanitizedPath, body);

    Mono<T> call = limiter.acquire(permitsNeeded)
        .then(buildRequest(client, method, sanitizedPath, uriCustomizer, body))
        .flatMap(request -> request.exchangeToMono(response -> handleResponse(response, responseType)))
        .timeout(requestTimeout);

    if (requestType == RequestType.READ) {
      Retry retry = buildReadRetry();
      if (retry != null) {
        call = call.retryWhen(retry);
      }
    }

    return call;
  }

  private <T> T exchangeBlocking(RequestType requestType,
                                 boolean authenticated,
                                 HttpMethod method,
                                 String path,
                                 Consumer<UriBuilder> uriCustomizer,
                                 Object body,
                                 Class<T> responseType) {
    return exchangeMono(requestType, authenticated, method, path, uriCustomizer, body, responseType)
        .block(requestTimeout.plusSeconds(5));
  }

  private Mono<WebClient.RequestHeadersSpec<?>> buildRequest(WebClient client,
                                                             HttpMethod method,
                                                             String sanitizedPath,
                                                             Consumer<UriBuilder> uriCustomizer,
                                                             Object body) {
    return Mono.fromSupplier(() -> {
      WebClient.RequestBodySpec requestSpec = client.method(method)
          .uri(uriBuilder -> buildUri(uriBuilder, sanitizedPath, uriCustomizer));

      if (body != null) {
        return requestSpec.bodyValue(body);
      }
      return requestSpec;
    });
  }

  private URI buildUri(UriBuilder uriBuilder, String sanitizedPath, Consumer<UriBuilder> uriCustomizer) {
    UriBuilder builder = uriBuilder.path(sanitizedPath);
    if (uriCustomizer != null) {
      uriCustomizer.accept(builder);
    }
    return builder.build();
  }

  private String sanitizePath(String path) {
    if (path == null || path.isBlank()) {
      throw new IllegalArgumentException("Kalshi path must not be blank");
    }
    String withoutQuery = path.split("\\?")[0].trim();
    String normalized = withoutQuery.startsWith("/") ? withoutQuery : "/" + withoutQuery;

    if (!restBasePathNormalized.isBlank()) {
      if (normalized.equals(restBasePathNormalized)) {
        normalized = "/";
      } else if (normalized.startsWith(restBasePathNormalized + "/")) {
        normalized = normalized.substring(restBasePathNormalized.length());
      }
    }

    while (normalized.startsWith("//")) {
      normalized = normalized.substring(1);
    }
    return normalized;
  }

  private String normalizeBasePath(String baseUrl) {
    URI uri = URI.create(baseUrl);
    String path = uri.getPath();
    if (path == null || path.isBlank()) {
      return "";
    }
    String normalized = path.startsWith("/") ? path : "/" + path;
    while (normalized.endsWith("/") && normalized.length() > 1) {
      normalized = normalized.substring(0, normalized.length() - 1);
    }
    return normalized;
  }

  int permitsNeeded(RequestType requestType, HttpMethod method, String sanitizedPath, Object body) {
    if (requestType != RequestType.WRITE) {
      return 1;
    }

    int writeScale = rateLimiters.writeScale();
    int defaultPermits = writeScale;

    if (!BATCHED_ORDERS_PATH.equals(sanitizedPath)) {
      return defaultPermits;
    }
    if (method == HttpMethod.POST) {
      return defaultPermits;
    }
    if (method == HttpMethod.DELETE) {
      int items = countBatchCancelItems(body);
      int weightedPermits = (int) Math.ceil(items * BATCH_CANCEL_WEIGHT * writeScale);
      return Math.max(1, weightedPermits);
    }

    return defaultPermits;
  }

  private int countBatchCancelItems(Object body) {
    if (body instanceof BatchCancelOrdersRequest request && request.ids() != null && !request.ids().isEmpty()) {
      return request.ids().size();
    }
    return 1;
  }

  private <T> Mono<T> handleResponse(org.springframework.web.reactive.function.client.ClientResponse response,
                                     Class<T> responseType) {
    if (response.statusCode().is2xxSuccessful()) {
      return response.bodyToMono(responseType);
    }

    return response.bodyToMono(String.class)
        .defaultIfEmpty("")
        .flatMap(body -> Mono.error(new KalshiApiException(response.statusCode(), body, errorParser.parse(body))));
  }

  private Retry buildReadRetry() {
    int maxRetries = properties.getRetry().getMaxRetries();
    if (maxRetries <= 0) {
      return null;
    }

    Duration baseBackoff = Duration.ofMillis(properties.getRetry().getBaseBackoffMs());
    Duration maxBackoff = Duration.ofMillis(properties.getRetry().getMaxBackoffMs());

    return Retry.backoff(maxRetries, baseBackoff)
        .maxBackoff(maxBackoff)
        .jitter(0.5d)
        .filter(this::isRetryable)
        .doBeforeRetry(signal -> log.warn(
            "Retrying Kalshi read request due to {} (attempt {}/{})",
            signal.failure().toString(),
            signal.totalRetries() + 1,
            maxRetries
        ))
        .onRetryExhaustedThrow((spec, signal) -> signal.failure());
  }

  private boolean isRetryable(Throwable throwable) {
    if (throwable instanceof KalshiApiException apiException) {
      int status = apiException.getStatusCode();
      return status == 429 || (status >= 500 && status <= 599);
    }
    return false;
  }
}
