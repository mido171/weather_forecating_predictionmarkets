package com.predictionmarkets.weather.kalshiapi.http;

import com.predictionmarkets.weather.kalshiapi.auth.KalshiSignerProvider;
import com.predictionmarkets.weather.kalshiapi.auth.SignedHeaders;
import com.predictionmarkets.weather.kalshiapi.config.KalshiExecutionProperties;
import java.net.URI;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.ClientRequest;
import org.springframework.web.reactive.function.client.ClientResponse;
import org.springframework.web.reactive.function.client.ExchangeFilterFunction;
import org.springframework.web.reactive.function.client.ExchangeFunction;
import reactor.core.publisher.Mono;

@Component
public class KalshiAuthFilter implements ExchangeFilterFunction {

  private static final Logger log = LoggerFactory.getLogger(KalshiAuthFilter.class);

  private final KalshiSignerProvider signerProvider;
  private final KalshiExecutionProperties properties;

  public KalshiAuthFilter(KalshiSignerProvider signerProvider,
                          KalshiExecutionProperties properties) {
    this.signerProvider = signerProvider;
    this.properties = properties;
  }

  @Override
  public Mono<ClientResponse> filter(ClientRequest request, ExchangeFunction next) {
    URI url = request.url();
    String path = url.getRawPath();
    SignedHeaders signedHeaders = signerProvider.getSigner().sign(request.method().name(), path);

    if (properties.isAuthDebug()) {
      log.info("Kalshi auth signing method={} path={} tsMs={} key={}",
          request.method().name(),
          path,
          signedHeaders.accessTimestamp(),
          maskAccessKey(signedHeaders.accessKey()));
    }

    ClientRequest signedRequest = ClientRequest.from(request)
        .headers(headers -> signedHeaders.apply(headers))
        .build();

    return next.exchange(signedRequest);
  }

  private static String maskAccessKey(String accessKey) {
    if (accessKey == null || accessKey.isBlank()) {
      return "<missing>";
    }
    String trimmed = accessKey.trim();
    if (trimmed.length() <= 8) {
      return "****";
    }
    return trimmed.substring(0, 4) + "..." + trimmed.substring(trimmed.length() - 4);
  }
}
