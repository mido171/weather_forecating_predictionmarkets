package com.predictionmarkets.weather.kalshiapi.security;

import com.predictionmarkets.weather.kalshiapi.config.LiveTradingProperties;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import org.springframework.core.Ordered;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;
import org.springframework.web.server.WebFilter;
import org.springframework.web.server.WebFilterChain;
import reactor.core.publisher.Mono;

@Component
public class LocalControlTokenWebFilter implements WebFilter, Ordered {
  public static final String CONTROL_TOKEN_HEADER = "X-Local-Control-Token";

  private final LiveTradingProperties liveTradingProperties;
  private final LocalControlProperties localControlProperties;

  public LocalControlTokenWebFilter(LiveTradingProperties liveTradingProperties,
                                    LocalControlProperties localControlProperties) {
    this.liveTradingProperties = liveTradingProperties;
    this.localControlProperties = localControlProperties;
  }

  @Override
  public Mono<Void> filter(ServerWebExchange exchange, WebFilterChain chain) {
    String path = exchange.getRequest().getPath().pathWithinApplication().value();
    if (!isSensitivePath(path)) {
      return chain.filter(exchange);
    }
    if (!liveTradingProperties.isEnabled()) {
      exchange.getResponse().setStatusCode(HttpStatus.NOT_FOUND);
      return exchange.getResponse().setComplete();
    }
    if (exchange.getRequest().getMethod() == HttpMethod.OPTIONS) {
      return chain.filter(exchange);
    }

    String suppliedToken = exchange.getRequest().getHeaders().getFirst(CONTROL_TOKEN_HEADER);
    if (!tokenMatches(localControlProperties.getToken(), suppliedToken)) {
      exchange.getResponse().setStatusCode(HttpStatus.UNAUTHORIZED);
      return exchange.getResponse().setComplete();
    }
    return chain.filter(exchange);
  }

  @Override
  public int getOrder() {
    return Ordered.HIGHEST_PRECEDENCE;
  }

  private boolean isSensitivePath(String path) {
    return path.startsWith("/api/live-trading/account")
        || path.startsWith("/api/live-trading/inference");
  }

  private boolean tokenMatches(String expected, String supplied) {
    if (expected == null || expected.isBlank() || supplied == null) {
      return false;
    }
    return MessageDigest.isEqual(
        expected.getBytes(StandardCharsets.UTF_8),
        supplied.getBytes(StandardCharsets.UTF_8));
  }
}
