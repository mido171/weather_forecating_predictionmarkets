package com.predictionmarkets.weather.kalshiapi.security;

import static org.assertj.core.api.Assertions.assertThat;

import com.predictionmarkets.weather.kalshiapi.config.LiveTradingProperties;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.mock.http.server.reactive.MockServerHttpRequest;
import org.springframework.mock.web.server.MockServerWebExchange;
import org.springframework.web.server.WebFilterChain;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

class LocalControlTokenWebFilterTest {

  @Test
  void disabledLiveRuntimeHidesSensitiveEndpoints() {
    LiveTradingProperties live = new LiveTradingProperties();
    LocalControlTokenWebFilter filter = new LocalControlTokenWebFilter(live, new LocalControlProperties());
    MockServerWebExchange exchange = exchange("/api/live-trading/account/balance", null);

    StepVerifier.create(filter.filter(exchange, successfulChain(new AtomicBoolean()))).verifyComplete();

    assertThat(exchange.getResponse().getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
  }

  @Test
  void enabledLiveRuntimeRequiresMatchingTokenForSensitiveEndpoints() {
    LiveTradingProperties live = new LiveTradingProperties();
    live.setEnabled(true);
    LocalControlProperties control = new LocalControlProperties();
    control.setToken("test-control-token");
    LocalControlTokenWebFilter filter = new LocalControlTokenWebFilter(live, control);

    MockServerWebExchange rejected = exchange("/api/live-trading/inference/run", "wrong");
    StepVerifier.create(filter.filter(rejected, successfulChain(new AtomicBoolean()))).verifyComplete();
    assertThat(rejected.getResponse().getStatusCode()).isEqualTo(HttpStatus.UNAUTHORIZED);

    AtomicBoolean called = new AtomicBoolean();
    MockServerWebExchange accepted = exchange("/api/live-trading/inference/run", "test-control-token");
    StepVerifier.create(filter.filter(accepted, successfulChain(called))).verifyComplete();
    assertThat(called).isTrue();
  }

  @Test
  void enabledLiveRuntimeAllowsSideEffectFreeCorsPreflight() {
    LiveTradingProperties live = new LiveTradingProperties();
    live.setEnabled(true);
    LocalControlProperties control = new LocalControlProperties();
    control.setToken("test-control-token");
    LocalControlTokenWebFilter filter = new LocalControlTokenWebFilter(live, control);
    AtomicBoolean called = new AtomicBoolean();
    MockServerWebExchange exchange = MockServerWebExchange.from(
        MockServerHttpRequest.method(
                HttpMethod.OPTIONS,
                "/api/live-trading/inference/run")
            .build());

    StepVerifier.create(filter.filter(exchange, successfulChain(called))).verifyComplete();

    assertThat(called).isTrue();
  }

  private MockServerWebExchange exchange(String path, String token) {
    MockServerHttpRequest.BodyBuilder request = MockServerHttpRequest.post(path);
    if (token != null) {
      request.header(LocalControlTokenWebFilter.CONTROL_TOKEN_HEADER, token);
    }
    return MockServerWebExchange.from(request.build());
  }

  private WebFilterChain successfulChain(AtomicBoolean called) {
    return exchange -> {
      called.set(true);
      exchange.getResponse().setStatusCode(HttpStatus.OK);
      return Mono.empty();
    };
  }
}
