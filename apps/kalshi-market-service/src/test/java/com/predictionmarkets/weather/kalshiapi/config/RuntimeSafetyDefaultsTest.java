package com.predictionmarkets.weather.kalshiapi.config;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class RuntimeSafetyDefaultsTest {

  @Test
  void allLiveCapabilitiesAreDisabledAndBoundedByDefault() {
    KalshiExecutionProperties execution = new KalshiExecutionProperties();
    LiveTradingProperties live = new LiveTradingProperties();
    BacktestGridProperties backtest = new BacktestGridProperties();

    execution.validateConfiguration();

    assertThat(execution.getEnvironment()).isEqualTo(KalshiEnvironment.DEMO);
    assertThat(execution.isProductionAcknowledged()).isFalse();
    assertThat(execution.isAuthEnabled()).isFalse();
    assertThat(execution.isTradingEnabled()).isFalse();
    assertThat(execution.getWebSocket().isEnabled()).isFalse();
    assertThat(execution.getWebSocket().getMaxReconnectAttempts()).isOne();
    assertThat(execution.getRetry().getMaxRetries()).isOne();
    assertThat(execution.getRateLimiting().getMaxReadRequestsPerSecond()).isOne();
    assertThat(execution.getRateLimiting().getMaxWriteRequestsPerSecond()).isOne();
    assertThat(execution.getGuardrails().isStartupReconcile()).isFalse();
    assertThat(execution.getSmoke().isEnabled()).isFalse();
    assertThat(live.isEnabled()).isFalse();
    assertThat(live.isInferenceInvokeEnabled()).isFalse();
    assertThat(backtest.isEnabled()).isFalse();
    assertThat(backtest.getThreadCount()).isOne();
    assertThat(backtest.isOverwriteSqlite()).isFalse();
  }

  @Test
  void authenticatedProductionAccessRequiresExplicitAcknowledgement() {
    KalshiExecutionProperties execution = new KalshiExecutionProperties();
    execution.setEnvironment(KalshiEnvironment.PROD);
    execution.setAuthEnabled(true);

    assertThatThrownBy(execution::validateConfiguration)
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("KALSHI_PRODUCTION_ACKNOWLEDGED");
  }

  @Test
  void tradingAndWebSocketCannotBypassAuthenticationAndGuardrails() {
    KalshiExecutionProperties trading = new KalshiExecutionProperties();
    trading.setTradingEnabled(true);
    assertThatThrownBy(trading::validateConfiguration)
        .hasMessageContaining("trading-enabled requires kalshi.auth-enabled");

    KalshiExecutionProperties webSocket = new KalshiExecutionProperties();
    webSocket.getWebSocket().setEnabled(true);
    assertThatThrownBy(webSocket::validateConfiguration)
        .hasMessageContaining("web-socket.enabled requires kalshi.auth-enabled");
  }
}
