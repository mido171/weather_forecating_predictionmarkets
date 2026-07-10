package com.predictionmarkets.weather.kalshiapi.security;

import com.predictionmarkets.weather.kalshiapi.config.KalshiExecutionProperties;
import com.predictionmarkets.weather.kalshiapi.config.LiveTradingProperties;
import jakarta.annotation.PostConstruct;
import org.springframework.stereotype.Component;

@Component
public class KalshiRuntimeSafetyValidator {
  private final KalshiExecutionProperties executionProperties;
  private final LiveTradingProperties liveTradingProperties;
  private final LocalControlProperties localControlProperties;

  public KalshiRuntimeSafetyValidator(KalshiExecutionProperties executionProperties,
                                      LiveTradingProperties liveTradingProperties,
                                      LocalControlProperties localControlProperties) {
    this.executionProperties = executionProperties;
    this.liveTradingProperties = liveTradingProperties;
    this.localControlProperties = localControlProperties;
  }

  @PostConstruct
  public void validate() {
    if (liveTradingProperties.isInferenceInvokeEnabled() && !liveTradingProperties.isEnabled()) {
      throw new IllegalStateException(
          "kalshi.live-trading.inference-invoke-enabled requires kalshi.live-trading.enabled=true");
    }
    if (!liveTradingProperties.isEnabled()) {
      return;
    }
    if (!executionProperties.isAuthEnabled()) {
      throw new IllegalStateException(
          "kalshi.live-trading.enabled requires kalshi.auth-enabled=true");
    }
    if (!executionProperties.getWebSocket().isEnabled()) {
      throw new IllegalStateException(
          "kalshi.live-trading.enabled requires kalshi.web-socket.enabled=true");
    }
    if (!localControlProperties.hasToken()) {
      throw new IllegalStateException(
          "KALSHI_LOCAL_CONTROL_TOKEN is required when kalshi.live-trading.enabled=true");
    }
  }
}
