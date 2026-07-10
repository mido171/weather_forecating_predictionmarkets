package com.predictionmarkets.weather.kalshiapi.security;

import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.predictionmarkets.weather.kalshiapi.config.KalshiExecutionProperties;
import com.predictionmarkets.weather.kalshiapi.config.LiveTradingProperties;
import org.junit.jupiter.api.Test;

class KalshiRuntimeSafetyValidatorTest {

  @Test
  void disabledRuntimeIsValidWithoutCredentialsOrControlToken() {
    KalshiRuntimeSafetyValidator validator = validator(
        new KalshiExecutionProperties(), new LiveTradingProperties(), new LocalControlProperties());

    assertThatCode(validator::validate).doesNotThrowAnyException();
  }

  @Test
  void inferenceInvocationCannotBeEnabledOutsideLiveRuntime() {
    LiveTradingProperties live = new LiveTradingProperties();
    live.setInferenceInvokeEnabled(true);

    assertThatThrownBy(validator(
        new KalshiExecutionProperties(), live, new LocalControlProperties())::validate)
        .hasMessageContaining("inference-invoke-enabled requires");
  }

  @Test
  void liveRuntimeRequiresAuthWebSocketAndLocalControlToken() {
    KalshiExecutionProperties execution = new KalshiExecutionProperties();
    LiveTradingProperties live = new LiveTradingProperties();
    LocalControlProperties localControl = new LocalControlProperties();
    live.setEnabled(true);

    assertThatThrownBy(validator(execution, live, localControl)::validate)
        .hasMessageContaining("auth-enabled");

    execution.setAuthEnabled(true);
    assertThatThrownBy(validator(execution, live, localControl)::validate)
        .hasMessageContaining("web-socket.enabled");

    execution.getWebSocket().setEnabled(true);
    assertThatThrownBy(validator(execution, live, localControl)::validate)
        .hasMessageContaining("KALSHI_LOCAL_CONTROL_TOKEN");

    localControl.setToken("test-control-token");
    assertThatCode(validator(execution, live, localControl)::validate).doesNotThrowAnyException();
  }

  private KalshiRuntimeSafetyValidator validator(KalshiExecutionProperties execution,
                                                  LiveTradingProperties live,
                                                  LocalControlProperties localControl) {
    return new KalshiRuntimeSafetyValidator(execution, live, localControl);
  }
}
