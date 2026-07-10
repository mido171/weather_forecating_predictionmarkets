package com.predictionmarkets.weather.kalshiapi;

import static org.assertj.core.api.Assertions.assertThat;

import com.predictionmarkets.weather.kalshiapi.backtest.MosBacktestGridRunner;
import com.predictionmarkets.weather.kalshiapi.executor.KalshiExecutionSmokeRunner;
import com.predictionmarkets.weather.kalshiapi.live.LiveAccountController;
import com.predictionmarkets.weather.kalshiapi.live.LiveInferenceInvokeService;
import com.predictionmarkets.weather.kalshiapi.live.LiveInferenceController;
import com.predictionmarkets.weather.kalshiapi.live.LiveOrderbookController;
import com.predictionmarkets.weather.kalshiapi.live.LiveOrderbookStreamService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.ApplicationContext;

@SpringBootTest(properties = "spring.main.web-application-type=none")
class KalshiApiServiceApplicationTests {
  @Autowired
  private ApplicationContext applicationContext;

  @Test
  void failClosedContextLoadsWithoutLiveBeans() {
    assertThat(applicationContext.getBeansOfType(LiveAccountController.class)).isEmpty();
    assertThat(applicationContext.getBeansOfType(LiveInferenceController.class)).isEmpty();
    assertThat(applicationContext.getBeansOfType(LiveOrderbookController.class)).isEmpty();
    assertThat(applicationContext.getBeansOfType(LiveOrderbookStreamService.class)).isEmpty();
    assertThat(applicationContext.getBeansOfType(LiveInferenceInvokeService.class)).isEmpty();
    assertThat(applicationContext.getBeansOfType(KalshiExecutionSmokeRunner.class)).isEmpty();
    assertThat(applicationContext.getBeansOfType(MosBacktestGridRunner.class)).isEmpty();
  }
}
