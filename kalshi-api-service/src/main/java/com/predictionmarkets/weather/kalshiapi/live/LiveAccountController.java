package com.predictionmarkets.weather.kalshiapi.live;

import com.predictionmarkets.weather.kalshiapi.service.KalshiTradingService;
import com.predictionmarkets.weather.kalshiapi.trading.KalshiAccountBalance;
import com.predictionmarkets.weather.kalshiapi.trading.KalshiAccountSnapshot;
import com.predictionmarkets.weather.kalshiapi.trading.KalshiPositionExposure;
import java.util.List;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

@RestController
@CrossOrigin(origins = "*")
@RequestMapping("/api/live-trading/account")
public class LiveAccountController {

  private final KalshiTradingService tradingService;

  public LiveAccountController(KalshiTradingService tradingService) {
    this.tradingService = tradingService;
  }

  @GetMapping("/balance")
  public Mono<KalshiAccountBalance> balance() {
    return Mono.fromCallable(tradingService::getAccountBalance)
        .subscribeOn(Schedulers.boundedElastic());
  }

  @GetMapping("/positions")
  public Mono<List<KalshiPositionExposure>> positions() {
    return Mono.fromCallable(tradingService::getOpenPositionsWithExposure)
        .subscribeOn(Schedulers.boundedElastic());
  }

  @GetMapping("/snapshot")
  public Mono<KalshiAccountSnapshot> snapshot() {
    return Mono.fromCallable(tradingService::getAccountSnapshot)
        .subscribeOn(Schedulers.boundedElastic());
  }
}
