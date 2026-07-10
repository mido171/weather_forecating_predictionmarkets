package com.predictionmarkets.weather.kalshiapi.live;

import java.time.LocalDate;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestParam;

@RestController
@CrossOrigin(origins = {"http://127.0.0.1:5173", "http://localhost:5173"})
@RequestMapping("/api/live-trading/orderbooks")
@ConditionalOnProperty(prefix = "kalshi.live-trading", name = "enabled", havingValue = "true")
public class LiveOrderbookController {

  private final LiveOrderbookStreamService streamService;

  public LiveOrderbookController(LiveOrderbookStreamService streamService) {
    this.streamService = streamService;
  }

  @GetMapping("/snapshot")
  public LiveOrderbookFrame snapshot(
      @RequestParam(name = "targetDateLocal", required = false)
      @DateTimeFormat(iso = DateTimeFormat.ISO.DATE)
      LocalDate targetDateLocal) {
    return streamService.currentSnapshot(targetDateLocal);
  }
}
