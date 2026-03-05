package com.predictionmarkets.weather.kalshiapi.live;

import java.time.LocalDate;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestParam;

@RestController
@CrossOrigin(origins = "*")
@RequestMapping("/api/live-trading/orderbooks")
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
