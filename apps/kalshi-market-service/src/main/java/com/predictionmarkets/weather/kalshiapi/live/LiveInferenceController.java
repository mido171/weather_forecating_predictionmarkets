package com.predictionmarkets.weather.kalshiapi.live;

import java.time.LocalDate;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@CrossOrigin(
    origins = {"http://127.0.0.1:5173", "http://localhost:5173"},
    allowedHeaders = {"Content-Type", "X-Local-Control-Token"})
@RequestMapping("/api/live-trading/inference")
@ConditionalOnProperty(prefix = "kalshi.live-trading", name = "enabled", havingValue = "true")
public class LiveInferenceController {

  private final LiveInferenceInvokeService invokeService;

  public LiveInferenceController(LiveInferenceInvokeService invokeService) {
    this.invokeService = invokeService;
  }

  @PostMapping("/run")
  public LiveInferenceRunResponse runInference(
      @RequestParam(name = "targetDateLocal")
      @DateTimeFormat(iso = DateTimeFormat.ISO.DATE)
      LocalDate targetDateLocal) {
    return invokeService.invokeForDate(targetDateLocal);
  }
}
