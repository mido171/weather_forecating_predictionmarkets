package com.predictionmarkets.weather.kalshiapi.live;

import java.time.LocalDate;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@CrossOrigin(origins = "*")
@RequestMapping("/api/live-trading/inference")
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
