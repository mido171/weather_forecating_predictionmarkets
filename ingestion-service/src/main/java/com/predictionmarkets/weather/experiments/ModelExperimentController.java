package com.predictionmarkets.weather.experiments;

import java.util.List;
import java.util.Optional;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/model-experiments")
public class ModelExperimentController {
  private final ModelExperimentService service;

  public ModelExperimentController(ModelExperimentService service) {
    this.service = service;
  }

  @PostMapping
  @ResponseStatus(HttpStatus.CREATED)
  public ModelExperimentResponse create(@RequestBody ModelExperimentRequest request) {
    return service.createOrUpdateByKey(request);
  }

  @GetMapping("/{id}")
  public ModelExperimentResponse get(@PathVariable("id") Long id) {
    return service.get(id);
  }

  @GetMapping
  public List<ModelExperimentResponse> list(
      @RequestParam(value = "stationId", required = false) String stationId,
      @RequestParam(value = "modelFamily", required = false) String modelFamily,
      @RequestParam(value = "limit", required = false, defaultValue = "0") int limit) {
    return service.list(stationId, modelFamily, limit);
  }

  @GetMapping("/by-key")
  public Optional<ModelExperimentResponse> getByKey(
      @RequestParam("experimentKey") String experimentKey) {
    return service.getByKey(experimentKey);
  }

  @PutMapping("/{id}")
  public ModelExperimentResponse update(@PathVariable("id") Long id,
                                        @RequestBody ModelExperimentRequest request) {
    return service.update(id, request);
  }

  @DeleteMapping("/{id}")
  @ResponseStatus(HttpStatus.NO_CONTENT)
  public void delete(@PathVariable("id") Long id) {
    service.delete(id);
  }
}
