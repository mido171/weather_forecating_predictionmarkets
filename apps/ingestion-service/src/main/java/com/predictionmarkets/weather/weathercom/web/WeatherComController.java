package com.predictionmarkets.weather.weathercom.web;

import com.predictionmarkets.weather.models.WeatherComApiCall;
import com.predictionmarkets.weather.models.WeatherComIngestionRun;
import com.predictionmarkets.weather.models.WeatherComLocation;
import com.predictionmarkets.weather.weathercom.service.WeatherComIngestionService;
import com.predictionmarkets.weather.weathercom.service.WeatherComLocationService;
import com.predictionmarkets.weather.weathercom.service.WeatherComObservationService;
import com.predictionmarkets.weather.weathercom.web.dto.WeatherComApiCallResponse;
import com.predictionmarkets.weather.weathercom.web.dto.WeatherComApiCallStatusFilter;
import com.predictionmarkets.weather.weathercom.web.dto.WeatherComIngestionRunResponse;
import com.predictionmarkets.weather.weathercom.web.dto.WeatherComIngestionTriggerRequest;
import com.predictionmarkets.weather.weathercom.web.dto.WeatherComIngestionTriggerResponse;
import com.predictionmarkets.weather.weathercom.web.dto.WeatherComLocationRequest;
import com.predictionmarkets.weather.weathercom.web.dto.WeatherComLocationResponse;
import com.predictionmarkets.weather.weathercom.web.dto.WeatherComObservationResponse;
import jakarta.validation.Valid;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import org.springframework.data.domain.Page;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.HttpStatus;
import org.springframework.validation.annotation.Validated;
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
@Validated
@RequestMapping("/api/weathercom")
@ConditionalOnProperty(prefix = "ingestion.admin-api", name = "enabled", havingValue = "true")
public class WeatherComController {
  private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.BASIC_ISO_DATE;

  private final WeatherComLocationService locationService;
  private final WeatherComObservationService observationService;
  private final WeatherComIngestionService ingestionService;

  public WeatherComController(WeatherComLocationService locationService,
                              WeatherComObservationService observationService,
                              WeatherComIngestionService ingestionService) {
    this.locationService = locationService;
    this.observationService = observationService;
    this.ingestionService = ingestionService;
  }

  @PostMapping("/locations")
  @ResponseStatus(HttpStatus.CREATED)
  public WeatherComLocationResponse createLocation(@Valid @RequestBody WeatherComLocationRequest request) {
    WeatherComLocation location = locationService.create(
        request.locationId(),
        request.displayName(),
        request.active());
    return toLocationResponse(location);
  }

  @GetMapping("/locations")
  public Page<WeatherComLocationResponse> listLocations(
      @RequestParam(value = "page", defaultValue = "0") int page,
      @RequestParam(value = "size", defaultValue = "50") int size) {
    return locationService.list(page, size).map(this::toLocationResponse);
  }

  @GetMapping("/locations/{id}")
  public WeatherComLocationResponse getLocation(@PathVariable("id") Long id) {
    return toLocationResponse(locationService.get(id));
  }

  @PutMapping("/locations/{id}")
  public WeatherComLocationResponse updateLocation(@PathVariable("id") Long id,
                                                   @Valid @RequestBody WeatherComLocationRequest request) {
    WeatherComLocation updated = locationService.update(
        id,
        request.locationId(),
        request.displayName(),
        request.active());
    return toLocationResponse(updated);
  }

  @DeleteMapping("/locations/{id}")
  @ResponseStatus(HttpStatus.NO_CONTENT)
  public void deleteLocation(@PathVariable("id") Long id) {
    locationService.delete(id);
  }

  @GetMapping("/observations")
  public Page<WeatherComObservationResponse> listObservations(
      @RequestParam(value = "requestLocationId", required = false) String requestLocationId,
      @RequestParam(value = "obsId", required = false) String obsId,
      @RequestParam(value = "fromValidTimeGmt", required = false) Long fromValidTimeGmt,
      @RequestParam(value = "toValidTimeGmt", required = false) Long toValidTimeGmt,
      @RequestParam(value = "page", defaultValue = "0") int page,
      @RequestParam(value = "size", defaultValue = "100") int size) {
    return observationService.search(
        requestLocationId,
        obsId,
        fromValidTimeGmt,
        toValidTimeGmt,
        page,
        size);
  }

  @PostMapping("/ingestions")
  @ResponseStatus(HttpStatus.ACCEPTED)
  public WeatherComIngestionTriggerResponse triggerIngestion(
      @Valid @RequestBody WeatherComIngestionTriggerRequest request) {
    LocalDate startDate = LocalDate.parse(request.startDate(), DATE_FORMATTER);
    LocalDate endDate = LocalDate.parse(request.endDate(), DATE_FORMATTER);
    WeatherComIngestionRun run = ingestionService.triggerIngestion(
        request.locationIds(),
        startDate,
        endDate,
        request.units(),
        request.requestedBy());
    return new WeatherComIngestionTriggerResponse(run.getId(), run.getStatus().name());
  }

  @GetMapping("/ingestions/{runId}")
  public WeatherComIngestionRunResponse getIngestionRun(@PathVariable("runId") Long runId) {
    return toRunResponse(ingestionService.getRun(runId));
  }

  @GetMapping("/ingestions/{runId}/api-calls")
  public Page<WeatherComApiCallResponse> listRunApiCalls(
      @PathVariable("runId") Long runId,
      @RequestParam(value = "status", required = false, defaultValue = "ALL") String status,
      @RequestParam(value = "page", defaultValue = "0") int page,
      @RequestParam(value = "size", defaultValue = "100") int size) {
    WeatherComApiCallStatusFilter filter = WeatherComApiCallStatusFilter.parse(status);
    return ingestionService.listApiCalls(runId, filter, page, size)
        .map(this::toApiCallResponse);
  }

  private WeatherComLocationResponse toLocationResponse(WeatherComLocation location) {
    return new WeatherComLocationResponse(
        location.getId(),
        location.getLocationId(),
        location.getDisplayName(),
        location.isActive(),
        location.getCreatedAtUtc(),
        location.getUpdatedAtUtc());
  }

  private WeatherComIngestionRunResponse toRunResponse(WeatherComIngestionRun run) {
    return new WeatherComIngestionRunResponse(
        run.getId(),
        run.getStatus().name(),
        run.getStartedAtUtc(),
        run.getFinishedAtUtc(),
        run.getRequestedBy(),
        run.getTotalTasks(),
        run.getSucceededTasks(),
        run.getFailedTasks(),
        run.getCreatedAtUtc(),
        run.getUpdatedAtUtc());
  }

  private WeatherComApiCallResponse toApiCallResponse(WeatherComApiCall apiCall) {
    return new WeatherComApiCallResponse(
        apiCall.getId(),
        apiCall.getIngestionRun() == null ? null : apiCall.getIngestionRun().getId(),
        apiCall.getRequestLocationId(),
        apiCall.getUnits(),
        apiCall.getStartDate(),
        apiCall.getEndDate(),
        apiCall.getHttpStatus(),
        apiCall.getErrorType(),
        apiCall.getErrorMessage(),
        apiCall.getDurationMs(),
        apiCall.getFetchedAtUtc(),
        apiCall.getResponseLocationId(),
        apiCall.getResponseUnits(),
        apiCall.getResponseLanguage(),
        apiCall.getTransactionId(),
        apiCall.getApiVersion(),
        apiCall.getExpireTimeGmt());
  }
}

