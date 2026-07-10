package com.predictionmarkets.weather.weathercom.service;

import com.predictionmarkets.weather.models.WeatherComObservation;
import com.predictionmarkets.weather.repository.WeatherComObservationRepository;
import com.predictionmarkets.weather.weathercom.web.dto.WeatherComObservationResponse;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class WeatherComObservationService {
  private final WeatherComObservationRepository repository;

  public WeatherComObservationService(WeatherComObservationRepository repository) {
    this.repository = repository;
  }

  @Transactional(readOnly = true)
  public Page<WeatherComObservationResponse> search(String requestLocationId,
                                                    String obsId,
                                                    Long fromValidTimeGmt,
                                                    Long toValidTimeGmt,
                                                    int page,
                                                    int size) {
    Pageable pageable = PageRequest.of(Math.max(0, page), Math.max(1, Math.min(size, 500)));
    return repository.search(
            trimToNull(requestLocationId),
            trimToNull(obsId),
            fromValidTimeGmt,
            toValidTimeGmt,
            pageable)
        .map(this::toResponse);
  }

  private WeatherComObservationResponse toResponse(WeatherComObservation entity) {
    return new WeatherComObservationResponse(
        entity.getId(),
        entity.getApiCallId(),
        entity.getRequestLocationId(),
        entity.getObsId(),
        entity.getObsName(),
        entity.getValidTimeGmt(),
        entity.getValidTimeUtc(),
        entity.getTemp(),
        entity.getDewPt(),
        entity.getRh(),
        entity.getPressure(),
        entity.getWspd(),
        entity.getWxPhrase(),
        entity.getCreatedAtUtc(),
        entity.getUpdatedAtUtc());
  }

  private String trimToNull(String value) {
    if (value == null) {
      return null;
    }
    String trimmed = value.trim();
    return trimmed.isEmpty() ? null : trimmed;
  }
}
