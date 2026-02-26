package com.predictionmarkets.weather.weathercom.service;

import com.predictionmarkets.weather.models.WeatherComLocation;
import com.predictionmarkets.weather.repository.WeatherComLocationRepository;
import java.time.Instant;
import java.util.List;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class WeatherComLocationService {
  private final WeatherComLocationRepository repository;

  public WeatherComLocationService(WeatherComLocationRepository repository) {
    this.repository = repository;
  }

  @Transactional
  public WeatherComLocation create(String locationId, String displayName, Boolean active) {
    String normalizedLocationId = WeatherComValidation.normalizeLocationId(locationId);
    repository.findByLocationIdIgnoreCase(normalizedLocationId).ifPresent(existing -> {
      throw new IllegalArgumentException("weathercom_location already exists for " + normalizedLocationId);
    });

    Instant now = Instant.now();
    WeatherComLocation entity = new WeatherComLocation();
    entity.setLocationId(normalizedLocationId);
    entity.setDisplayName(trimToNull(displayName));
    entity.setActive(active == null || active);
    entity.setCreatedAtUtc(now);
    entity.setUpdatedAtUtc(now);
    return repository.save(entity);
  }

  @Transactional(readOnly = true)
  public Page<WeatherComLocation> list(int page, int size) {
    Pageable pageable = PageRequest.of(
        Math.max(0, page),
        Math.max(1, Math.min(size, 500)),
        Sort.by(Sort.Direction.ASC, "id"));
    return repository.findAll(pageable);
  }

  @Transactional(readOnly = true)
  public WeatherComLocation get(Long id) {
    return repository.findById(id)
        .orElseThrow(() -> new WeatherComNotFoundException("weathercom_location not found: " + id));
  }

  @Transactional
  public WeatherComLocation update(Long id, String locationId, String displayName, Boolean active) {
    WeatherComLocation entity = get(id);
    String normalizedLocationId = WeatherComValidation.normalizeLocationId(locationId);
    repository.findByLocationIdIgnoreCase(normalizedLocationId)
        .filter(existing -> !existing.getId().equals(id))
        .ifPresent(existing -> {
          throw new IllegalArgumentException(
              "weathercom_location already exists for " + normalizedLocationId);
        });

    entity.setLocationId(normalizedLocationId);
    entity.setDisplayName(trimToNull(displayName));
    if (active != null) {
      entity.setActive(active);
    }
    entity.setUpdatedAtUtc(Instant.now());
    return repository.save(entity);
  }

  @Transactional
  public void delete(Long id) {
    if (!repository.existsById(id)) {
      throw new WeatherComNotFoundException("weathercom_location not found: " + id);
    }
    repository.deleteById(id);
  }

  @Transactional(readOnly = true)
  public List<String> findAllActiveLocationIds() {
    return repository.findAllByActiveTrueOrderByLocationIdAsc().stream()
        .map(WeatherComLocation::getLocationId)
        .distinct()
        .toList();
  }

  private String trimToNull(String value) {
    if (value == null) {
      return null;
    }
    String trimmed = value.trim();
    return trimmed.isEmpty() ? null : trimmed;
  }
}
