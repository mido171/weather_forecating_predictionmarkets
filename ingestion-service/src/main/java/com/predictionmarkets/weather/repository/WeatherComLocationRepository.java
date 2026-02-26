package com.predictionmarkets.weather.repository;

import com.predictionmarkets.weather.models.WeatherComLocation;
import java.util.List;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface WeatherComLocationRepository extends JpaRepository<WeatherComLocation, Long> {
  Optional<WeatherComLocation> findByLocationIdIgnoreCase(String locationId);

  List<WeatherComLocation> findAllByActiveTrueOrderByLocationIdAsc();
}
