package com.predictionmarkets.weather.repository;

import com.predictionmarkets.weather.models.StationRegistry;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface StationRegistryRepository extends JpaRepository<StationRegistry, String> {
  Optional<StationRegistry> findBySeriesTicker(String seriesTicker);
}
