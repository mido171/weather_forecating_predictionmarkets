package com.predictionmarkets.weather.repository;

import com.predictionmarkets.weather.models.ModelExperiment;
import java.util.List;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ModelExperimentRepository extends JpaRepository<ModelExperiment, Long> {
  Optional<ModelExperiment> findByExperimentKey(String experimentKey);

  List<ModelExperiment> findAllByStationIdIgnoreCase(String stationId);
}
