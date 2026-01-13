package com.predictionmarkets.weather.repository;

import com.predictionmarkets.weather.models.GribstreamDailyFeatureEntity;
import com.predictionmarkets.weather.models.GribstreamMetric;
import java.time.Instant;
import java.time.LocalDate;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface GribstreamDailyFeatureRepository
    extends JpaRepository<GribstreamDailyFeatureEntity, Long> {

  Optional<GribstreamDailyFeatureEntity>
      findByStationIdAndTargetDateLocalAndAsofUtcAndModelCodeAndMetric(
          String stationId,
          LocalDate targetDateLocal,
          Instant asofUtc,
          String modelCode,
          GribstreamMetric metric);
}
