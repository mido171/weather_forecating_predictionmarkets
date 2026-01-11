package com.predictionmarkets.weather.repository;

import com.predictionmarkets.weather.models.CliDaily;
import com.predictionmarkets.weather.models.CliDailyId;
import java.time.LocalDate;
import org.springframework.data.jpa.repository.JpaRepository;

public interface CliDailyRepository extends JpaRepository<CliDaily, CliDailyId> {
  long countByIdStationIdAndIdTargetDateLocalBetween(String stationId, LocalDate start, LocalDate end);

  long countByIdStationIdAndTmaxFIsNotNullAndIdTargetDateLocalBetween(
      String stationId, LocalDate start, LocalDate end);
}
