package com.predictionmarkets.weather.repository;

import com.predictionmarkets.weather.models.MosDailyValue;
import java.time.Instant;
import java.time.LocalDate;
import java.util.List;
import java.util.Optional;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface MosDailyValueRepository extends JpaRepository<MosDailyValue, Long> {
  long countByStationIdAndModel(String stationId, String model);

  List<MosDailyValue> findByStationIdAndModelOrderByRuntimeUtcAscTargetDateLocalAscVariableCodeAsc(
      String stationId, String model);

  Optional<MosDailyValue>
      findFirstByStationIdAndModelAndTargetDateLocalAndVariableCodeAndRuntimeUtcLessThanEqualOrderByRuntimeUtcDesc(
          String stationId,
          String model,
          LocalDate targetDateLocal,
          String variableCode,
          Instant runtimeUtcMax);

  Page<MosDailyValue> findByStationIdAndModelIn(String stationId, List<String> models,
                                                Pageable pageable);

  Page<MosDailyValue> findByModelIn(List<String> models, Pageable pageable);

  @Query("select distinct m.variableCode from MosDailyValue m where m.model in :models "
      + "order by m.variableCode")
  List<String> findDistinctVariableCodesByModelIn(@Param("models") List<String> models);
}
