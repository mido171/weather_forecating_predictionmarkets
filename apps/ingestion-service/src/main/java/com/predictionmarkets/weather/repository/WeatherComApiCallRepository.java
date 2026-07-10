package com.predictionmarkets.weather.repository;

import com.predictionmarkets.weather.models.WeatherComApiCall;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface WeatherComApiCallRepository extends JpaRepository<WeatherComApiCall, Long> {
  Page<WeatherComApiCall> findByIngestionRunIdOrderByIdDesc(Long ingestionRunId, Pageable pageable);

  @Query("""
      select c
        from WeatherComApiCall c
       where c.ingestionRun.id = :runId
         and (c.httpStatus >= 400 or c.errorType is not null)
       order by c.id desc
      """)
  Page<WeatherComApiCall> findFailedByRunId(@Param("runId") Long runId, Pageable pageable);

  @Query("""
      select c
        from WeatherComApiCall c
       where c.ingestionRun.id = :runId
         and c.httpStatus < 400
         and c.errorType is null
       order by c.id desc
      """)
  Page<WeatherComApiCall> findSucceededByRunId(@Param("runId") Long runId, Pageable pageable);
}

