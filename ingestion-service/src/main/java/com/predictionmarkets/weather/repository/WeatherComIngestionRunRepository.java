package com.predictionmarkets.weather.repository;

import com.predictionmarkets.weather.models.WeatherComIngestionRun;
import com.predictionmarkets.weather.models.WeatherComIngestionStatus;
import java.time.Instant;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.transaction.annotation.Transactional;

public interface WeatherComIngestionRunRepository extends JpaRepository<WeatherComIngestionRun, Long> {
  @Modifying
  @Transactional
  @Query("""
      update WeatherComIngestionRun r
         set r.succeededTasks = r.succeededTasks + :delta,
             r.updatedAtUtc = :updatedAtUtc
       where r.id = :runId
      """)
  int incrementSucceededTasks(@Param("runId") Long runId,
                              @Param("delta") int delta,
                              @Param("updatedAtUtc") Instant updatedAtUtc);

  @Modifying
  @Transactional
  @Query("""
      update WeatherComIngestionRun r
         set r.failedTasks = r.failedTasks + :delta,
             r.updatedAtUtc = :updatedAtUtc
       where r.id = :runId
      """)
  int incrementFailedTasks(@Param("runId") Long runId,
                           @Param("delta") int delta,
                           @Param("updatedAtUtc") Instant updatedAtUtc);

  @Modifying
  @Transactional
  @Query("""
      update WeatherComIngestionRun r
         set r.status = :status,
             r.finishedAtUtc = :finishedAtUtc,
             r.updatedAtUtc = :updatedAtUtc
       where r.id = :runId
      """)
  int markFinished(@Param("runId") Long runId,
                   @Param("status") WeatherComIngestionStatus status,
                   @Param("finishedAtUtc") Instant finishedAtUtc,
                   @Param("updatedAtUtc") Instant updatedAtUtc);
}
