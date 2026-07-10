package com.predictionmarkets.weather.repository;

import com.predictionmarkets.weather.models.MosModel;
import com.predictionmarkets.weather.models.MosRun;
import com.predictionmarkets.weather.models.MosRunId;
import java.time.Instant;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface MosRunRepository extends JpaRepository<MosRun, MosRunId> {
  long countByIdStationIdAndIdModel(String stationId, MosModel model);

  Optional<MosRun> findTopByIdStationIdAndIdModelAndIdRuntimeUtcLessThanEqualOrderByIdRuntimeUtcDesc(
      String stationId, MosModel model, Instant runtimeUtc);
}
