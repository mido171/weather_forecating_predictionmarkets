package com.predictionmarkets.weather.repository;

import com.predictionmarkets.weather.models.IngestCheckpoint;
import com.predictionmarkets.weather.models.MosModel;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface IngestCheckpointRepository extends JpaRepository<IngestCheckpoint, Long> {
  Optional<IngestCheckpoint> findByJobNameAndStationIdAndModel(String jobName,
                                                               String stationId,
                                                               MosModel model);

  Optional<IngestCheckpoint> findByJobNameAndStationIdAndModelIsNull(String jobName,
                                                                     String stationId);
}
