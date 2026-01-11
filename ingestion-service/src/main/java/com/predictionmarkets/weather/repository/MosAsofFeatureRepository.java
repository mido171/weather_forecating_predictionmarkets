package com.predictionmarkets.weather.repository;

import com.predictionmarkets.weather.models.MosAsofFeature;
import com.predictionmarkets.weather.models.MosAsofFeatureId;
import com.predictionmarkets.weather.models.MosModel;
import java.time.LocalDate;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface MosAsofFeatureRepository extends JpaRepository<MosAsofFeature, MosAsofFeatureId> {
  Optional<MosAsofFeature> findByIdStationIdAndIdTargetDateLocalAndIdAsofPolicyIdAndIdModel(
      String stationId,
      LocalDate targetDateLocal,
      Long asofPolicyId,
      MosModel model);
}
