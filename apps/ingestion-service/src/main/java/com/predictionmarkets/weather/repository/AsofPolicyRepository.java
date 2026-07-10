package com.predictionmarkets.weather.repository;

import com.predictionmarkets.weather.models.AsofPolicy;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface AsofPolicyRepository extends JpaRepository<AsofPolicy, Long> {
  Optional<AsofPolicy> findByName(String name);
}
