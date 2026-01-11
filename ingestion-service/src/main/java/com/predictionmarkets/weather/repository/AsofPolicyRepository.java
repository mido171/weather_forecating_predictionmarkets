package com.predictionmarkets.weather.repository;

import com.predictionmarkets.weather.models.AsofPolicy;
import org.springframework.data.jpa.repository.JpaRepository;

public interface AsofPolicyRepository extends JpaRepository<AsofPolicy, Long> {
}
