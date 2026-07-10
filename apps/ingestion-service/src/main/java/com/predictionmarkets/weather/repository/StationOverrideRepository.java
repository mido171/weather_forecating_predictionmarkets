package com.predictionmarkets.weather.repository;

import com.predictionmarkets.weather.models.StationOverride;
import org.springframework.data.jpa.repository.JpaRepository;

public interface StationOverrideRepository extends JpaRepository<StationOverride, String> {
}
