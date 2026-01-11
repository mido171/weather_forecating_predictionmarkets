package com.predictionmarkets.weather.repository;

import com.predictionmarkets.weather.models.StationRegistry;
import org.springframework.data.jpa.repository.JpaRepository;

public interface StationRegistryRepository extends JpaRepository<StationRegistry, String> {
}
