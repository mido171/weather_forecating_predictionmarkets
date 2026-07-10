package com.predictionmarkets.weather.repository;

import com.predictionmarkets.weather.models.KalshiSeries;
import org.springframework.data.jpa.repository.JpaRepository;

public interface KalshiSeriesRepository extends JpaRepository<KalshiSeries, String> {
}
