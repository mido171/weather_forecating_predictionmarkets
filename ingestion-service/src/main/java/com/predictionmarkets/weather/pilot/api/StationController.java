package com.predictionmarkets.weather.pilot.api;

import com.predictionmarkets.weather.pilot.catalog.SqliteCatalogService;
import java.util.List;
import java.util.Map;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/internal/ingest/stations")
public class StationController {
  private final SqliteCatalogService catalogService;

  public StationController(SqliteCatalogService catalogService) {
    this.catalogService = catalogService;
  }

  @GetMapping
  public List<Map<String, Object>> stations() {
    return catalogService.query("""
        SELECT station_key, display_name, timezone, latitude, longitude, elevation_m,
               metar_reset_minute, created_at_utc, updated_at_utc
        FROM station_registry
        ORDER BY station_key
        """);
  }

  @GetMapping("/{stationKey}")
  public Map<String, Object> station(@PathVariable String stationKey) {
    return catalogService.querySingle("""
        SELECT *
        FROM station_registry
        WHERE station_key = ?
        """, stationKey);
  }
}
