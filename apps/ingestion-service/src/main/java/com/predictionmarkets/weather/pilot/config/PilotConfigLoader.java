package com.predictionmarkets.weather.pilot.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.springframework.stereotype.Service;
import org.yaml.snakeyaml.Yaml;

@Service
public class PilotConfigLoader {
  private final PilotIngestionProperties properties;
  private final ObjectMapper objectMapper;

  public PilotConfigLoader(PilotIngestionProperties properties, ObjectMapper objectMapper) {
    this.properties = properties;
    this.objectMapper = objectMapper;
  }

  public List<StationConfig> loadStations() {
    return readList("stations.yml", "stations", StationConfig.class);
  }

  public List<SourceConfig> loadSources() {
    return readList("sources.yml", "sources", SourceConfig.class);
  }

  public List<JobConfig> loadJobs() {
    return readList("jobs.yml", "jobs", JobConfig.class);
  }

  public StationConfig requireDefaultStation() {
    return loadStations().stream()
        .filter(station -> properties.getDefaultStationKey().equals(station.getStationKey()))
        .findFirst()
        .orElseThrow(() -> new IllegalStateException(
            "Default station not found: " + properties.getDefaultStationKey()));
  }

  private <T> List<T> readList(String fileName, String listKey, Class<T> itemType) {
    Path path = Path.of(properties.getConfigDir(), fileName);
    if (!Files.exists(path)) {
      throw new IllegalStateException("Missing pilot config file: " + path);
    }
    Yaml yaml = new Yaml();
    try (InputStream inputStream = Files.newInputStream(path)) {
      Object raw = yaml.load(inputStream);
      if (!(raw instanceof Map<?, ?> rawMap)) {
        return List.of();
      }
      Object values = rawMap.get(listKey);
      if (!(values instanceof List<?> rawList)) {
        return List.of();
      }
      List<T> converted = new ArrayList<>(rawList.size());
      for (Object value : rawList) {
        converted.add(objectMapper.convertValue(value, itemType));
      }
      return List.copyOf(converted);
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to load pilot config file: " + path, ex);
    }
  }
}
