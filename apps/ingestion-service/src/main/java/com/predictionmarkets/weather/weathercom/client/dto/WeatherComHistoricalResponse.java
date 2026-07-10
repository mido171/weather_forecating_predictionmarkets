package com.predictionmarkets.weather.weathercom.client.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.ArrayList;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public class WeatherComHistoricalResponse {
  @JsonProperty("metadata")
  private WeatherComResponseMetadata metadata;

  @JsonProperty("observations")
  private List<WeatherComObservationPayload> observations = new ArrayList<>();

  public WeatherComResponseMetadata getMetadata() {
    return metadata;
  }

  public void setMetadata(WeatherComResponseMetadata metadata) {
    this.metadata = metadata;
  }

  public List<WeatherComObservationPayload> getObservations() {
    return observations;
  }

  public void setObservations(List<WeatherComObservationPayload> observations) {
    this.observations = observations;
  }
}

