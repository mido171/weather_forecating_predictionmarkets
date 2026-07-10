package com.predictionmarkets.weather.gribstream;

public class GribstreamEmptyResponseException extends GribstreamResponseException {
  public GribstreamEmptyResponseException(String message) {
    super(message);
  }
}
