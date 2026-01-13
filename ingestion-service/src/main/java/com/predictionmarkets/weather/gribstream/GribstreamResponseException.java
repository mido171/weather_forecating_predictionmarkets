package com.predictionmarkets.weather.gribstream;

public class GribstreamResponseException extends RuntimeException {
  public GribstreamResponseException(String message) {
    super(message);
  }

  public GribstreamResponseException(String message, Throwable cause) {
    super(message, cause);
  }
}
