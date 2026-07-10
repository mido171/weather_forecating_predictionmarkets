package com.predictionmarkets.weather.klga.iemmos;

public class IemMosRateLimitException extends RuntimeException {
  public IemMosRateLimitException(String message) {
    super(message);
  }
}
