package com.predictionmarkets.weather.kalshiapi.http;

public class TradingDisabledException extends RuntimeException {
  public TradingDisabledException(String message) {
    super(message);
  }
}
