package com.predictionmarkets.weather.kalshiapi.config;

public enum KalshiEnvironment {
  DEMO(
      "https://demo-api.kalshi.co/trade-api/v2",
      "wss://demo-api.kalshi.co/trade-api/ws/v2"
  ),
  PROD(
      "https://api.elections.kalshi.com/trade-api/v2",
      "wss://api.elections.kalshi.com/trade-api/ws/v2"
  );

  private final String restBaseUrl;
  private final String wsUrl;

  KalshiEnvironment(String restBaseUrl, String wsUrl) {
    this.restBaseUrl = restBaseUrl;
    this.wsUrl = wsUrl;
  }

  public String restBaseUrl() {
    return restBaseUrl;
  }

  public String wsUrl() {
    return wsUrl;
  }
}
