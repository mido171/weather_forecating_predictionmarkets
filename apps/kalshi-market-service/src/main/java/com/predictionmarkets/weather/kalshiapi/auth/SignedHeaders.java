package com.predictionmarkets.weather.kalshiapi.auth;

import org.springframework.http.HttpHeaders;

public record SignedHeaders(String accessKey, String accessTimestamp, String accessSignature) {
  public void apply(HttpHeaders headers) {
    headers.set(KalshiAuthHeaders.ACCESS_KEY, accessKey);
    headers.set(KalshiAuthHeaders.ACCESS_TIMESTAMP, accessTimestamp);
    headers.set(KalshiAuthHeaders.ACCESS_SIGNATURE, accessSignature);
  }
}
