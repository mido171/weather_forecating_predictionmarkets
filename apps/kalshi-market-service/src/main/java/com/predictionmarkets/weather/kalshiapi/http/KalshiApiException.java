package com.predictionmarkets.weather.kalshiapi.http;

import com.predictionmarkets.weather.kalshiapi.model.ApiError;
import org.springframework.http.HttpStatusCode;

public class KalshiApiException extends RuntimeException {

  private final int statusCode;
  private final String responseBody;
  private final ApiError apiError;

  public KalshiApiException(HttpStatusCode statusCode, String responseBody, ApiError apiError) {
    super(buildMessage(statusCode, responseBody, apiError));
    this.statusCode = statusCode.value();
    this.responseBody = responseBody;
    this.apiError = apiError;
  }

  private static String buildMessage(HttpStatusCode statusCode, String responseBody, ApiError apiError) {
    if (apiError != null && apiError.resolvedMessage() != null) {
      return "Kalshi API error " + statusCode.value() + ": " + apiError.resolvedMessage();
    }
    return "Kalshi API error " + statusCode.value() + ": " + responseBody;
  }

  public int getStatusCode() {
    return statusCode;
  }

  public String getResponseBody() {
    return responseBody;
  }

  public ApiError getApiError() {
    return apiError;
  }
}
