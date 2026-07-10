package com.predictionmarkets.weather.kalshiapi.auth;

import java.security.PrivateKey;

public record KalshiCredentials(String apiKeyId, PrivateKey privateKey) {
}
