package com.predictionmarkets.weather.kalshiapi.auth;

import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.security.PrivateKey;
import java.security.Signature;
import java.security.spec.MGF1ParameterSpec;
import java.security.spec.PSSParameterSpec;
import java.time.Clock;
import java.util.Base64;

public class KalshiSigner {

  public static final String WS_PATH = "/trade-api/ws/v2";
  private static final PSSParameterSpec PSS_SHA256_SPEC = new PSSParameterSpec(
      "SHA-256",
      "MGF1",
      MGF1ParameterSpec.SHA256,
      32,
      1
  );

  private final PrivateKey privateKey;
  private final String apiKeyId;
  private final Clock clock;

  public KalshiSigner(PrivateKey privateKey, String apiKeyId, Clock clock) {
    if (privateKey == null) {
      throw new IllegalArgumentException("privateKey must not be null");
    }
    if (apiKeyId == null || apiKeyId.isBlank()) {
      throw new IllegalArgumentException("apiKeyId must not be blank");
    }
    if (clock == null) {
      throw new IllegalArgumentException("clock must not be null");
    }
    this.privateKey = privateKey;
    this.apiKeyId = apiKeyId;
    this.clock = clock;
  }

  public SignedHeaders sign(String httpMethodUpper, String pathWithLeadingSlash) {
    if (httpMethodUpper == null || httpMethodUpper.isBlank()) {
      throw new IllegalArgumentException("httpMethodUpper must not be blank");
    }
    if (pathWithLeadingSlash == null || pathWithLeadingSlash.isBlank()) {
      throw new IllegalArgumentException("pathWithLeadingSlash must not be blank");
    }

    String method = httpMethodUpper.toUpperCase();
    String pathWithoutQuery = stripQuery(pathWithLeadingSlash);
    String timestampMs = String.valueOf(clock.millis());
    String message = timestampMs + method + pathWithoutQuery;

    String signature = signMessage(message);
    return new SignedHeaders(apiKeyId, timestampMs, signature);
  }

  public SignedHeaders signWebSocketHandshake() {
    return sign("GET", WS_PATH);
  }

  private static String stripQuery(String pathWithLeadingSlash) {
    String pathWithoutQuery = pathWithLeadingSlash.split("\\?")[0];
    if (!pathWithoutQuery.startsWith("/")) {
      throw new IllegalArgumentException("Path must start with '/': " + pathWithLeadingSlash);
    }
    return pathWithoutQuery;
  }

  private String signMessage(String message) {
    try {
      Signature signature = Signature.getInstance("RSASSA-PSS");
      signature.setParameter(PSS_SHA256_SPEC);
      signature.initSign(privateKey);
      signature.update(message.getBytes(StandardCharsets.UTF_8));
      byte[] signed = signature.sign();
      return Base64.getEncoder().encodeToString(signed);
    } catch (GeneralSecurityException ex) {
      throw new IllegalStateException("Failed to sign Kalshi request", ex);
    }
  }
}
