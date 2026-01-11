package com.predictionmarkets.weather.common;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

public final class Hashing {
  private Hashing() {
  }

  public static String sha256Hex(String value) {
    if (value == null) {
      throw new IllegalArgumentException("value must not be null");
    }
    MessageDigest digest = sha256Digest();
    byte[] hash = digest.digest(value.getBytes(StandardCharsets.UTF_8));
    return HexFormat.of().formatHex(hash);
  }

  private static MessageDigest sha256Digest() {
    try {
      return MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException ex) {
      throw new IllegalStateException("SHA-256 not available", ex);
    }
  }
}
