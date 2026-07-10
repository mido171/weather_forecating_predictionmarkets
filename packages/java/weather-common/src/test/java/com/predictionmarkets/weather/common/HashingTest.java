package com.predictionmarkets.weather.common;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

class HashingTest {
  @Test
  void sha256Hex_isStableForBytes() {
    byte[] bytes = "hello".getBytes(StandardCharsets.UTF_8);
    String expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

    assertEquals(expected, Hashing.sha256Hex(bytes));
    assertEquals(expected, Hashing.sha256Hex("hello"));
  }
}
