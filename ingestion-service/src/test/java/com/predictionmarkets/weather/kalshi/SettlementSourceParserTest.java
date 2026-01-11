package com.predictionmarkets.weather.kalshi;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class SettlementSourceParserTest {
  private final SettlementSourceParser parser = new SettlementSourceParser();

  @Test
  void parseValidSettlementUrl() {
    String url = "https://forecast.weather.gov/product.php?site=MFL&product=CLI&issuedby=MIA";
    SettlementSource source = parser.parse(url);
    assertEquals("MFL", source.wfoSite());
    assertEquals("MIA", source.issuedby());
  }

  @Test
  void missingQueryParamsFailFast() {
    String url = "https://forecast.weather.gov/product.php?site=MFL&product=CLI";
    assertThrows(IllegalArgumentException.class, () -> parser.parse(url));
  }

  @Test
  void unexpectedFormatFailsFast() {
    String url = "https://example.com/product.php?site=MFL&product=CLI&issuedby=MIA";
    assertThrows(IllegalArgumentException.class, () -> parser.parse(url));
  }
}
