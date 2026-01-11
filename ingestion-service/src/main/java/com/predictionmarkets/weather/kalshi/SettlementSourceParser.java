package com.predictionmarkets.weather.kalshi;

import java.net.URI;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import org.springframework.stereotype.Component;

@Component
public class SettlementSourceParser {
  public SettlementSource parse(String url) {
    if (url == null || url.isBlank()) {
      throw new IllegalArgumentException("Settlement source URL is required");
    }

    URI uri = URI.create(url);
    if (uri.getHost() == null || !uri.getHost().equalsIgnoreCase("forecast.weather.gov")) {
      throw new IllegalArgumentException("Unexpected settlement host: " + uri.getHost());
    }
    if (uri.getPath() == null || !uri.getPath().equals("/product.php")) {
      throw new IllegalArgumentException("Unexpected settlement path: " + uri.getPath());
    }

    Map<String, String> params = parseQuery(uri.getRawQuery());
    String site = params.get("site");
    String issuedby = params.get("issuedby");
    String product = params.get("product");
    if (site == null || site.isBlank()) {
      throw new IllegalArgumentException("Settlement URL missing site parameter");
    }
    if (issuedby == null || issuedby.isBlank()) {
      throw new IllegalArgumentException("Settlement URL missing issuedby parameter");
    }
    if (product == null || !product.equalsIgnoreCase("CLI")) {
      throw new IllegalArgumentException("Settlement URL missing product=CLI");
    }

    return new SettlementSource(site.trim().toUpperCase(Locale.ROOT),
        issuedby.trim().toUpperCase(Locale.ROOT));
  }

  private Map<String, String> parseQuery(String rawQuery) {
    Map<String, String> params = new HashMap<>();
    if (rawQuery == null || rawQuery.isBlank()) {
      return params;
    }
    for (String part : rawQuery.split("&")) {
      if (part.isBlank()) {
        continue;
      }
      int idx = part.indexOf('=');
      if (idx <= 0) {
        continue;
      }
      String key = decode(part.substring(0, idx));
      String value = decode(part.substring(idx + 1));
      params.put(key, value);
    }
    return params;
  }

  private String decode(String value) {
    return URLDecoder.decode(value, StandardCharsets.UTF_8);
  }
}
