package com.predictionmarkets.weather.kalshi;

import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.zone.ZoneRules;
import java.util.Map;
import java.util.Optional;

public final class StationZoneDefaults {
  private static final Map<String, String> DEFAULT_ZONE_IDS = Map.of(
      "KNYC", "America/New_York",
      "KPHL", "America/New_York",
      "KMIA", "America/New_York",
      "KMDW", "America/Chicago",
      "KLAX", "America/Los_Angeles");

  private StationZoneDefaults() {
  }

  public static Optional<String> zoneIdFor(String stationId) {
    if (stationId == null) {
      return Optional.empty();
    }
    return Optional.ofNullable(DEFAULT_ZONE_IDS.get(stationId));
  }

  public static int standardOffsetMinutes(String zoneId) {
    ZoneRules rules = ZoneId.of(zoneId).getRules();
    Instant reference = LocalDate.of(2020, 1, 15).atStartOfDay(ZoneOffset.UTC).toInstant();
    ZoneOffset offset = rules.getStandardOffset(reference);
    return offset.getTotalSeconds() / 60;
  }
}
