package com.predictionmarkets.weather.klga.wu;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

record WuTruthStation(
    String stationId,
    String wundergroundStationId,
    String weatherComLocationId,
    String timezoneName,
    String wundergroundPageLocationPath
) {
  private static final Map<String, WuTruthStation> STATIONS = buildStations();

  static List<WuTruthStation> select(String selection) {
    if (selection == null || selection.isBlank() || "all".equalsIgnoreCase(selection.trim())) {
      return new ArrayList<>(STATIONS.values());
    }
    List<WuTruthStation> selected = new ArrayList<>();
    for (String token : selection.split(",")) {
      String stationId = token.trim().toUpperCase(Locale.ROOT);
      if (stationId.isEmpty()) {
        continue;
      }
      WuTruthStation station = STATIONS.get(stationId);
      if (station == null) {
        throw new IllegalArgumentException("Unknown Wunderground station: " + stationId);
      }
      selected.add(station);
    }
    if (selected.isEmpty()) {
      throw new IllegalArgumentException("No stations selected");
    }
    return selected;
  }

  static WuTruthStation byStationId(String stationId) {
    WuTruthStation station = STATIONS.get(stationId.toUpperCase(Locale.ROOT));
    if (station == null) {
      throw new IllegalArgumentException("Unknown Wunderground station: " + stationId);
    }
    return station;
  }

  String pageUrl(LocalDate localDate) {
    return "https://www.wunderground.com/history/daily/"
        + wundergroundPageLocationPath
        + "/"
        + wundergroundStationId
        + "/date/"
        + localDate.getYear()
        + "-"
        + localDate.getMonthValue()
        + "-"
        + localDate.getDayOfMonth();
  }

  private static Map<String, WuTruthStation> buildStations() {
    Map<String, WuTruthStation> rows = new LinkedHashMap<>();
    add(rows, "KLGA", "us/ny/new-york-city");
    add(rows, "KNYC", "us/ny/new-york-city");
    add(rows, "KJFK", "us/ny/jamaica");
    add(rows, "KEWR", "us/nj/newark");
    add(rows, "KTEB", "us/nj/teterboro");
    add(rows, "KHPN", "us/ny/white-plains");
    add(rows, "KISP", "us/ny/islip");
    add(rows, "KFRG", "us/ny/farmingdale");
    add(rows, "KBDR", "us/ct/stratford");
    add(rows, "KSWF", "us/ny/newburgh");
    add(rows, "KPOU", "us/ny/poughkeepsie");
    add(rows, "KMMU", "us/nj/morristown");
    add(rows, "KCDW", "us/nj/caldwell");
    add(rows, "KPHL", "us/pa/philadelphia");
    add(rows, "KBOS", "us/ma/boston");
    add(rows, "KDCA", "us/dc/washington");
    add(rows, "KBWI", "us/md/baltimore");
    add(rows, "KALB", "us/ny/albany");
    add(rows, "KABE", "us/pa/allentown");
    return rows;
  }

  private static void add(Map<String, WuTruthStation> rows, String stationId, String pageLocationPath) {
    rows.put(
        stationId,
        new WuTruthStation(stationId, stationId, stationId + ":9:US", "America/New_York", pageLocationPath));
  }
}
