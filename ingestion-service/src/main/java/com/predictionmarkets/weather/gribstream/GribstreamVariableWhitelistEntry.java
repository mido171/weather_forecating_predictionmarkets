package com.predictionmarkets.weather.gribstream;

public record GribstreamVariableWhitelistEntry(
    String name,
    String level,
    String info,
    int lineNumber) {

  public String normalizedKey() {
    return GribstreamVariableWhitelistLoader.normalizeKey(name, level, info);
  }

  public String describe() {
    String safeInfo = info == null ? "" : info;
    return name + " | " + level + " | " + safeInfo;
  }
}
