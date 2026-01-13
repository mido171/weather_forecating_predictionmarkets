package com.predictionmarkets.weather.gribstream;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class GribstreamDailyMetrics {
  private GribstreamDailyMetrics() {
  }

  public static TmaxResult computeTmax(List<GribstreamRow> rows) {
    if (rows == null || rows.isEmpty()) {
      throw new IllegalArgumentException("rows are required");
    }
    GribstreamRow maxRow = rows.stream()
        .max(Comparator.comparingDouble(GribstreamRow::tmpk))
        .orElseThrow(() -> new IllegalArgumentException("rows are required"));
    double tmaxK = maxRow.tmpk();
    double tmaxF = kelvinToF(tmaxK);
    Instant forecastedAtUtc = maxRow.forecastedAt();
    if (forecastedAtUtc == null) {
      throw new IllegalArgumentException("forecastedAt is required for max row");
    }
    return new TmaxResult(tmaxK, tmaxF, forecastedAtUtc);
  }

  public static SpreadResult computeSpread(List<GribstreamRow> rows, int minMembers) {
    if (rows == null || rows.isEmpty()) {
      throw new IllegalArgumentException("rows are required");
    }
    if (minMembers < 1) {
      throw new IllegalArgumentException("minMembers must be >= 1");
    }
    Map<Instant, List<Double>> grouped = new HashMap<>();
    for (GribstreamRow row : rows) {
      Instant forecastedTime = row.forecastedTime();
      if (forecastedTime == null) {
        continue;
      }
      grouped.computeIfAbsent(forecastedTime, ignored -> new ArrayList<>())
          .add(row.tmpk());
    }
    double maxStddev = Double.NEGATIVE_INFINITY;
    int usedTimes = 0;
    for (Map.Entry<Instant, List<Double>> entry : grouped.entrySet()) {
      List<Double> values = entry.getValue();
      if (values.size() < minMembers) {
        continue;
      }
      double stddev = stddev(values);
      if (Double.isFinite(stddev)) {
        usedTimes++;
        if (stddev > maxStddev) {
          maxStddev = stddev;
        }
      }
    }
    if (usedTimes == 0 || !Double.isFinite(maxStddev)) {
      throw new IllegalArgumentException("insufficient member coverage for spread");
    }
    double spreadF = maxStddev * 9.0 / 5.0;
    return new SpreadResult(maxStddev, spreadF, usedTimes);
  }

  public static double kelvinToF(double valueK) {
    return (valueK - 273.15) * 9.0 / 5.0 + 32.0;
  }

  private static double stddev(List<Double> values) {
    int n = values.size();
    if (n == 0) {
      return Double.NaN;
    }
    double mean = 0.0;
    for (double value : values) {
      mean += value;
    }
    mean /= n;
    double variance = 0.0;
    for (double value : values) {
      double delta = value - mean;
      variance += delta * delta;
    }
    variance /= n;
    return Math.sqrt(variance);
  }

  public record TmaxResult(double tmaxK, double tmaxF, Instant forecastedAtUtc) {
  }

  public record SpreadResult(double spreadK, double spreadF, int timesUsed) {
  }
}
