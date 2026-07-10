package com.predictionmarkets.weather.pilot.metrics;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class JobMetricsAccumulator {
  private final List<Double> durationsMs = new ArrayList<>();
  private final Map<String, Integer> parserFailures = new LinkedHashMap<>();
  private final Map<String, Integer> missingCounts = new LinkedHashMap<>();

  private int totalRequests;
  private int succeeded;
  private int failed;
  private int skipped;
  private int deduped;
  private long totalBytes;
  private long totalRows;

  public synchronized void recordRequest(double durationMs,
                                         long bytesDownloaded,
                                         int rowsParsed,
                                         String status) {
    totalRequests += 1;
    durationsMs.add(durationMs);
    totalBytes += Math.max(bytesDownloaded, 0);
    totalRows += Math.max(rowsParsed, 0);
    String normalized = status == null ? "" : status.trim().toUpperCase();
    switch (normalized) {
      case "SUCCESS", "OK", "COMPLETE" -> succeeded += 1;
      case "SKIPPED" -> skipped += 1;
      case "DEDUPED" -> deduped += 1;
      default -> failed += 1;
    }
  }

  public synchronized void recordParserFailure(String sourceName) {
    parserFailures.merge(sourceName, 1, Integer::sum);
  }

  public synchronized void recordMissing(String key) {
    missingCounts.merge(key, 1, Integer::sum);
  }

  public synchronized int totalRequests() {
    return totalRequests;
  }

  public synchronized int succeeded() {
    return succeeded;
  }

  public synchronized int failed() {
    return failed;
  }

  public synchronized int skipped() {
    return skipped;
  }

  public synchronized int deduped() {
    return deduped;
  }

  public synchronized long totalBytes() {
    return totalBytes;
  }

  public synchronized long totalRows() {
    return totalRows;
  }

  public synchronized Double minRequestDurationMs() {
    return durationsMs.isEmpty() ? null
        : durationsMs.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
  }

  public synchronized Double meanRequestDurationMs() {
    return durationsMs.isEmpty() ? null
        : durationsMs.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
  }

  public synchronized Double p95RequestDurationMs() {
    if (durationsMs.isEmpty()) {
      return null;
    }
    List<Double> sorted = new ArrayList<>(durationsMs);
    sorted.sort(Double::compareTo);
    int index = Math.max(0, (int) Math.ceil(sorted.size() * 0.95d) - 1);
    return sorted.get(index);
  }

  public synchronized Double throughputMbPerSec() {
    double totalDurationMs = durationsMs.stream().mapToDouble(Double::doubleValue).sum();
    if (totalDurationMs <= 0.0d) {
      return null;
    }
    double megabytes = totalBytes / (1024.0d * 1024.0d);
    return megabytes / (totalDurationMs / 1000.0d);
  }

  public synchronized Map<String, Integer> parserFailures() {
    return Map.copyOf(parserFailures);
  }

  public synchronized Map<String, Integer> missingCounts() {
    return Map.copyOf(missingCounts);
  }

  public synchronized Map<String, Object> toSummaryMap() {
    Map<String, Object> summary = new LinkedHashMap<>();
    summary.put("totalRequests", totalRequests);
    summary.put("succeeded", succeeded);
    summary.put("failed", failed);
    summary.put("skipped", skipped);
    summary.put("deduped", deduped);
    summary.put("totalBytes", totalBytes);
    summary.put("totalRows", totalRows);
    summary.put("minRequestDurationMs", minRequestDurationMs());
    summary.put("meanRequestDurationMs", meanRequestDurationMs());
    summary.put("p95RequestDurationMs", p95RequestDurationMs());
    summary.put("throughputMbPerSec", throughputMbPerSec());
    summary.put("parserFailures", parserFailures());
    summary.put("missingCounts", missingCounts());
    return summary;
  }
}
