package com.predictionmarkets.weather.klga.iemmos;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.common.Hashing;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.springframework.stereotype.Component;

@Component
public class IemMosPlanner {
  private final ObjectMapper objectMapper;

  public IemMosPlanner(ObjectMapper objectMapper) {
    this.objectMapper = objectMapper;
  }

  public List<IemMosChunk> plan(IemMosBackfillProperties properties, List<IemMosStation> stations) {
    if (properties.getThrough() == null) {
      throw new IllegalArgumentException("iem-mos.through is required");
    }
    if (stations == null || stations.isEmpty()) {
      throw new IllegalArgumentException("At least one MOS station is required");
    }
    List<IemMosChunk> chunks = new ArrayList<>();
    for (IemMosProduct product : selectedProducts(properties)) {
      LocalDate productStart = properties.getStart() == null
          ? product.defaultStartDate()
          : max(product.defaultStartDate(), properties.getStart());
      if (productStart.isAfter(properties.getThrough())) {
        continue;
      }
      for (IemMosStation station : stations) {
        chunks.addAll(planStationProduct(properties, station, product, productStart));
      }
    }
    return List.copyOf(chunks);
  }

  private List<IemMosProduct> selectedProducts(IemMosBackfillProperties properties) {
    if (properties.getProducts().isEmpty()) {
      return Arrays.asList(IemMosProduct.values());
    }
    Set<String> requested = properties.getProducts().stream()
        .map(value -> value.trim().toUpperCase(Locale.ROOT))
        .filter(value -> !value.isBlank())
        .collect(Collectors.toSet());
    List<IemMosProduct> selected = Arrays.stream(IemMosProduct.values())
        .filter(product -> requested.contains(product.productCode()))
        .toList();
    if (selected.isEmpty()) {
      throw new IllegalArgumentException("No supported IEM MOS products matched " + requested);
    }
    return selected;
  }

  public IemMosChunk splitChunk(IemMosChunk parent, Instant splitPointUtc, String suffix) {
    return buildChunk(
        parent.jobId(),
        parent.station(),
        parent.product(),
        parent.cutoffId(),
        parent.windowStartUtc(),
        splitPointUtc,
        suffix);
  }

  public IemMosChunk splitChunkTail(IemMosChunk parent, Instant splitPointUtc, String suffix) {
    return buildChunk(
        parent.jobId(),
        parent.station(),
        parent.product(),
        parent.cutoffId(),
        splitPointUtc,
        parent.windowEndUtc(),
        suffix);
  }

  private List<IemMosChunk> planStationProduct(IemMosBackfillProperties properties,
                                               IemMosStation station,
                                               IemMosProduct product,
                                               LocalDate startDate) {
    LocalDate cursor = startDate;
    LocalDate through = properties.getThrough();
    List<IemMosChunk> chunks = new ArrayList<>();
    while (!cursor.isAfter(through)) {
      LocalDate yearEndExclusive = LocalDate.of(cursor.getYear() + 1, 1, 1);
      LocalDate endExclusiveDate = min(yearEndExclusive, through.plusDays(1));
      chunks.add(buildChunk(
          properties.getJobId(),
          station,
          product,
          properties.getCutoffId(),
          cursor.atStartOfDay().toInstant(ZoneOffset.UTC),
          endExclusiveDate.atStartOfDay().toInstant(ZoneOffset.UTC),
          null));
      cursor = endExclusiveDate;
    }
    return chunks;
  }

  private IemMosChunk buildChunk(String jobId,
                                 IemMosStation station,
                                 IemMosProduct product,
                                 String cutoffId,
                                 Instant windowStartUtc,
                                 Instant windowEndUtc,
                                 String suffix) {
    String requestJson = requestJson(jobId, station, product, cutoffId, windowStartUtc, windowEndUtc);
    String requestSha = Hashing.sha256Hex(requestJson);
    String chunkId = "iem_mos:" + jobId + ":" + product.productCode() + ":" + station.stationId()
        + ":" + windowStartUtc.toString().replace(":", "")
        + ":" + windowEndUtc.toString().replace(":", "")
        + (suffix == null || suffix.isBlank() ? "" : ":" + suffix);
    return new IemMosChunk(
        chunkId,
        jobId,
        station,
        product,
        cutoffId,
        windowStartUtc,
        windowEndUtc,
        requestSha,
        requestJson);
  }

  private String requestJson(String jobId,
                             IemMosStation station,
                             IemMosProduct product,
                             String cutoffId,
                             Instant windowStartUtc,
                             Instant windowEndUtc) {
    Map<String, Object> payload = new LinkedHashMap<>();
    payload.put("source", "iem_mos");
    payload.put("endpoint", "/cgi-bin/request/mos.py");
    payload.put("jobId", jobId);
    payload.put("cutoffId", cutoffId);
    payload.put("stationId", station.stationId());
    payload.put("mosStationId", station.mosStationId());
    payload.put("sourceProduct", product.productCode());
    payload.put("endpointModel", product.endpointModel().name());
    payload.put("windowStartUtc", windowStartUtc.toString());
    payload.put("windowEndUtc", windowEndUtc.toString());
    payload.put("format", "json");
    try {
      return objectMapper.writeValueAsString(payload);
    } catch (JsonProcessingException ex) {
      throw new IllegalStateException("Failed to serialize IEM MOS request identity", ex);
    }
  }

  private static LocalDate max(LocalDate left, LocalDate right) {
    return left.isAfter(right) ? left : right;
  }

  private static LocalDate min(LocalDate left, LocalDate right) {
    return left.isBefore(right) ? left : right;
  }
}
