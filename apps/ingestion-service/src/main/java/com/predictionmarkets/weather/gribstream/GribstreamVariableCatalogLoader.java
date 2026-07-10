package com.predictionmarkets.weather.gribstream;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Component;

@Component
public class GribstreamVariableCatalogLoader {
  private static final Pattern ROW_PATTERN = Pattern.compile("<tr>(.*?)</tr>", Pattern.DOTALL);
  private static final Pattern CELL_PATTERN = Pattern.compile("<td[^>]*>(.*?)</td>", Pattern.DOTALL);

  private final ResourceLoader resourceLoader;

  public GribstreamVariableCatalogLoader(ResourceLoader resourceLoader) {
    this.resourceLoader = resourceLoader;
  }

  public Map<String, List<GribstreamVariableSpec>> loadCatalog(String resourcePath,
                                                                List<String> models) {
    Objects.requireNonNull(resourcePath, "resourcePath is required");
    Resource resource = resourceLoader.getResource(resourcePath);
    String content = readResource(resource);
    Map<String, List<GribstreamVariableSpec>> catalog = new HashMap<>();
    for (String model : models) {
      String normalized = normalizeModel(model);
      String section = extractSection(content, normalized);
      if (section == null) {
        continue;
      }
      catalog.put(normalized, parseTable(section));
    }
    return catalog;
  }

  private List<GribstreamVariableSpec> parseTable(String section) {
    List<GribstreamVariableSpec> specs = new ArrayList<>();
    Matcher rowMatcher = ROW_PATTERN.matcher(section);
    while (rowMatcher.find()) {
      String row = rowMatcher.group(1);
      List<String> cells = extractCells(row);
      if (cells.size() < 3) {
        continue;
      }
      String name = cells.get(0);
      String level = cells.get(1);
      String info = cells.get(2);
      if (name.equalsIgnoreCase("Name") || name.isBlank()) {
        continue;
      }
      int minHorizon = parseIntSafe(cells, 5);
      int maxHorizon = parseIntSafe(cells, 6);
      specs.add(new GribstreamVariableSpec(
          name,
          level,
          info == null ? "" : info,
          minHorizon,
          maxHorizon));
    }
    return specs;
  }

  private List<String> extractCells(String row) {
    List<String> cells = new ArrayList<>();
    Matcher cellMatcher = CELL_PATTERN.matcher(row);
    while (cellMatcher.find()) {
      String cell = stripTags(cellMatcher.group(1));
      cells.add(unescapeHtml(cell));
    }
    return cells;
  }

  private String extractSection(String content, String model) {
    String start = "===== GRIBSTREAM DISCOVERY START model_page_" + model + " =====";
    String end = "===== GRIBSTREAM DISCOVERY END model_page_" + model + " =====";
    int startIdx = content.indexOf(start);
    if (startIdx < 0) {
      return null;
    }
    int endIdx = content.indexOf(end, startIdx);
    if (endIdx < 0) {
      return null;
    }
    return content.substring(startIdx + start.length(), endIdx);
  }

  private String readResource(Resource resource) {
    try (BufferedReader reader = new BufferedReader(
        new InputStreamReader(resource.getInputStream(), StandardCharsets.UTF_8))) {
      StringBuilder builder = new StringBuilder();
      String line;
      while ((line = reader.readLine()) != null) {
        builder.append(line).append('\n');
      }
      return builder.toString();
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to read gribstream catalog resource", ex);
    }
  }

  private String normalizeModel(String model) {
    if (model == null || model.isBlank()) {
      return "";
    }
    return model.trim().toLowerCase(Locale.ROOT);
  }

  private int parseIntSafe(List<String> cells, int index) {
    if (index >= cells.size()) {
      return -1;
    }
    String raw = cells.get(index);
    if (raw == null || raw.isBlank()) {
      return -1;
    }
    try {
      return Integer.parseInt(raw.trim());
    } catch (NumberFormatException ex) {
      return -1;
    }
  }

  private String stripTags(String input) {
    if (input == null) {
      return "";
    }
    return input.replaceAll("<.*?>", "").replaceAll("\\s+", " ").trim();
  }

  private String unescapeHtml(String input) {
    if (input == null || input.isEmpty()) {
      return "";
    }
    return input.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'");
  }
}
