package com.predictionmarkets.weather.gribstream;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Component;

@Component
public class GribstreamVariableWhitelistLoader {
  private final ResourceLoader resourceLoader;

  public GribstreamVariableWhitelistLoader(ResourceLoader resourceLoader) {
    this.resourceLoader = resourceLoader;
  }

  public Map<String, List<GribstreamVariableWhitelistEntry>> loadWhitelist(String resourcePath) {
    if (resourcePath == null || resourcePath.isBlank()) {
      return Map.of();
    }
    Resource resource = resourceLoader.getResource(resourcePath);
    if (!resource.exists()) {
      throw new IllegalArgumentException("Gribstream whitelist resource not found: " + resourcePath);
    }
    Map<String, List<GribstreamVariableWhitelistEntry>> byModel = new LinkedHashMap<>();
    try (BufferedReader reader = new BufferedReader(
        new InputStreamReader(resource.getInputStream(), StandardCharsets.UTF_8))) {
      String line;
      int lineNumber = 0;
      while ((line = reader.readLine()) != null) {
        lineNumber++;
        String trimmed = line.trim();
        if (trimmed.isEmpty() || trimmed.startsWith("#")) {
          continue;
        }
        String[] parts = trimmed.split(",", -1);
        if (lineNumber == 1 && isHeader(parts)) {
          continue;
        }
        if (parts.length < 3) {
          throw new IllegalArgumentException("Invalid whitelist line " + lineNumber
              + " (expected model,name,level,info): " + line);
        }
        String model = normalizeModel(parts[0]);
        String name = parts[1].trim();
        String level = parts[2].trim();
        String info = parts.length > 3 ? parts[3].trim() : "";
        if (model.isEmpty() || name.isEmpty() || level.isEmpty()) {
          throw new IllegalArgumentException("Invalid whitelist line " + lineNumber
              + " (missing model/name/level): " + line);
        }
        byModel.computeIfAbsent(model, key -> new ArrayList<>())
            .add(new GribstreamVariableWhitelistEntry(name, level, info, lineNumber));
      }
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to read gribstream whitelist resource: " + resourcePath, ex);
    }
    return byModel;
  }

  static String normalizeKey(String name, String level, String info) {
    return normalizeToken(name) + "|" + normalizeToken(level) + "|" + normalizeToken(info);
  }

  static String normalizeToken(String value) {
    if (value == null) {
      return "";
    }
    String trimmed = value.trim();
    if (trimmed.isEmpty()) {
      return "";
    }
    String normalized = trimmed
        .replace('\u2013', '-')
        .replace('\u2014', '-')
        .replace('\u2212', '-')
        .replace('\u00a0', ' ')
        .replace('\u202f', ' ');
    normalized = normalized.replaceAll("\\s+", " ").trim();
    return normalized.toLowerCase(Locale.ROOT);
  }

  private String normalizeModel(String model) {
    if (model == null || model.isBlank()) {
      return "";
    }
    return model.trim().toLowerCase(Locale.ROOT);
  }

  private boolean isHeader(String[] parts) {
    if (parts.length < 3) {
      return false;
    }
    return "model".equalsIgnoreCase(parts[0].trim())
        && "variable_name".equalsIgnoreCase(parts[1].trim())
        && "variable_level".equalsIgnoreCase(parts[2].trim());
  }
}
