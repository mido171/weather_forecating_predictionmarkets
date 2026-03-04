package com.predictionmarkets.weather.kalshiapi.auth;

import com.predictionmarkets.weather.kalshiapi.config.KalshiExecutionProperties;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.PrivateKey;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Component;

@Component
public class KalshiCredentialsLoader {

  private static final Logger log = LoggerFactory.getLogger(KalshiCredentialsLoader.class);

  private static final Pattern UUID_PATTERN = Pattern.compile(
      "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}");
  private static final Pattern PRIVATE_KEY_BLOCK_PATTERN = Pattern.compile(
      "-----BEGIN [A-Z ]*PRIVATE KEY-----[\\s\\S]*?-----END [A-Z ]*PRIVATE KEY-----");

  private final KalshiExecutionProperties properties;
  private final ResourceLoader resourceLoader;

  public KalshiCredentialsLoader(KalshiExecutionProperties properties, ResourceLoader resourceLoader) {
    this.properties = properties;
    this.resourceLoader = resourceLoader;
  }

  public KalshiCredentials loadCredentials() {
    if (!isBlank(properties.getCredentialsPath())) {
      return loadFromCredentialsFile(properties.getCredentialsPath());
    }
    return loadFromExplicitProperties();
  }

  private KalshiCredentials loadFromCredentialsFile(String credentialsPath) {
    Resource resource = resolveResource(credentialsPath);
    if (!resource.exists()) {
      throw new IllegalStateException("Kalshi credentials file does not exist: " + credentialsPath);
    }
    if (!resource.isReadable()) {
      throw new IllegalStateException("Kalshi credentials file is not readable: " + credentialsPath);
    }

    String content = readResource(resource, credentialsPath);
    String apiKeyId = extractApiKeyId(content);
    String privateKeyPem = extractPrivateKeyPem(content, credentialsPath);
    PrivateKey privateKey = KalshiPrivateKeyLoader.loadPrivateKeyFromPem(privateKeyPem, credentialsPath);
    if (properties.isAuthDebug()) {
      log.info("Loaded Kalshi credentials from {} (apiKeyId={})", credentialsPath, maskApiKeyId(apiKeyId));
    }
    return new KalshiCredentials(apiKeyId, privateKey);
  }

  private KalshiCredentials loadFromExplicitProperties() {
    String apiKeyId = properties.getApiKeyId();
    Path privateKeyPath = properties.getPrivateKeyPemPath();
    if (isBlank(apiKeyId)) {
      throw new IllegalStateException("kalshi.api-key-id must be set when auth is enabled");
    }
    if (privateKeyPath == null) {
      throw new IllegalStateException("kalshi.private-key-pem-path must be set when auth is enabled");
    }
    PrivateKey privateKey = KalshiPrivateKeyLoader.loadPrivateKey(privateKeyPath);
    return new KalshiCredentials(apiKeyId, privateKey);
  }

  private Resource resolveResource(String location) {
    String trimmed = location.trim();
    if (trimmed.contains(":")) {
      return resourceLoader.getResource(trimmed);
    }

    Path asPath = Path.of(trimmed);
    if (asPath.isAbsolute() || Files.exists(asPath)) {
      return new FileSystemResource(asPath);
    }
    return resourceLoader.getResource("classpath:" + trimmed);
  }

  private String readResource(Resource resource, String description) {
    try (InputStream inputStream = resource.getInputStream()) {
      return new String(inputStream.readAllBytes(), StandardCharsets.UTF_8);
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to read Kalshi credentials from " + description, ex);
    }
  }

  private String extractApiKeyId(String content) {
    if (!isBlank(properties.getApiKeyId())) {
      return properties.getApiKeyId().trim();
    }

    String[] lines = content.split("\\R");
    for (int i = 0; i < lines.length; i++) {
      String line = lines[i] == null ? "" : lines[i].trim().toLowerCase();
      if (!line.contains("api key id")) {
        continue;
      }
      for (int j = i + 1; j < lines.length; j++) {
        String candidate = lines[j] == null ? "" : lines[j].trim();
        if (candidate.isEmpty()) {
          continue;
        }
        if (UUID_PATTERN.matcher(candidate).matches()) {
          return candidate;
        }
        break;
      }
    }

    Matcher matcher = UUID_PATTERN.matcher(content);
    if (matcher.find()) {
      return matcher.group();
    }
    throw new IllegalStateException("Unable to extract API key id from kalshi.credentials-path");
  }

  private String extractPrivateKeyPem(String content, String description) {
    Matcher matcher = PRIVATE_KEY_BLOCK_PATTERN.matcher(content);
    if (!matcher.find()) {
      throw new IllegalStateException("Unable to locate private key block in " + description);
    }
    return matcher.group().trim() + System.lineSeparator();
  }

  private static String maskApiKeyId(String apiKeyId) {
    if (isBlank(apiKeyId)) {
      return "<missing>";
    }
    String trimmed = apiKeyId.trim();
    if (trimmed.length() <= 8) {
      return "****";
    }
    return trimmed.substring(0, 4) + "..." + trimmed.substring(trimmed.length() - 4);
  }

  private static boolean isBlank(String value) {
    return value == null || value.isBlank();
  }
}
