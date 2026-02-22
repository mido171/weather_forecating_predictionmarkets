package com.predictionmarkets.weather.gribstream;

import com.predictionmarkets.weather.config.EvomiProxyProperties;
import java.net.InetSocketAddress;
import java.net.Proxy;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import okhttp3.Credentials;
import okhttp3.OkHttpClient;
import org.slf4j.Logger;

final class EvomiProxySupport {
  private static final String PLACEHOLDER = "<PUT_EVOMI_PROXY_CRED_HERE>";
  private static final Pattern FORMAT2 =
      Pattern.compile("^(https?|socks5)://([^:/]+):(\\d+):([^:]+):(.+)$");

  private EvomiProxySupport() {
  }

  static void applyIfEnabled(OkHttpClient.Builder builder,
                             EvomiProxyProperties properties,
                             Logger logger) {
    Objects.requireNonNull(builder, "builder is required");
    if (properties == null || !properties.isEnabled()) {
      if (logger != null) {
        logger.info("[GRIBSTREAM] Evomi proxy disabled (evomi.proxy.enabled=false)");
      }
      return;
    }
    String credential = properties.getCredential();
    if (credential == null || credential.isBlank()) {
      throw new IllegalArgumentException(
          "evomi.proxy.credential is required when evomi.proxy.enabled=true");
    }
    if (credential.trim().equalsIgnoreCase(PLACEHOLDER)) {
      throw new IllegalArgumentException(
          "evomi.proxy.credential must be set (placeholder found)");
    }
    ProxySettings settings = parseCredential(credential.trim());
    Proxy.Type proxyType = settings.isSocks() ? Proxy.Type.SOCKS : Proxy.Type.HTTP;
    builder.proxy(new Proxy(proxyType, new InetSocketAddress(settings.host(), settings.port())));
    if (!settings.username().isBlank()) {
      if (proxyType == Proxy.Type.HTTP) {
        String basic = Credentials.basic(
            settings.username(),
            settings.password(),
            StandardCharsets.ISO_8859_1);
        builder.proxyAuthenticator((route, response) -> {
          if (response.request().header("Proxy-Authorization") != null) {
            return null;
          }
          return response.request()
              .newBuilder()
              .header("Proxy-Authorization", basic)
              .build();
        });
      } else {
        logger.warn("Evomi SOCKS5 proxy credentials configured; "
            + "ensure JVM SOCKS authentication is configured if required.");
      }
    }
    if (logger != null) {
      logger.info("[GRIBSTREAM] Evomi proxy enabled host={} port={} scheme={} userLen={}",
          settings.host(),
          settings.port(),
          settings.normalizedScheme(),
          settings.username().length());
    }
  }

  private static ProxySettings parseCredential(String credential) {
    Matcher matcher = FORMAT2.matcher(credential);
    if (matcher.matches()) {
      String scheme = matcher.group(1);
      String host = matcher.group(2);
      int port = Integer.parseInt(matcher.group(3));
      String username = matcher.group(4);
      String password = matcher.group(5);
      return new ProxySettings(host, port, username, password, scheme);
    }
    URI uri = URI.create(credential);
    String scheme = uri.getScheme();
    if (scheme == null || scheme.isBlank()) {
      throw new IllegalArgumentException("evomi.proxy.credential missing scheme");
    }
    String host = uri.getHost();
    if (host == null || host.isBlank()) {
      throw new IllegalArgumentException("evomi.proxy.credential missing host");
    }
    int port = uri.getPort();
    if (port <= 0) {
      throw new IllegalArgumentException("evomi.proxy.credential missing port");
    }
    String userInfo = uri.getUserInfo();
    if (userInfo == null || !userInfo.contains(":")) {
      throw new IllegalArgumentException("evomi.proxy.credential missing username/password");
    }
    String[] parts = userInfo.split(":", 2);
    String username = parts[0];
    String password = parts[1];
    return new ProxySettings(host, port, username, password, scheme);
  }

  private record ProxySettings(String host,
                               int port,
                               String username,
                               String password,
                               String scheme) {
    private boolean isSocks() {
      return "socks5".equalsIgnoreCase(scheme);
    }

    private String normalizedScheme() {
      return scheme == null ? "" : scheme.toLowerCase(Locale.ROOT);
    }
  }
}
