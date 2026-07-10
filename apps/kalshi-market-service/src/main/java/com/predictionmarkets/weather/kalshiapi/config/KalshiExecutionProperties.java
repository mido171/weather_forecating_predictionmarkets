package com.predictionmarkets.weather.kalshiapi.config;

import jakarta.annotation.PostConstruct;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@ConfigurationProperties(prefix = "kalshi")
@Validated
public class KalshiExecutionProperties {

  @NotNull
  private KalshiEnvironment environment = KalshiEnvironment.PROD;

  private String restBaseUrl;
  private String wsUrl;

  private boolean authEnabled = true;
  private boolean tradingEnabled = true;
  private boolean authDebug = false;

  private String credentialsPath;
  private String apiKeyId;
  private Path privateKeyPemPath;

  @Valid
  private final Timeouts timeouts = new Timeouts();

  @Valid
  private final Retry retry = new Retry();

  @Valid
  private final RateLimiting rateLimiting = new RateLimiting();

  @Valid
  private final WebSocket webSocket = new WebSocket();

  @Valid
  private final Guardrails guardrails = new Guardrails();

  @Valid
  private final Smoke smoke = new Smoke();

  @PostConstruct
  void validateConfiguration() {
    if (tradingEnabled && !authEnabled) {
      throw new IllegalStateException("kalshi.trading-enabled requires kalshi.auth-enabled=true");
    }

    if (authEnabled) {
      if (isBlank(credentialsPath)) {
        if (isBlank(apiKeyId)) {
          throw new IllegalStateException("kalshi.api-key-id is required when auth is enabled and credentials-path is not set");
        }
        if (privateKeyPemPath == null) {
          throw new IllegalStateException("kalshi.private-key-pem-path is required when auth is enabled and credentials-path is not set");
        }
        if (!Files.exists(privateKeyPemPath) || !Files.isReadable(privateKeyPemPath)) {
          throw new IllegalStateException("kalshi.private-key-pem-path does not exist or is not readable: " + privateKeyPemPath);
        }
      }
    }

    if (guardrails.defaultMaxBuyContracts < 1) {
      throw new IllegalStateException("kalshi.guardrails.default-max-buy-contracts must be >= 1");
    }
  }

  public String resolvedRestBaseUrl() {
    return isBlank(restBaseUrl) ? environment.restBaseUrl() : restBaseUrl.trim();
  }

  public String resolvedWsUrl() {
    return isBlank(wsUrl) ? environment.wsUrl() : wsUrl.trim();
  }

  public KalshiEnvironment getEnvironment() {
    return environment;
  }

  public void setEnvironment(KalshiEnvironment environment) {
    this.environment = environment;
  }

  public String getRestBaseUrl() {
    return restBaseUrl;
  }

  public void setRestBaseUrl(String restBaseUrl) {
    this.restBaseUrl = restBaseUrl;
  }

  public String getWsUrl() {
    return wsUrl;
  }

  public void setWsUrl(String wsUrl) {
    this.wsUrl = wsUrl;
  }

  public boolean isAuthEnabled() {
    return authEnabled;
  }

  public void setAuthEnabled(boolean authEnabled) {
    this.authEnabled = authEnabled;
  }

  public boolean isTradingEnabled() {
    return tradingEnabled;
  }

  public void setTradingEnabled(boolean tradingEnabled) {
    this.tradingEnabled = tradingEnabled;
  }

  public boolean isAuthDebug() {
    return authDebug;
  }

  public void setAuthDebug(boolean authDebug) {
    this.authDebug = authDebug;
  }

  public String getCredentialsPath() {
    return credentialsPath;
  }

  public void setCredentialsPath(String credentialsPath) {
    this.credentialsPath = credentialsPath;
  }

  public String getApiKeyId() {
    return apiKeyId;
  }

  public void setApiKeyId(String apiKeyId) {
    this.apiKeyId = apiKeyId;
  }

  public Path getPrivateKeyPemPath() {
    return privateKeyPemPath;
  }

  public void setPrivateKeyPemPath(Path privateKeyPemPath) {
    this.privateKeyPemPath = privateKeyPemPath;
  }

  public Timeouts getTimeouts() {
    return timeouts;
  }

  public Retry getRetry() {
    return retry;
  }

  public RateLimiting getRateLimiting() {
    return rateLimiting;
  }

  public WebSocket getWebSocket() {
    return webSocket;
  }

  public Guardrails getGuardrails() {
    return guardrails;
  }

  public Smoke getSmoke() {
    return smoke;
  }

  private static boolean isBlank(String value) {
    return value == null || value.isBlank();
  }

  public static class Timeouts {
    @Min(100)
    private int connectTimeoutMs = 5_000;

    @Min(100)
    private int requestTimeoutMs = 10_000;

    public int getConnectTimeoutMs() {
      return connectTimeoutMs;
    }

    public void setConnectTimeoutMs(int connectTimeoutMs) {
      this.connectTimeoutMs = connectTimeoutMs;
    }

    public int getRequestTimeoutMs() {
      return requestTimeoutMs;
    }

    public void setRequestTimeoutMs(int requestTimeoutMs) {
      this.requestTimeoutMs = requestTimeoutMs;
    }
  }

  public static class Retry {
    @Min(0)
    private int maxRetries = 3;

    @Min(1)
    private long baseBackoffMs = 250;

    @Min(1)
    private long maxBackoffMs = 3_000;

    public int getMaxRetries() {
      return maxRetries;
    }

    public void setMaxRetries(int maxRetries) {
      this.maxRetries = maxRetries;
    }

    public long getBaseBackoffMs() {
      return baseBackoffMs;
    }

    public void setBaseBackoffMs(long baseBackoffMs) {
      this.baseBackoffMs = baseBackoffMs;
    }

    public long getMaxBackoffMs() {
      return maxBackoffMs;
    }

    public void setMaxBackoffMs(long maxBackoffMs) {
      this.maxBackoffMs = maxBackoffMs;
    }
  }

  public static class RateLimiting {
    @Min(1)
    private int maxReadRequestsPerSecond = 30;

    @Min(1)
    private int maxWriteRequestsPerSecond = 10;

    public int getMaxReadRequestsPerSecond() {
      return maxReadRequestsPerSecond;
    }

    public void setMaxReadRequestsPerSecond(int maxReadRequestsPerSecond) {
      this.maxReadRequestsPerSecond = maxReadRequestsPerSecond;
    }

    public int getMaxWriteRequestsPerSecond() {
      return maxWriteRequestsPerSecond;
    }

    public void setMaxWriteRequestsPerSecond(int maxWriteRequestsPerSecond) {
      this.maxWriteRequestsPerSecond = maxWriteRequestsPerSecond;
    }
  }

  public static class WebSocket {
    private boolean enabled = true;

    @Min(0)
    private int maxReconnectAttempts = 0;

    @Min(1)
    private long reconnectBaseBackoffMs = 500;

    @Min(5)
    private int watchdogTimeoutSeconds = 30;

    public boolean isEnabled() {
      return enabled;
    }

    public void setEnabled(boolean enabled) {
      this.enabled = enabled;
    }

    public int getMaxReconnectAttempts() {
      return maxReconnectAttempts;
    }

    public void setMaxReconnectAttempts(int maxReconnectAttempts) {
      this.maxReconnectAttempts = maxReconnectAttempts;
    }

    public long getReconnectBaseBackoffMs() {
      return reconnectBaseBackoffMs;
    }

    public void setReconnectBaseBackoffMs(long reconnectBaseBackoffMs) {
      this.reconnectBaseBackoffMs = reconnectBaseBackoffMs;
    }

    public int getWatchdogTimeoutSeconds() {
      return watchdogTimeoutSeconds;
    }

    public void setWatchdogTimeoutSeconds(int watchdogTimeoutSeconds) {
      this.watchdogTimeoutSeconds = watchdogTimeoutSeconds;
    }
  }

  public static class Guardrails {
    private boolean enabled = true;
    private int defaultMaxBuyContracts = 1;
    private final Map<String, Integer> marketOverrides = new HashMap<>();
    private boolean failClosedOnUnknownOrderState = true;
    private boolean cancelRestingOnCapBreach = true;
    private boolean startupReconcile = true;

    public boolean isEnabled() {
      return enabled;
    }

    public void setEnabled(boolean enabled) {
      this.enabled = enabled;
    }

    public int getDefaultMaxBuyContracts() {
      return defaultMaxBuyContracts;
    }

    public void setDefaultMaxBuyContracts(int defaultMaxBuyContracts) {
      this.defaultMaxBuyContracts = defaultMaxBuyContracts;
    }

    public Map<String, Integer> getMarketOverrides() {
      return marketOverrides;
    }

    public boolean isFailClosedOnUnknownOrderState() {
      return failClosedOnUnknownOrderState;
    }

    public void setFailClosedOnUnknownOrderState(boolean failClosedOnUnknownOrderState) {
      this.failClosedOnUnknownOrderState = failClosedOnUnknownOrderState;
    }

    public boolean isCancelRestingOnCapBreach() {
      return cancelRestingOnCapBreach;
    }

    public void setCancelRestingOnCapBreach(boolean cancelRestingOnCapBreach) {
      this.cancelRestingOnCapBreach = cancelRestingOnCapBreach;
    }

    public boolean isStartupReconcile() {
      return startupReconcile;
    }

    public void setStartupReconcile(boolean startupReconcile) {
      this.startupReconcile = startupReconcile;
    }
  }

  public static class Smoke {
    private boolean enabled = false;
    private String marketTicker;
    private String side = "yes";
    @Min(1)
    private int desiredContracts = 1;
    @Min(1)
    private int buyMaxCostCents = 100;
    @Min(0)
    private int orderbookDepth = 20;
    @Min(0)
    private int orderbookWarmupSeconds = 5;

    public boolean isEnabled() {
      return enabled;
    }

    public void setEnabled(boolean enabled) {
      this.enabled = enabled;
    }

    public String getMarketTicker() {
      return marketTicker;
    }

    public void setMarketTicker(String marketTicker) {
      this.marketTicker = marketTicker;
    }

    public String getSide() {
      return side;
    }

    public void setSide(String side) {
      this.side = side;
    }

    public int getDesiredContracts() {
      return desiredContracts;
    }

    public void setDesiredContracts(int desiredContracts) {
      this.desiredContracts = desiredContracts;
    }

    public int getBuyMaxCostCents() {
      return buyMaxCostCents;
    }

    public void setBuyMaxCostCents(int buyMaxCostCents) {
      this.buyMaxCostCents = buyMaxCostCents;
    }

    public int getOrderbookDepth() {
      return orderbookDepth;
    }

    public void setOrderbookDepth(int orderbookDepth) {
      this.orderbookDepth = orderbookDepth;
    }

    public int getOrderbookWarmupSeconds() {
      return orderbookWarmupSeconds;
    }

    public void setOrderbookWarmupSeconds(int orderbookWarmupSeconds) {
      this.orderbookWarmupSeconds = orderbookWarmupSeconds;
    }
  }
}
