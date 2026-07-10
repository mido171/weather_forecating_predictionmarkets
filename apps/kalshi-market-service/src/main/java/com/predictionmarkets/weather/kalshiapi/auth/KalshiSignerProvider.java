package com.predictionmarkets.weather.kalshiapi.auth;

import com.predictionmarkets.weather.kalshiapi.config.KalshiExecutionProperties;
import java.time.Clock;
import java.util.concurrent.atomic.AtomicReference;
import org.springframework.stereotype.Component;

@Component
public class KalshiSignerProvider {

  private final KalshiExecutionProperties properties;
  private final KalshiCredentialsLoader credentialsLoader;
  private final Clock clock;
  private final AtomicReference<KalshiSigner> signerRef = new AtomicReference<>();

  public KalshiSignerProvider(KalshiExecutionProperties properties,
                              KalshiCredentialsLoader credentialsLoader,
                              Clock clock) {
    this.properties = properties;
    this.credentialsLoader = credentialsLoader;
    this.clock = clock;
  }

  public boolean isAuthEnabled() {
    return properties.isAuthEnabled();
  }

  public KalshiSigner getSigner() {
    if (!properties.isAuthEnabled()) {
      throw new IllegalStateException("Kalshi authentication is disabled");
    }

    KalshiSigner existing = signerRef.get();
    if (existing != null) {
      return existing;
    }

    KalshiCredentials credentials = credentialsLoader.loadCredentials();
    KalshiSigner created = new KalshiSigner(credentials.privateKey(), credentials.apiKeyId(), clock);
    signerRef.compareAndSet(null, created);
    return signerRef.get();
  }
}
