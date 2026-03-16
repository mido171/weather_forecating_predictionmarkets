package com.predictionmarkets.weather.pilot.source;

import com.predictionmarkets.weather.pilot.config.PilotIngestionProperties;
import java.util.concurrent.TimeUnit;
import okhttp3.OkHttpClient;
import org.springframework.stereotype.Component;

@Component
public class HttpClientFactory {
  private final PilotIngestionProperties properties;
  private volatile OkHttpClient client;

  public HttpClientFactory(PilotIngestionProperties properties) {
    this.properties = properties;
  }

  public OkHttpClient client() {
    if (client == null) {
      synchronized (this) {
        if (client == null) {
          client = new OkHttpClient.Builder()
              .connectTimeout(properties.getConnectTimeoutMs(), TimeUnit.MILLISECONDS)
              .readTimeout(properties.getReadTimeoutMs(), TimeUnit.MILLISECONDS)
              .writeTimeout(properties.getReadTimeoutMs(), TimeUnit.MILLISECONDS)
              .callTimeout(properties.getReadTimeoutMs(), TimeUnit.MILLISECONDS)
              .retryOnConnectionFailure(true)
              .build();
        }
      }
    }
    return client;
  }
}
