package com.predictionmarkets.weather.kalshiapi.config;

import com.predictionmarkets.weather.kalshiapi.http.KalshiAuthFilter;
import com.predictionmarkets.weather.kalshiapi.http.KalshiObservabilityFilter;
import io.netty.channel.ChannelOption;
import java.time.Duration;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.netty.http.client.HttpClient;

@Configuration
public class KalshiWebClientConfig {

  @Bean
  @Qualifier("kalshiPublicWebClient")
  public WebClient kalshiPublicWebClient(KalshiExecutionProperties properties,
                                         KalshiObservabilityFilter observabilityFilter) {
    return baseBuilder(properties, observabilityFilter).build();
  }

  @Bean
  @Qualifier("kalshiAuthWebClient")
  public WebClient kalshiAuthWebClient(KalshiExecutionProperties properties,
                                       KalshiObservabilityFilter observabilityFilter,
                                       KalshiAuthFilter authFilter) {
    return baseBuilder(properties, observabilityFilter)
        .filter(authFilter)
        .build();
  }

  private WebClient.Builder baseBuilder(KalshiExecutionProperties properties,
                                        KalshiObservabilityFilter observabilityFilter) {
    HttpClient httpClient = HttpClient.create()
        .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, properties.getTimeouts().getConnectTimeoutMs())
        .responseTimeout(Duration.ofMillis(properties.getTimeouts().getRequestTimeoutMs()));

    return WebClient.builder()
        .baseUrl(properties.resolvedRestBaseUrl())
        .clientConnector(new ReactorClientHttpConnector(httpClient))
        .defaultHeader(HttpHeaders.ACCEPT, MediaType.APPLICATION_JSON_VALUE)
        .filter(observabilityFilter);
  }
}
