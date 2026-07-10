package com.predictionmarkets.weather;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.config.CliSettlementIngestionProperties;
import com.predictionmarkets.weather.config.EvomiProxyProperties;
import com.predictionmarkets.weather.config.PipelineProperties;
import com.predictionmarkets.weather.gribstream.GribstreamClient;
import com.predictionmarkets.weather.gribstream.GribstreamExecutorProperties;
import com.predictionmarkets.weather.gribstream.GribstreamForecastClient;
import com.predictionmarkets.weather.gribstream.GribstreamProperties;
import com.predictionmarkets.weather.gribstream.GribstreamVariableIngestProperties;
import com.predictionmarkets.weather.http.HttpClientConfig;
import com.predictionmarkets.weather.klga.iemmos.IemMosBackfillProperties;
import com.predictionmarkets.weather.pilot.config.PilotIngestionProperties;
import com.predictionmarkets.weather.weathercom.config.WeatherComProperties;
import org.junit.jupiter.api.Test;

class RuntimeSafetyDefaultsTest {

  @Test
  void networkJobsAreDisabledAndSingleThreadedByDefault() {
    CliSettlementIngestionProperties settlement = new CliSettlementIngestionProperties();
    GribstreamVariableIngestProperties gribstreamIngest = new GribstreamVariableIngestProperties();
    IemMosBackfillProperties iemMos = new IemMosBackfillProperties();
    PilotIngestionProperties pilot = new PilotIngestionProperties();
    WeatherComProperties weatherCom = new WeatherComProperties();

    assertThat(settlement.isIngestEnabled()).isFalse();
    assertThat(settlement.getSourceFetchThreads()).isOne();
    assertThat(gribstreamIngest.isEnabled()).isFalse();
    assertThat(new GribstreamExecutorProperties().getThreadCount()).isOne();
    assertThat(new PipelineProperties().getThreadCount()).isOne();
    assertThat(iemMos.isEnabled()).isFalse();
    assertThat(iemMos.getThreads()).isOne();
    assertThat(iemMos.getMaxAttempts()).isOne();
    assertThat(pilot.isEnabled()).isFalse();
    assertThat(pilot.getJobs().isBootstrapEnabled()).isFalse();
    assertThat(pilot.getJobs().isLightweightEnabled()).isFalse();
    assertThat(pilot.getJobs().isHeavyEnabled()).isFalse();
    assertThat(pilot.getJobs().isSnapshotEnabled()).isFalse();
    assertThat(weatherCom.getIngestion().isEnabled()).isFalse();
    assertThat(weatherCom.getIngestion().getThreadPoolSize()).isOne();
    assertThat(weatherCom.getIngestion().getMaxRetries()).isOne();
    assertThat(weatherCom.getIngestion().isStoreResponseBody()).isFalse();
  }

  @Test
  void gribstreamClientsStartWithoutCredentialsButRejectRequestsBeforeNetworkAccess() {
    GribstreamProperties properties = new GribstreamProperties();
    EvomiProxyProperties proxyProperties = new EvomiProxyProperties();
    ObjectMapper objectMapper = new ObjectMapper();

    GribstreamClient historyClient = new GribstreamClient(
        properties,
        proxyProperties,
        objectMapper,
        new HttpClientConfig().httpClientSettings());
    GribstreamForecastClient forecastClient = new GribstreamForecastClient(
        properties,
        proxyProperties,
        objectMapper);

    assertThatThrownBy(() -> historyClient.fetchHistory("gfs", null))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("GRIBSTREAM_API_TOKEN");
    assertThatThrownBy(() -> forecastClient.fetchForecastsRaw("gfs", null))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("GRIBSTREAM_API_TOKEN");
    assertThat(new HttpClientConfig().httpClientSettings().retryPolicy().maxAttempts()).isEqualTo(2);
  }
}
