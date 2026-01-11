package com.predictionmarkets.weather.kalshi;

import com.predictionmarkets.weather.models.KalshiSeries;
import com.predictionmarkets.weather.models.StationMappingStatus;
import com.predictionmarkets.weather.models.StationOverride;
import com.predictionmarkets.weather.models.StationRegistry;
import com.predictionmarkets.weather.repository.KalshiSeriesRepository;
import com.predictionmarkets.weather.repository.StationOverrideRepository;
import com.predictionmarkets.weather.repository.StationRegistryRepository;
import java.time.Instant;
import java.util.Locale;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class KalshiSeriesResolver {
  private final KalshiClient kalshiClient;
  private final SettlementSourceParser settlementSourceParser;
  private final KalshiSeriesRepository kalshiSeriesRepository;
  private final StationRegistryRepository stationRegistryRepository;
  private final StationOverrideRepository stationOverrideRepository;

  public KalshiSeriesResolver(
      KalshiClient kalshiClient,
      SettlementSourceParser settlementSourceParser,
      KalshiSeriesRepository kalshiSeriesRepository,
      StationRegistryRepository stationRegistryRepository,
      StationOverrideRepository stationOverrideRepository) {
    this.kalshiClient = kalshiClient;
    this.settlementSourceParser = settlementSourceParser;
    this.kalshiSeriesRepository = kalshiSeriesRepository;
    this.stationRegistryRepository = stationRegistryRepository;
    this.stationOverrideRepository = stationOverrideRepository;
  }

  @Transactional
  public StationRegistry resolveAndUpsert(String seriesTicker) {
    KalshiSeriesPayload payload = kalshiClient.fetchSeries(seriesTicker);
    SettlementSource settlementSource = settlementSourceParser.parse(payload.settlementSourceUrl());
    String issuedby = settlementSource.issuedby();
    StationOverride override = stationOverrideRepository.findById(issuedby).orElse(null);

    String stationId = resolveStationId(issuedby, override);
    String zoneId = resolveZoneId(stationId, override);
    int standardOffsetMinutes = StationZoneDefaults.standardOffsetMinutes(zoneId);
    Instant now = Instant.now();

    KalshiSeries series = kalshiSeriesRepository.findById(payload.seriesTicker())
        .orElseGet(KalshiSeries::new);
    series.setSeriesTicker(payload.seriesTicker());
    series.setTitle(payload.title());
    series.setCategory(payload.category());
    series.setSettlementSourceName(payload.settlementSourceName());
    series.setSettlementSourceUrl(payload.settlementSourceUrl());
    series.setContractTermsUrl(payload.contractTermsUrl());
    series.setContractUrl(payload.contractUrl());
    series.setRetrievedAtUtc(now);
    series.setRawPayloadHash(payload.rawPayloadHash());
    series.setRawJson(payload.rawJson());
    kalshiSeriesRepository.save(series);

    StationRegistry registry = stationRegistryRepository.findById(stationId)
        .orElseGet(StationRegistry::new);
    if (registry.getCreatedAtUtc() == null) {
      registry.setCreatedAtUtc(now);
    }
    registry.setStationId(stationId);
    registry.setIssuedby(issuedby);
    registry.setWfoSite(settlementSource.wfoSite());
    registry.setSeriesTicker(payload.seriesTicker());
    registry.setZoneId(zoneId);
    registry.setStandardOffsetMinutes(standardOffsetMinutes);
    registry.setMappingStatus(StationMappingStatus.AUTO_OK);
    registry.setUpdatedAtUtc(now);
    return stationRegistryRepository.save(registry);
  }

  private String resolveStationId(String issuedby, StationOverride override) {
    if (override != null) {
      return normalize(override.getStationIdOverride());
    }
    String normalizedIssuedby = normalize(issuedby);
    if (normalizedIssuedby.length() == 3) {
      return "K" + normalizedIssuedby;
    }
    if (normalizedIssuedby.length() == 4 && normalizedIssuedby.startsWith("K")) {
      return normalizedIssuedby;
    }
    throw new IllegalArgumentException("Unexpected issuedby format: " + issuedby);
  }

  private String resolveZoneId(String stationId, StationOverride override) {
    if (override != null) {
      String overrideZone = override.getZoneIdOverride();
      if (overrideZone == null || overrideZone.isBlank()) {
        throw new IllegalArgumentException("Station override missing zoneIdOverride");
      }
      return overrideZone.trim();
    }
    return StationZoneDefaults.zoneIdFor(stationId)
        .orElseThrow(() -> new IllegalStateException("No zone mapping for station " + stationId));
  }

  private String normalize(String value) {
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("Required value missing");
    }
    return value.trim().toUpperCase(Locale.ROOT);
  }
}
