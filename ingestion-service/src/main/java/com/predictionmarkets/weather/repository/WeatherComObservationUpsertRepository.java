package com.predictionmarkets.weather.repository;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.Arrays;
import java.util.List;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.jdbc.core.namedparam.SqlParameterSource;
import org.springframework.stereotype.Repository;

@Repository
public class WeatherComObservationUpsertRepository {
  private static final String UPSERT_SQL = """
      INSERT INTO wunderground_ml.wunderground_station_observation_30m (
        api_call_id,
        request_location_id,
        obs_id,
        obs_key,
        obs_name,
        valid_time_gmt,
        valid_time_utc,
        day_ind,
        temp,
        dew_pt,
        heat_index,
        rh,
        pressure,
        pressure_tend,
        pressure_desc,
        vis,
        wc,
        wdir,
        wdir_cardinal,
        gust,
        wspd,
        wx_phrase,
        wx_icon,
        icon_extd,
        precip_total,
        precip_hrly,
        snow_hrly,
        max_temp,
        min_temp,
        uv_desc,
        uv_index,
        feels_like,
        clds,
        qualifier,
        qualifier_svrty,
        blunt_phrase,
        terse_phrase,
        observation_class,
        water_temp,
        primary_wave_period,
        primary_wave_height,
        primary_swell_period,
        primary_swell_height,
        primary_swell_direction,
        secondary_swell_period,
        secondary_swell_height,
        secondary_swell_direction,
        created_at_utc,
        updated_at_utc
      ) VALUES (
        :apiCallId,
        :requestLocationId,
        :obsId,
        :obsKey,
        :obsName,
        :validTimeGmt,
        :validTimeUtc,
        :dayInd,
        :temp,
        :dewPt,
        :heatIndex,
        :rh,
        :pressure,
        :pressureTend,
        :pressureDesc,
        :vis,
        :wc,
        :wdir,
        :wdirCardinal,
        :gust,
        :wspd,
        :wxPhrase,
        :wxIcon,
        :iconExtd,
        :precipTotal,
        :precipHrly,
        :snowHrly,
        :maxTemp,
        :minTemp,
        :uvDesc,
        :uvIndex,
        :feelsLike,
        :clds,
        :qualifier,
        :qualifierSvrty,
        :bluntPhrase,
        :tersePhrase,
        :observationClass,
        :waterTemp,
        :primaryWavePeriod,
        :primaryWaveHeight,
        :primarySwellPeriod,
        :primarySwellHeight,
        :primarySwellDirection,
        :secondarySwellPeriod,
        :secondarySwellHeight,
        :secondarySwellDirection,
        :createdAtUtc,
        :updatedAtUtc
      )
      ON DUPLICATE KEY UPDATE
        api_call_id = VALUES(api_call_id),
        obs_key = VALUES(obs_key),
        obs_name = VALUES(obs_name),
        valid_time_utc = VALUES(valid_time_utc),
        day_ind = VALUES(day_ind),
        temp = VALUES(temp),
        dew_pt = VALUES(dew_pt),
        heat_index = VALUES(heat_index),
        rh = VALUES(rh),
        pressure = VALUES(pressure),
        pressure_tend = VALUES(pressure_tend),
        pressure_desc = VALUES(pressure_desc),
        vis = VALUES(vis),
        wc = VALUES(wc),
        wdir = VALUES(wdir),
        wdir_cardinal = VALUES(wdir_cardinal),
        gust = VALUES(gust),
        wspd = VALUES(wspd),
        wx_phrase = VALUES(wx_phrase),
        wx_icon = VALUES(wx_icon),
        icon_extd = VALUES(icon_extd),
        precip_total = VALUES(precip_total),
        precip_hrly = VALUES(precip_hrly),
        snow_hrly = VALUES(snow_hrly),
        max_temp = VALUES(max_temp),
        min_temp = VALUES(min_temp),
        uv_desc = VALUES(uv_desc),
        uv_index = VALUES(uv_index),
        feels_like = VALUES(feels_like),
        clds = VALUES(clds),
        qualifier = VALUES(qualifier),
        qualifier_svrty = VALUES(qualifier_svrty),
        blunt_phrase = VALUES(blunt_phrase),
        terse_phrase = VALUES(terse_phrase),
        observation_class = VALUES(observation_class),
        water_temp = VALUES(water_temp),
        primary_wave_period = VALUES(primary_wave_period),
        primary_wave_height = VALUES(primary_wave_height),
        primary_swell_period = VALUES(primary_swell_period),
        primary_swell_height = VALUES(primary_swell_height),
        primary_swell_direction = VALUES(primary_swell_direction),
        secondary_swell_period = VALUES(secondary_swell_period),
        secondary_swell_height = VALUES(secondary_swell_height),
        secondary_swell_direction = VALUES(secondary_swell_direction),
        updated_at_utc = VALUES(updated_at_utc)
      """;

  private final NamedParameterJdbcTemplate jdbcTemplate;

  public WeatherComObservationUpsertRepository(NamedParameterJdbcTemplate jdbcTemplate) {
    this.jdbcTemplate = jdbcTemplate;
  }

  public int upsertAll(List<UpsertRow> rows, int batchSize) {
    if (rows == null || rows.isEmpty()) {
      return 0;
    }
    int effectiveBatchSize = Math.max(1, batchSize);
    int updated = 0;
    for (int start = 0; start < rows.size(); start += effectiveBatchSize) {
      int end = Math.min(rows.size(), start + effectiveBatchSize);
      SqlParameterSource[] batch = rows.subList(start, end).stream()
          .map(WeatherComObservationUpsertRepository::toParams)
          .toArray(SqlParameterSource[]::new);
      updated += Arrays.stream(jdbcTemplate.batchUpdate(UPSERT_SQL, batch)).sum();
    }
    return updated;
  }

  public void deleteAll() {
    jdbcTemplate.update("DELETE FROM wunderground_ml.wunderground_station_observation_30m",
        new MapSqlParameterSource());
  }

  private static SqlParameterSource toParams(UpsertRow row) {
    return new MapSqlParameterSource()
        .addValue("apiCallId", row.apiCallId)
        .addValue("requestLocationId", row.requestLocationId)
        .addValue("obsId", row.obsId)
        .addValue("obsKey", row.obsKey)
        .addValue("obsName", row.obsName)
        .addValue("validTimeGmt", row.validTimeGmt)
        .addValue("validTimeUtc", row.validTimeUtc)
        .addValue("dayInd", row.dayInd)
        .addValue("temp", row.temp)
        .addValue("dewPt", row.dewPt)
        .addValue("heatIndex", row.heatIndex)
        .addValue("rh", row.rh)
        .addValue("pressure", row.pressure)
        .addValue("pressureTend", row.pressureTend)
        .addValue("pressureDesc", row.pressureDesc)
        .addValue("vis", row.vis)
        .addValue("wc", row.wc)
        .addValue("wdir", row.wdir)
        .addValue("wdirCardinal", row.wdirCardinal)
        .addValue("gust", row.gust)
        .addValue("wspd", row.wspd)
        .addValue("wxPhrase", row.wxPhrase)
        .addValue("wxIcon", row.wxIcon)
        .addValue("iconExtd", row.iconExtd)
        .addValue("precipTotal", row.precipTotal)
        .addValue("precipHrly", row.precipHrly)
        .addValue("snowHrly", row.snowHrly)
        .addValue("maxTemp", row.maxTemp)
        .addValue("minTemp", row.minTemp)
        .addValue("uvDesc", row.uvDesc)
        .addValue("uvIndex", row.uvIndex)
        .addValue("feelsLike", row.feelsLike)
        .addValue("clds", row.clds)
        .addValue("qualifier", row.qualifier)
        .addValue("qualifierSvrty", row.qualifierSvrty)
        .addValue("bluntPhrase", row.bluntPhrase)
        .addValue("tersePhrase", row.tersePhrase)
        .addValue("observationClass", row.observationClass)
        .addValue("waterTemp", row.waterTemp)
        .addValue("primaryWavePeriod", row.primaryWavePeriod)
        .addValue("primaryWaveHeight", row.primaryWaveHeight)
        .addValue("primarySwellPeriod", row.primarySwellPeriod)
        .addValue("primarySwellHeight", row.primarySwellHeight)
        .addValue("primarySwellDirection", row.primarySwellDirection)
        .addValue("secondarySwellPeriod", row.secondarySwellPeriod)
        .addValue("secondarySwellHeight", row.secondarySwellHeight)
        .addValue("secondarySwellDirection", row.secondarySwellDirection)
        .addValue("createdAtUtc", row.createdAtUtc)
        .addValue("updatedAtUtc", row.updatedAtUtc);
  }

  public static final class UpsertRow {
    private final Long apiCallId;
    private final String requestLocationId;
    private final String obsId;
    private final String obsKey;
    private final String obsName;
    private final long validTimeGmt;
    private final Instant validTimeUtc;
    private final String dayInd;
    private final Integer temp;
    private final Integer dewPt;
    private final Integer heatIndex;
    private final Integer rh;
    private final BigDecimal pressure;
    private final Integer pressureTend;
    private final String pressureDesc;
    private final BigDecimal vis;
    private final Integer wc;
    private final Integer wdir;
    private final String wdirCardinal;
    private final Integer gust;
    private final Integer wspd;
    private final String wxPhrase;
    private final Integer wxIcon;
    private final Integer iconExtd;
    private final BigDecimal precipTotal;
    private final BigDecimal precipHrly;
    private final BigDecimal snowHrly;
    private final Integer maxTemp;
    private final Integer minTemp;
    private final String uvDesc;
    private final Integer uvIndex;
    private final Integer feelsLike;
    private final String clds;
    private final String qualifier;
    private final String qualifierSvrty;
    private final String bluntPhrase;
    private final String tersePhrase;
    private final String observationClass;
    private final Integer waterTemp;
    private final BigDecimal primaryWavePeriod;
    private final BigDecimal primaryWaveHeight;
    private final BigDecimal primarySwellPeriod;
    private final BigDecimal primarySwellHeight;
    private final Integer primarySwellDirection;
    private final BigDecimal secondarySwellPeriod;
    private final BigDecimal secondarySwellHeight;
    private final Integer secondarySwellDirection;
    private final Instant createdAtUtc;
    private final Instant updatedAtUtc;

    public UpsertRow(Long apiCallId,
                     String requestLocationId,
                     String obsId,
                     String obsKey,
                     String obsName,
                     long validTimeGmt,
                     Instant validTimeUtc,
                     String dayInd,
                     Integer temp,
                     Integer dewPt,
                     Integer heatIndex,
                     Integer rh,
                     BigDecimal pressure,
                     Integer pressureTend,
                     String pressureDesc,
                     BigDecimal vis,
                     Integer wc,
                     Integer wdir,
                     String wdirCardinal,
                     Integer gust,
                     Integer wspd,
                     String wxPhrase,
                     Integer wxIcon,
                     Integer iconExtd,
                     BigDecimal precipTotal,
                     BigDecimal precipHrly,
                     BigDecimal snowHrly,
                     Integer maxTemp,
                     Integer minTemp,
                     String uvDesc,
                     Integer uvIndex,
                     Integer feelsLike,
                     String clds,
                     String qualifier,
                     String qualifierSvrty,
                     String bluntPhrase,
                     String tersePhrase,
                     String observationClass,
                     Integer waterTemp,
                     BigDecimal primaryWavePeriod,
                     BigDecimal primaryWaveHeight,
                     BigDecimal primarySwellPeriod,
                     BigDecimal primarySwellHeight,
                     Integer primarySwellDirection,
                     BigDecimal secondarySwellPeriod,
                     BigDecimal secondarySwellHeight,
                     Integer secondarySwellDirection,
                     Instant createdAtUtc,
                     Instant updatedAtUtc) {
      this.apiCallId = apiCallId;
      this.requestLocationId = requestLocationId;
      this.obsId = obsId;
      this.obsKey = obsKey;
      this.obsName = obsName;
      this.validTimeGmt = validTimeGmt;
      this.validTimeUtc = validTimeUtc;
      this.dayInd = dayInd;
      this.temp = temp;
      this.dewPt = dewPt;
      this.heatIndex = heatIndex;
      this.rh = rh;
      this.pressure = pressure;
      this.pressureTend = pressureTend;
      this.pressureDesc = pressureDesc;
      this.vis = vis;
      this.wc = wc;
      this.wdir = wdir;
      this.wdirCardinal = wdirCardinal;
      this.gust = gust;
      this.wspd = wspd;
      this.wxPhrase = wxPhrase;
      this.wxIcon = wxIcon;
      this.iconExtd = iconExtd;
      this.precipTotal = precipTotal;
      this.precipHrly = precipHrly;
      this.snowHrly = snowHrly;
      this.maxTemp = maxTemp;
      this.minTemp = minTemp;
      this.uvDesc = uvDesc;
      this.uvIndex = uvIndex;
      this.feelsLike = feelsLike;
      this.clds = clds;
      this.qualifier = qualifier;
      this.qualifierSvrty = qualifierSvrty;
      this.bluntPhrase = bluntPhrase;
      this.tersePhrase = tersePhrase;
      this.observationClass = observationClass;
      this.waterTemp = waterTemp;
      this.primaryWavePeriod = primaryWavePeriod;
      this.primaryWaveHeight = primaryWaveHeight;
      this.primarySwellPeriod = primarySwellPeriod;
      this.primarySwellHeight = primarySwellHeight;
      this.primarySwellDirection = primarySwellDirection;
      this.secondarySwellPeriod = secondarySwellPeriod;
      this.secondarySwellHeight = secondarySwellHeight;
      this.secondarySwellDirection = secondarySwellDirection;
      this.createdAtUtc = createdAtUtc;
      this.updatedAtUtc = updatedAtUtc;
    }
  }
}
