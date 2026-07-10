package com.predictionmarkets.weather.models;

import java.math.BigDecimal;
import java.time.Instant;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.UniqueConstraint;

@Entity
@Table(
    schema = "wunderground_ml",
    name = "wunderground_station_observation_30m",
    uniqueConstraints = {
        @UniqueConstraint(columnNames = {"request_location_id", "obs_id", "valid_time_gmt"})
    }
)
public class WeatherComObservation {
  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  @Column(name = "id", nullable = false)
  private Long id;

  @Column(name = "api_call_id", nullable = false)
  private Long apiCallId;

  @Column(name = "request_location_id", nullable = false, length = 64)
  private String requestLocationId;

  @Column(name = "obs_id", nullable = false, length = 32)
  private String obsId;

  @Column(name = "obs_key", length = 32)
  private String obsKey;

  @Column(name = "obs_name", length = 255)
  private String obsName;

  @Column(name = "valid_time_gmt", nullable = false)
  private long validTimeGmt;

  @Column(name = "valid_time_utc", nullable = false)
  private Instant validTimeUtc;

  @Column(name = "day_ind", length = 1)
  private String dayInd;

  @Column(name = "temp")
  private Integer temp;

  @Column(name = "dew_pt")
  private Integer dewPt;

  @Column(name = "heat_index")
  private Integer heatIndex;

  @Column(name = "rh")
  private Integer rh;

  @Column(name = "pressure", precision = 10, scale = 4)
  private BigDecimal pressure;

  @Column(name = "pressure_tend")
  private Integer pressureTend;

  @Column(name = "pressure_desc", length = 64)
  private String pressureDesc;

  @Column(name = "vis", precision = 10, scale = 4)
  private BigDecimal vis;

  @Column(name = "wc")
  private Integer wc;

  @Column(name = "wdir")
  private Integer wdir;

  @Column(name = "wdir_cardinal", length = 16)
  private String wdirCardinal;

  @Column(name = "gust")
  private Integer gust;

  @Column(name = "wspd")
  private Integer wspd;

  @Column(name = "wx_phrase", length = 255)
  private String wxPhrase;

  @Column(name = "wx_icon")
  private Integer wxIcon;

  @Column(name = "icon_extd")
  private Integer iconExtd;

  @Column(name = "precip_total", precision = 10, scale = 4)
  private BigDecimal precipTotal;

  @Column(name = "precip_hrly", precision = 10, scale = 4)
  private BigDecimal precipHrly;

  @Column(name = "snow_hrly", precision = 10, scale = 4)
  private BigDecimal snowHrly;

  @Column(name = "max_temp")
  private Integer maxTemp;

  @Column(name = "min_temp")
  private Integer minTemp;

  @Column(name = "uv_desc", length = 64)
  private String uvDesc;

  @Column(name = "uv_index")
  private Integer uvIndex;

  @Column(name = "feels_like")
  private Integer feelsLike;

  @Column(name = "clds", length = 64)
  private String clds;

  @Column(name = "qualifier", length = 64)
  private String qualifier;

  @Column(name = "qualifier_svrty", length = 64)
  private String qualifierSvrty;

  @Column(name = "blunt_phrase", length = 255)
  private String bluntPhrase;

  @Column(name = "terse_phrase", length = 255)
  private String tersePhrase;

  @Column(name = "observation_class", length = 32)
  private String observationClass;

  @Column(name = "water_temp")
  private Integer waterTemp;

  @Column(name = "primary_wave_period", precision = 10, scale = 4)
  private BigDecimal primaryWavePeriod;

  @Column(name = "primary_wave_height", precision = 10, scale = 4)
  private BigDecimal primaryWaveHeight;

  @Column(name = "primary_swell_period", precision = 10, scale = 4)
  private BigDecimal primarySwellPeriod;

  @Column(name = "primary_swell_height", precision = 10, scale = 4)
  private BigDecimal primarySwellHeight;

  @Column(name = "primary_swell_direction")
  private Integer primarySwellDirection;

  @Column(name = "secondary_swell_period", precision = 10, scale = 4)
  private BigDecimal secondarySwellPeriod;

  @Column(name = "secondary_swell_height", precision = 10, scale = 4)
  private BigDecimal secondarySwellHeight;

  @Column(name = "secondary_swell_direction")
  private Integer secondarySwellDirection;

  @Column(name = "created_at_utc", nullable = false)
  private Instant createdAtUtc;

  @Column(name = "updated_at_utc", nullable = false)
  private Instant updatedAtUtc;

  public Long getId() {
    return id;
  }

  public Long getApiCallId() {
    return apiCallId;
  }

  public String getRequestLocationId() {
    return requestLocationId;
  }

  public String getObsId() {
    return obsId;
  }

  public String getObsName() {
    return obsName;
  }

  public long getValidTimeGmt() {
    return validTimeGmt;
  }

  public Instant getValidTimeUtc() {
    return validTimeUtc;
  }

  public Integer getTemp() {
    return temp;
  }

  public Integer getDewPt() {
    return dewPt;
  }

  public Integer getRh() {
    return rh;
  }

  public BigDecimal getPressure() {
    return pressure;
  }

  public Integer getWspd() {
    return wspd;
  }

  public String getWxPhrase() {
    return wxPhrase;
  }

  public Instant getCreatedAtUtc() {
    return createdAtUtc;
  }

  public Instant getUpdatedAtUtc() {
    return updatedAtUtc;
  }
}
