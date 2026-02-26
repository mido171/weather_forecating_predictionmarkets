package com.predictionmarkets.weather.weathercom.client.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.math.BigDecimal;

@JsonIgnoreProperties(ignoreUnknown = true)
public class WeatherComObservationPayload {
  @JsonProperty("obs_id")
  private String obsId;

  @JsonProperty("key")
  private String key;

  @JsonProperty("obs_name")
  private String obsName;

  @JsonProperty("valid_time_gmt")
  private Long validTimeGmt;

  @JsonProperty("day_ind")
  private String dayInd;

  @JsonProperty("temp")
  private Integer temp;

  @JsonProperty("dewPt")
  private Integer dewPt;

  @JsonProperty("heat_index")
  private Integer heatIndex;

  @JsonProperty("rh")
  private Integer rh;

  @JsonProperty("pressure")
  private BigDecimal pressure;

  @JsonProperty("pressure_tend")
  private Integer pressureTend;

  @JsonProperty("pressure_desc")
  private String pressureDesc;

  @JsonProperty("vis")
  private BigDecimal vis;

  @JsonProperty("wc")
  private Integer wc;

  @JsonProperty("wdir")
  private Integer wdir;

  @JsonProperty("wdir_cardinal")
  private String wdirCardinal;

  @JsonProperty("gust")
  private Integer gust;

  @JsonProperty("wspd")
  private Integer wspd;

  @JsonProperty("wx_phrase")
  private String wxPhrase;

  @JsonProperty("wx_icon")
  private Integer wxIcon;

  @JsonProperty("icon_extd")
  private Integer iconExtd;

  @JsonProperty("precip_total")
  private BigDecimal precipTotal;

  @JsonProperty("precip_hrly")
  private BigDecimal precipHrly;

  @JsonProperty("snow_hrly")
  private BigDecimal snowHrly;

  @JsonProperty("max_temp")
  private Integer maxTemp;

  @JsonProperty("min_temp")
  private Integer minTemp;

  @JsonProperty("uv_desc")
  private String uvDesc;

  @JsonProperty("uv_index")
  private Integer uvIndex;

  @JsonProperty("feels_like")
  private Integer feelsLike;

  @JsonProperty("clds")
  private String clds;

  @JsonProperty("qualifier")
  private String qualifier;

  @JsonProperty("qualifier_svrty")
  private String qualifierSvrty;

  @JsonProperty("blunt_phrase")
  private String bluntPhrase;

  @JsonProperty("terse_phrase")
  private String tersePhrase;

  @JsonProperty("class")
  private String observationClass;

  @JsonProperty("water_temp")
  private Integer waterTemp;

  @JsonProperty("primary_wave_period")
  private BigDecimal primaryWavePeriod;

  @JsonProperty("primary_wave_height")
  private BigDecimal primaryWaveHeight;

  @JsonProperty("primary_swell_period")
  private BigDecimal primarySwellPeriod;

  @JsonProperty("primary_swell_height")
  private BigDecimal primarySwellHeight;

  @JsonProperty("primary_swell_direction")
  private Integer primarySwellDirection;

  @JsonProperty("secondary_swell_period")
  private BigDecimal secondarySwellPeriod;

  @JsonProperty("secondary_swell_height")
  private BigDecimal secondarySwellHeight;

  @JsonProperty("secondary_swell_direction")
  private Integer secondarySwellDirection;

  public String getObsId() {
    return obsId;
  }

  public void setObsId(String obsId) {
    this.obsId = obsId;
  }

  public String getKey() {
    return key;
  }

  public void setKey(String key) {
    this.key = key;
  }

  public String getObsName() {
    return obsName;
  }

  public void setObsName(String obsName) {
    this.obsName = obsName;
  }

  public Long getValidTimeGmt() {
    return validTimeGmt;
  }

  public void setValidTimeGmt(Long validTimeGmt) {
    this.validTimeGmt = validTimeGmt;
  }

  public String getDayInd() {
    return dayInd;
  }

  public void setDayInd(String dayInd) {
    this.dayInd = dayInd;
  }

  public Integer getTemp() {
    return temp;
  }

  public void setTemp(Integer temp) {
    this.temp = temp;
  }

  public Integer getDewPt() {
    return dewPt;
  }

  public void setDewPt(Integer dewPt) {
    this.dewPt = dewPt;
  }

  public Integer getHeatIndex() {
    return heatIndex;
  }

  public void setHeatIndex(Integer heatIndex) {
    this.heatIndex = heatIndex;
  }

  public Integer getRh() {
    return rh;
  }

  public void setRh(Integer rh) {
    this.rh = rh;
  }

  public BigDecimal getPressure() {
    return pressure;
  }

  public void setPressure(BigDecimal pressure) {
    this.pressure = pressure;
  }

  public Integer getPressureTend() {
    return pressureTend;
  }

  public void setPressureTend(Integer pressureTend) {
    this.pressureTend = pressureTend;
  }

  public String getPressureDesc() {
    return pressureDesc;
  }

  public void setPressureDesc(String pressureDesc) {
    this.pressureDesc = pressureDesc;
  }

  public BigDecimal getVis() {
    return vis;
  }

  public void setVis(BigDecimal vis) {
    this.vis = vis;
  }

  public Integer getWc() {
    return wc;
  }

  public void setWc(Integer wc) {
    this.wc = wc;
  }

  public Integer getWdir() {
    return wdir;
  }

  public void setWdir(Integer wdir) {
    this.wdir = wdir;
  }

  public String getWdirCardinal() {
    return wdirCardinal;
  }

  public void setWdirCardinal(String wdirCardinal) {
    this.wdirCardinal = wdirCardinal;
  }

  public Integer getGust() {
    return gust;
  }

  public void setGust(Integer gust) {
    this.gust = gust;
  }

  public Integer getWspd() {
    return wspd;
  }

  public void setWspd(Integer wspd) {
    this.wspd = wspd;
  }

  public String getWxPhrase() {
    return wxPhrase;
  }

  public void setWxPhrase(String wxPhrase) {
    this.wxPhrase = wxPhrase;
  }

  public Integer getWxIcon() {
    return wxIcon;
  }

  public void setWxIcon(Integer wxIcon) {
    this.wxIcon = wxIcon;
  }

  public Integer getIconExtd() {
    return iconExtd;
  }

  public void setIconExtd(Integer iconExtd) {
    this.iconExtd = iconExtd;
  }

  public BigDecimal getPrecipTotal() {
    return precipTotal;
  }

  public void setPrecipTotal(BigDecimal precipTotal) {
    this.precipTotal = precipTotal;
  }

  public BigDecimal getPrecipHrly() {
    return precipHrly;
  }

  public void setPrecipHrly(BigDecimal precipHrly) {
    this.precipHrly = precipHrly;
  }

  public BigDecimal getSnowHrly() {
    return snowHrly;
  }

  public void setSnowHrly(BigDecimal snowHrly) {
    this.snowHrly = snowHrly;
  }

  public Integer getMaxTemp() {
    return maxTemp;
  }

  public void setMaxTemp(Integer maxTemp) {
    this.maxTemp = maxTemp;
  }

  public Integer getMinTemp() {
    return minTemp;
  }

  public void setMinTemp(Integer minTemp) {
    this.minTemp = minTemp;
  }

  public String getUvDesc() {
    return uvDesc;
  }

  public void setUvDesc(String uvDesc) {
    this.uvDesc = uvDesc;
  }

  public Integer getUvIndex() {
    return uvIndex;
  }

  public void setUvIndex(Integer uvIndex) {
    this.uvIndex = uvIndex;
  }

  public Integer getFeelsLike() {
    return feelsLike;
  }

  public void setFeelsLike(Integer feelsLike) {
    this.feelsLike = feelsLike;
  }

  public String getClds() {
    return clds;
  }

  public void setClds(String clds) {
    this.clds = clds;
  }

  public String getQualifier() {
    return qualifier;
  }

  public void setQualifier(String qualifier) {
    this.qualifier = qualifier;
  }

  public String getQualifierSvrty() {
    return qualifierSvrty;
  }

  public void setQualifierSvrty(String qualifierSvrty) {
    this.qualifierSvrty = qualifierSvrty;
  }

  public String getBluntPhrase() {
    return bluntPhrase;
  }

  public void setBluntPhrase(String bluntPhrase) {
    this.bluntPhrase = bluntPhrase;
  }

  public String getTersePhrase() {
    return tersePhrase;
  }

  public void setTersePhrase(String tersePhrase) {
    this.tersePhrase = tersePhrase;
  }

  public String getObservationClass() {
    return observationClass;
  }

  public void setObservationClass(String observationClass) {
    this.observationClass = observationClass;
  }

  public Integer getWaterTemp() {
    return waterTemp;
  }

  public void setWaterTemp(Integer waterTemp) {
    this.waterTemp = waterTemp;
  }

  public BigDecimal getPrimaryWavePeriod() {
    return primaryWavePeriod;
  }

  public void setPrimaryWavePeriod(BigDecimal primaryWavePeriod) {
    this.primaryWavePeriod = primaryWavePeriod;
  }

  public BigDecimal getPrimaryWaveHeight() {
    return primaryWaveHeight;
  }

  public void setPrimaryWaveHeight(BigDecimal primaryWaveHeight) {
    this.primaryWaveHeight = primaryWaveHeight;
  }

  public BigDecimal getPrimarySwellPeriod() {
    return primarySwellPeriod;
  }

  public void setPrimarySwellPeriod(BigDecimal primarySwellPeriod) {
    this.primarySwellPeriod = primarySwellPeriod;
  }

  public BigDecimal getPrimarySwellHeight() {
    return primarySwellHeight;
  }

  public void setPrimarySwellHeight(BigDecimal primarySwellHeight) {
    this.primarySwellHeight = primarySwellHeight;
  }

  public Integer getPrimarySwellDirection() {
    return primarySwellDirection;
  }

  public void setPrimarySwellDirection(Integer primarySwellDirection) {
    this.primarySwellDirection = primarySwellDirection;
  }

  public BigDecimal getSecondarySwellPeriod() {
    return secondarySwellPeriod;
  }

  public void setSecondarySwellPeriod(BigDecimal secondarySwellPeriod) {
    this.secondarySwellPeriod = secondarySwellPeriod;
  }

  public BigDecimal getSecondarySwellHeight() {
    return secondarySwellHeight;
  }

  public void setSecondarySwellHeight(BigDecimal secondarySwellHeight) {
    this.secondarySwellHeight = secondarySwellHeight;
  }

  public Integer getSecondarySwellDirection() {
    return secondarySwellDirection;
  }

  public void setSecondarySwellDirection(Integer secondarySwellDirection) {
    this.secondarySwellDirection = secondarySwellDirection;
  }
}

