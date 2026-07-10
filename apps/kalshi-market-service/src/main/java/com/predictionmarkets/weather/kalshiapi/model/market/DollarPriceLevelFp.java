package com.predictionmarkets.weather.kalshiapi.model.market;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonFormat(shape = JsonFormat.Shape.ARRAY)
@JsonIgnoreProperties(ignoreUnknown = true)
public record DollarPriceLevelFp(String price, String quantity) {
}
