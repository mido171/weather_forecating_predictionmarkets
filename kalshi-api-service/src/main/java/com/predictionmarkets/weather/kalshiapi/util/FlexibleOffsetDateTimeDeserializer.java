package com.predictionmarkets.weather.kalshiapi.util;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import java.io.IOException;
import java.time.OffsetDateTime;

public class FlexibleOffsetDateTimeDeserializer extends JsonDeserializer<OffsetDateTime> {

  @Override
  public OffsetDateTime deserialize(JsonParser parser, DeserializationContext context) throws IOException {
    String text = parser.getValueAsString();
    if (text == null || text.isBlank()) {
      return null;
    }
    return OffsetDateTime.parse(text.trim());
  }
}
