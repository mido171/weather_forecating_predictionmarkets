package com.predictionmarkets.weather.kalshiapi.model.market;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

class EventResponseDeserializationTest {

  private final ObjectMapper objectMapper = new ObjectMapper();

  @Test
  void parsesTopLevelMarketsFromEventResponse() throws Exception {
    String json = """
        {
          "event": {
            "event_ticker": "KXHIGHNY-26MAR05",
            "title": "Highest temperature in NYC on Mar 5, 2026?",
            "sub_title": "On Mar 5, 2026"
          },
          "markets": [
            {
              "ticker": "KXHIGHNY-26MAR05-T41",
              "event_ticker": "KXHIGHNY-26MAR05",
              "subtitle": "40 deg or below",
              "status": "active"
            }
          ]
        }
        """;

    EventResponse response = objectMapper.readValue(json, EventResponse.class);

    assertNotNull(response);
    assertNotNull(response.event());
    assertEquals("KXHIGHNY-26MAR05", response.event().eventTicker());
    assertNotNull(response.markets());
    assertEquals(1, response.markets().size());
    assertEquals("KXHIGHNY-26MAR05-T41", response.markets().get(0).ticker());
  }
}

