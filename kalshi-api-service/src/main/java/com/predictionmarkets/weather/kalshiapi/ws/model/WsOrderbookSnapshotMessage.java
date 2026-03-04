package com.predictionmarkets.weather.kalshiapi.ws.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record WsOrderbookSnapshotMessage(String type, Integer sid, Long seq, WsOrderbookSnapshotPayload msg) {
}
