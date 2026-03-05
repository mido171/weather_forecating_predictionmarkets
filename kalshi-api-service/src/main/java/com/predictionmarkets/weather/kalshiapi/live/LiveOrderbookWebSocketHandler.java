package com.predictionmarkets.weather.kalshiapi.live;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.socket.WebSocketHandler;
import org.springframework.web.reactive.socket.WebSocketMessage;
import org.springframework.web.reactive.socket.WebSocketSession;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

@Component
public class LiveOrderbookWebSocketHandler implements WebSocketHandler {

  private static final Logger log = LoggerFactory.getLogger(LiveOrderbookWebSocketHandler.class);

  private final LiveOrderbookStreamService streamService;
  private final ObjectMapper objectMapper;

  public LiveOrderbookWebSocketHandler(LiveOrderbookStreamService streamService,
                                       ObjectMapper objectMapper) {
    this.streamService = streamService;
    this.objectMapper = objectMapper;
  }

  @Override
  public Mono<Void> handle(WebSocketSession session) {
    Flux<WebSocketMessage> outbound = streamService.stream()
        .map(this::toJson)
        .map(session::textMessage);

    Mono<Void> send = session.send(outbound);
    Mono<Void> receive = session.receive().then();
    return Mono.when(send, receive)
        .doOnSubscribe(unused -> log.info("Live websocket client connected id={}", session.getId()))
        .doFinally(signal -> log.info("Live websocket client disconnected id={} signal={}", session.getId(), signal));
  }

  private String toJson(LiveOrderbookFrame frame) {
    try {
      return objectMapper.writeValueAsString(frame);
    } catch (JsonProcessingException ex) {
      log.warn("Failed to serialize live orderbook frame: {}", ex.toString());
      return "{\"error\":\"serialization_failure\"}";
    }
  }
}

