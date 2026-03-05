package com.predictionmarkets.weather.kalshiapi.live;

import com.predictionmarkets.weather.kalshiapi.config.LiveTradingProperties;
import java.util.Map;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.util.StringUtils;
import org.springframework.web.reactive.HandlerMapping;
import org.springframework.web.reactive.handler.SimpleUrlHandlerMapping;
import org.springframework.web.reactive.socket.server.support.WebSocketHandlerAdapter;

@Configuration
public class LiveOrderbookWebSocketConfig {

  @Bean
  public HandlerMapping liveOrderbookHandlerMapping(LiveOrderbookWebSocketHandler webSocketHandler,
                                                    LiveTradingProperties properties) {
    SimpleUrlHandlerMapping mapping = new SimpleUrlHandlerMapping();
    mapping.setOrder(-1);
    mapping.setUrlMap(Map.of(normalizePath(properties.getFrontendWsPath()), webSocketHandler));
    return mapping;
  }

  @Bean
  public WebSocketHandlerAdapter webSocketHandlerAdapter() {
    return new WebSocketHandlerAdapter();
  }

  private String normalizePath(String rawPath) {
    if (!StringUtils.hasText(rawPath)) {
      return "/ws/live-orderbooks";
    }
    String path = rawPath.trim();
    if (!path.startsWith("/")) {
      path = "/" + path;
    }
    return path;
  }
}

