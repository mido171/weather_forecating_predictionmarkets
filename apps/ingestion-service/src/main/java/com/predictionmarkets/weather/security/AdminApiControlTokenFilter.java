package com.predictionmarkets.weather.security;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

@Component
public class AdminApiControlTokenFilter extends OncePerRequestFilter {
  public static final String CONTROL_TOKEN_HEADER = "X-Local-Control-Token";

  private final AdminApiProperties properties;

  public AdminApiControlTokenFilter(AdminApiProperties properties) {
    this.properties = properties;
  }

  @Override
  protected boolean shouldNotFilter(HttpServletRequest request) {
    String path = request.getRequestURI();
    return !(path.startsWith("/api/weathercom") || path.startsWith("/internal/ingest"));
  }

  @Override
  protected void doFilterInternal(HttpServletRequest request,
                                  HttpServletResponse response,
                                  FilterChain filterChain) throws ServletException, IOException {
    if (!properties.isEnabled()) {
      response.sendError(HttpStatus.NOT_FOUND.value());
      return;
    }

    String suppliedToken = request.getHeader(CONTROL_TOKEN_HEADER);
    if (!tokenMatches(properties.getControlToken(), suppliedToken)) {
      response.sendError(HttpStatus.UNAUTHORIZED.value());
      return;
    }

    filterChain.doFilter(request, response);
  }

  private boolean tokenMatches(String expected, String supplied) {
    if (expected == null || expected.isBlank() || supplied == null) {
      return false;
    }
    return MessageDigest.isEqual(
        expected.getBytes(StandardCharsets.UTF_8),
        supplied.getBytes(StandardCharsets.UTF_8));
  }
}
