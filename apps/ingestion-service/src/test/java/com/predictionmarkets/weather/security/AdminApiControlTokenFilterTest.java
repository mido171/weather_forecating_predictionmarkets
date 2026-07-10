package com.predictionmarkets.weather.security;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockFilterChain;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;

class AdminApiControlTokenFilterTest {

  @Test
  void disabledAdminApiIsHidden() throws Exception {
    AdminApiProperties properties = new AdminApiProperties();
    AdminApiControlTokenFilter filter = new AdminApiControlTokenFilter(properties);
    MockHttpServletResponse response = invoke(filter, "/internal/ingest/jobs/heavy", null);

    assertThat(response.getStatus()).isEqualTo(404);
  }

  @Test
  void enabledAdminApiRequiresMatchingControlToken() throws Exception {
    AdminApiProperties properties = enabledProperties();
    AdminApiControlTokenFilter filter = new AdminApiControlTokenFilter(properties);

    assertThat(invoke(filter, "/api/weathercom/ingestions", null).getStatus()).isEqualTo(401);
    assertThat(invoke(filter, "/api/weathercom/ingestions", "wrong").getStatus()).isEqualTo(401);
    assertThat(invoke(filter, "/api/weathercom/ingestions", "test-control-token").getStatus())
        .isEqualTo(200);
  }

  @Test
  void enabledAdminApiCannotStartWithoutControlToken() {
    AdminApiProperties properties = new AdminApiProperties();
    properties.setEnabled(true);

    assertThatThrownBy(properties::validate)
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("INGESTION_LOCAL_CONTROL_TOKEN");
  }

  private AdminApiProperties enabledProperties() {
    AdminApiProperties properties = new AdminApiProperties();
    properties.setEnabled(true);
    properties.setControlToken("test-control-token");
    properties.validate();
    return properties;
  }

  private MockHttpServletResponse invoke(AdminApiControlTokenFilter filter,
                                         String path,
                                         String token) throws Exception {
    MockHttpServletRequest request = new MockHttpServletRequest("POST", path);
    request.setRequestURI(path);
    if (token != null) {
      request.addHeader(AdminApiControlTokenFilter.CONTROL_TOKEN_HEADER, token);
    }
    MockHttpServletResponse response = new MockHttpServletResponse();
    filter.doFilter(request, response, new MockFilterChain());
    return response;
  }
}
