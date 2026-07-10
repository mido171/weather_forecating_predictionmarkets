package com.predictionmarkets.weather.klga.wu;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class WuTruthConfigTest {
  @Test
  void defaultRateLimitIsOneHundredTwentyPerMinute() {
    WuTruthConfig config = WuTruthConfig.fromArgs(new String[] {
        "--command", "rebuild",
        "--jdbc-url", "jdbc:postgresql://127.0.0.1:5432/klga_tmax_research"
    });

    assertThat(config.rateLimitPerMinute()).isEqualTo(120);
    assertThat(config.resume()).isFalse();
  }

  @Test
  void parsesResumeAliasesAndRateLimitOverride() {
    WuTruthConfig resumeConfig = WuTruthConfig.fromArgs(new String[] {
        "--command", "rebuild",
        "--jdbc-url", "jdbc:postgresql://127.0.0.1:5432/klga_tmax_research",
        "--resume",
        "--rate-limit-per-minute", "240"
    });
    WuTruthConfig missingOnlyConfig = WuTruthConfig.fromArgs(new String[] {
        "--command", "rebuild",
        "--jdbc-url", "jdbc:postgresql://127.0.0.1:5432/klga_tmax_research",
        "--missing-only"
    });

    assertThat(resumeConfig.resume()).isTrue();
    assertThat(resumeConfig.rateLimitPerMinute()).isEqualTo(240);
    assertThat(missingOnlyConfig.resume()).isTrue();
  }
}
