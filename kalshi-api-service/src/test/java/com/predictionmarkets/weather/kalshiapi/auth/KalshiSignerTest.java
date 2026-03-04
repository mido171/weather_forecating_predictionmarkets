package com.predictionmarkets.weather.kalshiapi.auth;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.charset.StandardCharsets;
import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.security.PublicKey;
import java.security.Signature;
import java.security.spec.MGF1ParameterSpec;
import java.security.spec.PSSParameterSpec;
import java.time.Clock;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.Base64;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class KalshiSignerTest {

  private static final PSSParameterSpec PSS_SHA256_SPEC = new PSSParameterSpec(
      "SHA-256",
      "MGF1",
      MGF1ParameterSpec.SHA256,
      32,
      1
  );

  private KeyPair keyPair;
  private KalshiSigner signer;

  @BeforeEach
  void setUp() throws Exception {
    KeyPairGenerator generator = KeyPairGenerator.getInstance("RSA");
    generator.initialize(2048);
    keyPair = generator.generateKeyPair();
    Clock fixedClock = Clock.fixed(Instant.parse("2024-01-01T00:00:00Z"), ZoneOffset.UTC);
    signer = new KalshiSigner(keyPair.getPrivate(), "test-key", fixedClock);
  }

  @Test
  void signStripsQueryParametersBeforeSigning() throws Exception {
    SignedHeaders headers = signer.sign("GET", "/portfolio/orders?limit=5");
    String ts = headers.accessTimestamp();
    String expected = ts + "GET" + "/portfolio/orders";
    String wrong = ts + "GET" + "/portfolio/orders?limit=5";
    assertThat(verify(keyPair.getPublic(), expected, headers.accessSignature())).isTrue();
    assertThat(verify(keyPair.getPublic(), wrong, headers.accessSignature())).isFalse();
  }

  @Test
  void signWebSocketHandshakeUsesExpectedPath() throws Exception {
    SignedHeaders headers = signer.signWebSocketHandshake();
    String message = headers.accessTimestamp() + "GET" + KalshiSigner.WS_PATH;
    assertThat(verify(keyPair.getPublic(), message, headers.accessSignature())).isTrue();
  }

  private boolean verify(PublicKey publicKey, String message, String signatureBase64) throws Exception {
    Signature verifier = Signature.getInstance("RSASSA-PSS");
    verifier.setParameter(PSS_SHA256_SPEC);
    verifier.initVerify(publicKey);
    verifier.update(message.getBytes(StandardCharsets.UTF_8));
    return verifier.verify(Base64.getDecoder().decode(signatureBase64));
  }
}
