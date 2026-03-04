package com.predictionmarkets.weather.kalshiapi.auth;

import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.PrivateKey;
import java.security.Security;
import org.bouncycastle.asn1.pkcs.PrivateKeyInfo;
import org.bouncycastle.jce.provider.BouncyCastleProvider;
import org.bouncycastle.openssl.PEMKeyPair;
import org.bouncycastle.openssl.PEMParser;
import org.bouncycastle.openssl.jcajce.JcaPEMKeyConverter;
import org.bouncycastle.pkcs.PKCS8EncryptedPrivateKeyInfo;

public final class KalshiPrivateKeyLoader {

  private KalshiPrivateKeyLoader() {
  }

  public static PrivateKey loadPrivateKey(Path pemPath) {
    if (pemPath == null) {
      throw new IllegalArgumentException("PEM path must not be null");
    }
    ensureBouncyCastleProvider();

    try (Reader reader = Files.newBufferedReader(pemPath, StandardCharsets.UTF_8)) {
      return parsePem(reader, pemPath.toString());
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to read Kalshi private key PEM from " + pemPath, ex);
    }
  }

  public static PrivateKey loadPrivateKeyFromPem(String pemContent, String description) {
    if (pemContent == null || pemContent.isBlank()) {
      throw new IllegalArgumentException("PEM content must not be blank");
    }
    ensureBouncyCastleProvider();
    String label = (description == null || description.isBlank()) ? "inline PEM content" : description;
    try (Reader reader = new StringReader(pemContent)) {
      return parsePem(reader, label);
    } catch (IOException ex) {
      throw new IllegalStateException("Failed to read Kalshi private key PEM from " + label, ex);
    }
  }

  private static PrivateKey parsePem(Reader reader, String description) throws IOException {
    try (PEMParser parser = new PEMParser(reader)) {
      Object parsed = parser.readObject();
      if (parsed == null) {
        throw new IllegalStateException("No PEM object found in " + description);
      }

      JcaPEMKeyConverter converter = new JcaPEMKeyConverter().setProvider(BouncyCastleProvider.PROVIDER_NAME);

      if (parsed instanceof PEMKeyPair pemKeyPair) {
        return converter.getKeyPair(pemKeyPair).getPrivate();
      }
      if (parsed instanceof PrivateKeyInfo privateKeyInfo) {
        return converter.getPrivateKey(privateKeyInfo);
      }
      if (parsed instanceof PKCS8EncryptedPrivateKeyInfo) {
        throw new IllegalStateException("Encrypted private keys are not supported: " + description);
      }

      throw new IllegalStateException("Unsupported PEM object type " + parsed.getClass().getName()
          + " in " + description);
    }
  }

  private static void ensureBouncyCastleProvider() {
    if (Security.getProvider(BouncyCastleProvider.PROVIDER_NAME) == null) {
      Security.addProvider(new BouncyCastleProvider());
    }
  }
}
