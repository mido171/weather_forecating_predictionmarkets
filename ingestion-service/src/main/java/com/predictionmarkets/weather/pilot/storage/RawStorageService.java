package com.predictionmarkets.weather.pilot.storage;

import com.predictionmarkets.weather.pilot.manifest.ManifestService;
import com.predictionmarkets.weather.pilot.manifest.ObjectManifestRecord;
import java.time.Instant;
import java.util.UUID;
import org.springframework.stereotype.Service;

@Service
public class RawStorageService {
  private final ChecksumService checksumService;
  private final ManifestService manifestService;

  public RawStorageService(ChecksumService checksumService, ManifestService manifestService) {
    this.checksumService = checksumService;
    this.manifestService = manifestService;
  }

  public String storeText(String runId,
                          String stationKey,
                          String sourceName,
                          String sourceFamily,
                          String sourceIdentifier,
                          String requestKey,
                          int httpStatus,
                          String text,
                          String parserStatus,
                          int rowCount) {
    byte[] bytes = text == null ? new byte[0] : text.getBytes(java.nio.charset.StandardCharsets.UTF_8);
    String checksum = checksumService.sha256(bytes);
    String objectId = UUID.randomUUID().toString();
    manifestService.recordObjectManifest(new ObjectManifestRecord(
        objectId,
        runId,
        stationKey,
        sourceName,
        sourceFamily,
        sourceIdentifier,
        requestKey,
        null,
        null,
        null,
        null,
        null,
        httpStatus,
        bytes.length,
        checksum,
        "text/plain;charset=utf-8",
        text,
        null,
        parserStatus,
        rowCount,
        null,
        null,
        null,
        Instant.now().toString()));
    return checksum;
  }

  public String storeBytes(String runId,
                           String stationKey,
                           String sourceName,
                           String sourceFamily,
                           String sourceIdentifier,
                           String requestKey,
                           int httpStatus,
                           byte[] payload,
                           String payloadEncoding,
                           String parserStatus,
                           int rowCount) {
    byte[] bytes = payload == null ? new byte[0] : payload;
    String checksum = checksumService.sha256(bytes);
    String objectId = UUID.randomUUID().toString();
    manifestService.recordObjectManifest(new ObjectManifestRecord(
        objectId,
        runId,
        stationKey,
        sourceName,
        sourceFamily,
        sourceIdentifier,
        requestKey,
        null,
        null,
        null,
        null,
        null,
        httpStatus,
        bytes.length,
        checksum,
        payloadEncoding,
        null,
        bytes,
        parserStatus,
        rowCount,
        null,
        null,
        null,
        Instant.now().toString()));
    return checksum;
  }
}
