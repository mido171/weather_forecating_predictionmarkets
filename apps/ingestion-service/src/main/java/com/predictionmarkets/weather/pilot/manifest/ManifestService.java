package com.predictionmarkets.weather.pilot.manifest;

import com.predictionmarkets.weather.pilot.catalog.SourceInventoryRecord;
import com.predictionmarkets.weather.pilot.catalog.SqliteCatalogService;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service
public class ManifestService {
  private static final Logger logger = LoggerFactory.getLogger(ManifestService.class);

  private final SqliteCatalogService catalogService;

  public ManifestService(SqliteCatalogService catalogService) {
    this.catalogService = catalogService;
  }

  public void recordHttpRequest(HttpRequestLogRecord record) {
    catalogService.execute("""
        INSERT INTO http_request_log (
          run_id, job_id, source_name, source_family, station_key, action,
          request_url_or_key, http_method, status_code, issue_time_utc, valid_time_utc,
          duration_ms, bytes_downloaded, rows_parsed, retry_count, status,
          exception_class, exception_message, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        record.runId(),
        record.jobId(),
        record.sourceName(),
        record.sourceFamily(),
        record.stationKey(),
        record.action(),
        record.requestUrlOrKey(),
        record.httpMethod(),
        record.statusCode(),
        record.issueTimeUtc(),
        record.validTimeUtc(),
        record.durationMs(),
        record.bytesDownloaded(),
        record.rowsParsed(),
        record.retryCount(),
        record.status(),
        record.exceptionClass(),
        record.exceptionMessage(),
        record.createdAtUtc());
    Map<String, Object> payload = new LinkedHashMap<>();
    payload.put("timestamp", Instant.now().toString());
    payload.put("level", "INFO");
    payload.put("job_id", record.jobId());
    payload.put("run_id", record.runId());
    payload.put("source_name", record.sourceName());
    payload.put("source_family", record.sourceFamily());
    payload.put("station_key", record.stationKey());
    payload.put("action", record.action());
    payload.put("request_url_or_key", record.requestUrlOrKey());
    payload.put("issue_time_utc", record.issueTimeUtc());
    payload.put("valid_time_utc", record.validTimeUtc());
    payload.put("duration_ms", record.durationMs());
    payload.put("bytes_downloaded", record.bytesDownloaded());
    payload.put("rows_parsed", record.rowsParsed());
    payload.put("status", record.status());
    payload.put("retry_count", record.retryCount());
    payload.put("exception_class", record.exceptionClass());
    payload.put("exception_message", record.exceptionMessage());
    logEvent(payload);
  }

  public void recordObjectManifest(ObjectManifestRecord record) {
    catalogService.execute("""
        INSERT INTO object_manifest (
          object_id, run_id, station_key, source_name, source_family, source_identifier,
          request_url_or_bucket_key, requested_range_start_utc, requested_range_end_utc,
          cycle_time_utc, forecast_hour, domain_name, http_status, content_length,
          checksum_sha256, payload_encoding, payload_text, payload_blob, parser_status,
          row_count, duplicate_of_checksum, extraction_status, grib_message_count, ingested_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(object_id) DO UPDATE SET
          parser_status=excluded.parser_status,
          row_count=excluded.row_count,
          duplicate_of_checksum=excluded.duplicate_of_checksum,
          extraction_status=excluded.extraction_status,
          ingested_at_utc=excluded.ingested_at_utc
        """,
        record.objectId(),
        record.runId(),
        record.stationKey(),
        record.sourceName(),
        record.sourceFamily(),
        record.sourceIdentifier(),
        record.requestUrlOrBucketKey(),
        record.requestedRangeStartUtc(),
        record.requestedRangeEndUtc(),
        record.cycleTimeUtc(),
        record.forecastHour(),
        record.domainName(),
        record.httpStatus(),
        record.contentLength(),
        record.checksumSha256(),
        record.payloadEncoding(),
        record.payloadText(),
        record.payloadBlob(),
        record.parserStatus(),
        record.rowCount(),
        record.duplicateOfChecksum(),
        record.extractionStatus(),
        record.gribMessageCount(),
        record.ingestedAtUtc());
    Map<String, Object> payload = new LinkedHashMap<>();
    payload.put("timestamp", Instant.now().toString());
    payload.put("level", "INFO");
    payload.put("run_id", record.runId());
    payload.put("source_name", record.sourceName());
    payload.put("source_family", record.sourceFamily());
    payload.put("station_key", record.stationKey());
    payload.put("action", "object_manifest");
    payload.put("request_url_or_key", record.requestUrlOrBucketKey());
    payload.put("issue_time_utc", record.cycleTimeUtc());
    payload.put("valid_time_utc", null);
    payload.put("bytes_downloaded", record.contentLength());
    payload.put("rows_parsed", record.rowCount());
    payload.put("status", record.parserStatus());
    payload.put("forecast_hour", record.forecastHour());
    payload.put("domain", record.domainName());
    payload.put("extraction_status", record.extractionStatus());
    payload.put("grib_message_count", record.gribMessageCount());
    payload.put("checksum_sha256", record.checksumSha256());
    logEvent(payload);
  }

  public void recordParserRun(ParserRunRecord record) {
    catalogService.execute("""
        INSERT INTO parser_run (
          parser_run_id, run_id, source_name, parser_version, object_id,
          status, rows_parsed, duration_ms, details_json, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        record.parserRunId(),
        record.runId(),
        record.sourceName(),
        record.parserVersion(),
        record.objectId(),
        record.status(),
        record.rowsParsed(),
        record.durationMs(),
        record.detailsJson(),
        record.createdAtUtc());
    Map<String, Object> payload = new LinkedHashMap<>();
    payload.put("timestamp", Instant.now().toString());
    payload.put("level", "INFO");
    payload.put("run_id", record.runId());
    payload.put("source_name", record.sourceName());
    payload.put("action", "parser_run");
    payload.put("status", record.status());
    payload.put("rows_parsed", record.rowsParsed());
    payload.put("duration_ms", record.durationMs());
    payload.put("parser_version", record.parserVersion());
    payload.put("object_id", record.objectId());
    logEvent(payload);
  }

  public void recordNormalizedPartition(String runId,
                                        String datasetName,
                                        String stationKey,
                                        String partitionDateUtc,
                                        int rowCount,
                                        String checksumSha256,
                                        String detailsJson) {
    catalogService.execute("""
        INSERT INTO normalized_partition (
          run_id, dataset_name, station_key, partition_date_utc, row_count,
          checksum_sha256, details_json, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        runId,
        datasetName,
        stationKey,
        partitionDateUtc,
        rowCount,
        checksumSha256,
        detailsJson,
        catalogService.nowUtc());
    Map<String, Object> payload = new LinkedHashMap<>();
    payload.put("timestamp", Instant.now().toString());
    payload.put("level", "INFO");
    payload.put("run_id", runId);
    payload.put("station_key", stationKey);
    payload.put("action", "normalized_partition");
    payload.put("dataset_name", datasetName);
    payload.put("valid_time_utc", partitionDateUtc);
    payload.put("rows_parsed", rowCount);
    payload.put("status", "RECORDED");
    payload.put("checksum_sha256", checksumSha256);
    logEvent(payload);
  }

  public void upsertSourceInventory(SourceInventoryRecord record) {
    catalogService.execute("""
        INSERT INTO source_inventory (
          inventory_key, station_key, source_name, source_family, item_type, item_key,
          issue_time_utc, valid_time_utc, status, details_json, created_at_utc, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(inventory_key) DO UPDATE SET
          status=excluded.status,
          details_json=excluded.details_json,
          updated_at_utc=excluded.updated_at_utc
        """,
        record.inventoryKey(),
        record.stationKey(),
        record.sourceName(),
        record.sourceFamily(),
        record.itemType(),
        record.itemKey(),
        record.issueTimeUtc(),
        record.validTimeUtc(),
        record.status(),
        record.detailsJson(),
        record.createdAtUtc(),
        record.updatedAtUtc());
  }

  private void logEvent(Map<String, Object> payload) {
    logger.info(catalogService.toJson(payload));
  }
}
