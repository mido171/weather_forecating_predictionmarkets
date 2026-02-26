package com.predictionmarkets.weather.models;

import java.time.Instant;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Lob;
import jakarta.persistence.Table;

@Entity
@Table(name = "weathercom_ingestion_run")
public class WeatherComIngestionRun {
  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  @Column(name = "id", nullable = false)
  private Long id;

  @Enumerated(EnumType.STRING)
  @Column(name = "status", nullable = false, length = 32)
  private WeatherComIngestionStatus status;

  @Column(name = "started_at_utc", nullable = false)
  private Instant startedAtUtc;

  @Column(name = "finished_at_utc")
  private Instant finishedAtUtc;

  @Column(name = "requested_by", length = 128)
  private String requestedBy;

  @Lob
  @Column(name = "request_payload_json", nullable = false)
  private String requestPayloadJson;

  @Column(name = "total_tasks", nullable = false)
  private int totalTasks;

  @Column(name = "succeeded_tasks", nullable = false)
  private int succeededTasks;

  @Column(name = "failed_tasks", nullable = false)
  private int failedTasks;

  @Column(name = "created_at_utc", nullable = false)
  private Instant createdAtUtc;

  @Column(name = "updated_at_utc", nullable = false)
  private Instant updatedAtUtc;

  public Long getId() {
    return id;
  }

  public void setId(Long id) {
    this.id = id;
  }

  public WeatherComIngestionStatus getStatus() {
    return status;
  }

  public void setStatus(WeatherComIngestionStatus status) {
    this.status = status;
  }

  public Instant getStartedAtUtc() {
    return startedAtUtc;
  }

  public void setStartedAtUtc(Instant startedAtUtc) {
    this.startedAtUtc = startedAtUtc;
  }

  public Instant getFinishedAtUtc() {
    return finishedAtUtc;
  }

  public void setFinishedAtUtc(Instant finishedAtUtc) {
    this.finishedAtUtc = finishedAtUtc;
  }

  public String getRequestedBy() {
    return requestedBy;
  }

  public void setRequestedBy(String requestedBy) {
    this.requestedBy = requestedBy;
  }

  public String getRequestPayloadJson() {
    return requestPayloadJson;
  }

  public void setRequestPayloadJson(String requestPayloadJson) {
    this.requestPayloadJson = requestPayloadJson;
  }

  public int getTotalTasks() {
    return totalTasks;
  }

  public void setTotalTasks(int totalTasks) {
    this.totalTasks = totalTasks;
  }

  public int getSucceededTasks() {
    return succeededTasks;
  }

  public void setSucceededTasks(int succeededTasks) {
    this.succeededTasks = succeededTasks;
  }

  public int getFailedTasks() {
    return failedTasks;
  }

  public void setFailedTasks(int failedTasks) {
    this.failedTasks = failedTasks;
  }

  public Instant getCreatedAtUtc() {
    return createdAtUtc;
  }

  public void setCreatedAtUtc(Instant createdAtUtc) {
    this.createdAtUtc = createdAtUtc;
  }

  public Instant getUpdatedAtUtc() {
    return updatedAtUtc;
  }

  public void setUpdatedAtUtc(Instant updatedAtUtc) {
    this.updatedAtUtc = updatedAtUtc;
  }
}

