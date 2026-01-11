package com.predictionmarkets.weather.models;

import java.time.Instant;
import java.time.LocalDate;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.UniqueConstraint;

@Entity
@Table(
    name = "ingest_checkpoint",
    uniqueConstraints = {
        @UniqueConstraint(columnNames = {"job_name", "station_id", "model"})
    }
)
public class IngestCheckpoint {
  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  @Column(name = "id", nullable = false)
  private Long id;

  @Column(name = "job_name", nullable = false, length = 128)
  private String jobName;

  @Column(name = "station_id", nullable = false, length = 8)
  private String stationId;

  @Enumerated(EnumType.STRING)
  @Column(name = "model", length = 16)
  private MosModel model;

  @Column(name = "cursor_date")
  private LocalDate cursorDate;

  @Column(name = "cursor_runtime_utc")
  private Instant cursorRuntimeUtc;

  @Enumerated(EnumType.STRING)
  @Column(name = "status", nullable = false, length = 16)
  private IngestCheckpointStatus status;

  @Column(name = "updated_at_utc", nullable = false)
  private Instant updatedAtUtc;

  public Long getId() {
    return id;
  }

  public void setId(Long id) {
    this.id = id;
  }

  public String getJobName() {
    return jobName;
  }

  public void setJobName(String jobName) {
    this.jobName = jobName;
  }

  public String getStationId() {
    return stationId;
  }

  public void setStationId(String stationId) {
    this.stationId = stationId;
  }

  public MosModel getModel() {
    return model;
  }

  public void setModel(MosModel model) {
    this.model = model;
  }

  public LocalDate getCursorDate() {
    return cursorDate;
  }

  public void setCursorDate(LocalDate cursorDate) {
    this.cursorDate = cursorDate;
  }

  public Instant getCursorRuntimeUtc() {
    return cursorRuntimeUtc;
  }

  public void setCursorRuntimeUtc(Instant cursorRuntimeUtc) {
    this.cursorRuntimeUtc = cursorRuntimeUtc;
  }

  public IngestCheckpointStatus getStatus() {
    return status;
  }

  public void setStatus(IngestCheckpointStatus status) {
    this.status = status;
  }

  public Instant getUpdatedAtUtc() {
    return updatedAtUtc;
  }

  public void setUpdatedAtUtc(Instant updatedAtUtc) {
    this.updatedAtUtc = updatedAtUtc;
  }
}
