package com.predictionmarkets.weather.models;

import java.time.Instant;
import jakarta.persistence.Column;
import jakarta.persistence.EmbeddedId;
import jakarta.persistence.Entity;
import jakarta.persistence.Table;

@Entity
@Table(name = "mos_run")
public class MosRun {
  @EmbeddedId
  private MosRunId id;

  @Column(name = "raw_payload_hash", nullable = false, length = 64)
  private String rawPayloadHash;

  @Column(name = "retrieved_at_utc", nullable = false)
  private Instant retrievedAtUtc;

  public MosRunId getId() {
    return id;
  }

  public void setId(MosRunId id) {
    this.id = id;
  }

  public String getRawPayloadHash() {
    return rawPayloadHash;
  }

  public void setRawPayloadHash(String rawPayloadHash) {
    this.rawPayloadHash = rawPayloadHash;
  }

  public Instant getRetrievedAtUtc() {
    return retrievedAtUtc;
  }

  public void setRetrievedAtUtc(Instant retrievedAtUtc) {
    this.retrievedAtUtc = retrievedAtUtc;
  }
}
