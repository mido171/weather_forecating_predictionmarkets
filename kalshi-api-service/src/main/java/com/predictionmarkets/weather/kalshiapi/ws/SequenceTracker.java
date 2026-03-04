package com.predictionmarkets.weather.kalshiapi.ws;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.springframework.stereotype.Component;

@Component
public class SequenceTracker {

  public enum SequenceStatus {
    INITIAL,
    IN_ORDER,
    GAP,
    OUT_OF_ORDER
  }

  public record SequenceResult(SequenceStatus status, Long previousSeq) {
  }

  private final Map<Integer, Long> lastSeqBySid = new ConcurrentHashMap<>();
  private final Counter gapCounter;
  private final Counter outOfOrderCounter;

  public SequenceTracker(MeterRegistry meterRegistry) {
    this.gapCounter = meterRegistry.counter("kalshi.api.ws.seq_events", "result", "gap");
    this.outOfOrderCounter = meterRegistry.counter("kalshi.api.ws.seq_events", "result", "out_of_order");
  }

  public SequenceResult evaluate(Integer sid, Long seq) {
    if (sid == null || seq == null) {
      return new SequenceResult(SequenceStatus.IN_ORDER, null);
    }

    Long previous = lastSeqBySid.get(sid);
    if (previous == null) {
      lastSeqBySid.put(sid, seq);
      return new SequenceResult(SequenceStatus.INITIAL, null);
    }
    if (seq == previous + 1) {
      lastSeqBySid.put(sid, seq);
      return new SequenceResult(SequenceStatus.IN_ORDER, previous);
    }
    if (seq <= previous) {
      outOfOrderCounter.increment();
      return new SequenceResult(SequenceStatus.OUT_OF_ORDER, previous);
    }

    gapCounter.increment();
    lastSeqBySid.put(sid, seq);
    return new SequenceResult(SequenceStatus.GAP, previous);
  }

  public void reset(Integer sid) {
    if (sid != null) {
      lastSeqBySid.remove(sid);
    }
  }

  public void resetAll() {
    lastSeqBySid.clear();
  }
}
