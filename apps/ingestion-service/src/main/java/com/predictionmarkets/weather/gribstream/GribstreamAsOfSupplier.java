package com.predictionmarkets.weather.gribstream;

import java.time.Instant;
import java.time.LocalDate;

@FunctionalInterface
public interface GribstreamAsOfSupplier {
  Instant resolve(StationSpec station, LocalDate targetDateLocal);
}
