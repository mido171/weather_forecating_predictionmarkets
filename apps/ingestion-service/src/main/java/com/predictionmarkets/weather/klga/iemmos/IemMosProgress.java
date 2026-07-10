package com.predictionmarkets.weather.klga.iemmos;

public record IemMosProgress(
    int chunksTotal,
    int completed,
    int completedEmpty,
    int failed,
    int remaining,
    long rowsUpserted,
    long featureRowsUpserted,
    long bytesFetched) {

  public double percentComplete() {
    if (chunksTotal <= 0) {
      return 0.0;
    }
    return ((completed + completedEmpty) * 100.0) / chunksTotal;
  }
}
