package com.predictionmarkets.weather.common.http;

@FunctionalInterface
public interface Sleeper {
  void sleep(long millis) throws InterruptedException;

  static Sleeper defaultSleeper() {
    return Thread::sleep;
  }
}
