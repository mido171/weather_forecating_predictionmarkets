package com.predictionmarkets.weather.pilot.storage;

import com.predictionmarkets.weather.common.Hashing;
import org.springframework.stereotype.Service;

@Service
public class ChecksumService {
  public String sha256(byte[] bytes) {
    return Hashing.sha256Hex(bytes);
  }

  public String sha256(String text) {
    return Hashing.sha256Hex(text);
  }
}
