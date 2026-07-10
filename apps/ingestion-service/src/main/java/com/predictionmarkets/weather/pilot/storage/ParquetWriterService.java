package com.predictionmarkets.weather.pilot.storage;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.springframework.stereotype.Service;

@Service
public class ParquetWriterService {
  public Path writePlaceholder(Path outputPath, List<String> rows) {
    try {
      Files.createDirectories(outputPath.getParent());
      Files.writeString(outputPath, String.join(System.lineSeparator(), rows));
      return outputPath;
    } catch (Exception ex) {
      throw new IllegalStateException("Failed to write placeholder artifact to " + outputPath, ex);
    }
  }
}
