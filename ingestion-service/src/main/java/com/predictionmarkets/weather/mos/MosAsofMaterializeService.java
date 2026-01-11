package com.predictionmarkets.weather.mos;

import com.predictionmarkets.weather.models.MosModel;
import java.time.LocalDate;
import java.util.List;

public interface MosAsofMaterializeService {
  void materializeForTargetDate(String stationId,
                                LocalDate targetDate,
                                Long asofPolicyId,
                                List<MosModel> models);
}
