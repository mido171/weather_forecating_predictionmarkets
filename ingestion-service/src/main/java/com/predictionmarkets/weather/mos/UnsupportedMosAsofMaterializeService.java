package com.predictionmarkets.weather.mos;

import com.predictionmarkets.weather.models.MosModel;
import java.time.LocalDate;
import java.util.List;
import org.springframework.stereotype.Service;

@Service
public class UnsupportedMosAsofMaterializeService implements MosAsofMaterializeService {
  @Override
  public void materializeForTargetDate(String stationId,
                                       LocalDate targetDate,
                                       Long asofPolicyId,
                                       List<MosModel> models) {
    throw new UnsupportedOperationException(
        "MOS as-of materialization is not implemented yet (requires WX-107).");
  }
}
