package com.predictionmarkets.weather.klga.iemmos;

import com.predictionmarkets.weather.models.MosModel;
import java.time.LocalDate;

public enum IemMosProduct {
  MAV("MAV", MosModel.GFS, LocalDate.of(2003, 12, 16)),
  MET("MET", MosModel.NAM, LocalDate.of(2008, 12, 9)),
  MEX("MEX", MosModel.MEX, LocalDate.of(2020, 7, 12)),
  LAV("LAV", MosModel.LAV, LocalDate.of(2020, 7, 12)),
  NBS("NBS", MosModel.NBS, LocalDate.of(2020, 1, 1)),
  NBE("NBE", MosModel.NBE, LocalDate.of(2021, 1, 1));

  private final String productCode;
  private final MosModel endpointModel;
  private final LocalDate defaultStartDate;

  IemMosProduct(String productCode, MosModel endpointModel, LocalDate defaultStartDate) {
    this.productCode = productCode;
    this.endpointModel = endpointModel;
    this.defaultStartDate = defaultStartDate;
  }

  public String productCode() {
    return productCode;
  }

  public MosModel endpointModel() {
    return endpointModel;
  }

  public LocalDate defaultStartDate() {
    return defaultStartDate;
  }
}
