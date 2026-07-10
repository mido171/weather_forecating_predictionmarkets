package com.predictionmarkets.weather.klga.iemmos;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictionmarkets.weather.models.MosModel;
import java.time.LocalDate;
import java.util.List;
import org.junit.jupiter.api.Test;

class IemMosPlannerTest {
  private final IemMosPlanner planner = new IemMosPlanner(new ObjectMapper());

  @Test
  void productRegistryIncludesEveryRequiredMosFamily() {
    assertThat(IemMosProduct.MAV.endpointModel()).isEqualTo(MosModel.GFS);
    assertThat(IemMosProduct.MET.endpointModel()).isEqualTo(MosModel.NAM);
    assertThat(IemMosProduct.MEX.endpointModel()).isEqualTo(MosModel.MEX);
    assertThat(IemMosProduct.LAV.endpointModel()).isEqualTo(MosModel.LAV);
    assertThat(IemMosProduct.NBS.endpointModel()).isEqualTo(MosModel.NBS);
    assertThat(IemMosProduct.NBE.endpointModel()).isEqualTo(MosModel.NBE);
  }

  @Test
  void t1245PlannerUsesProductStartDateWhenNoOverrideIsGiven() {
    IemMosBackfillProperties properties = new IemMosBackfillProperties();
    properties.setJobId("test_job");
    properties.setCutoffId("T_1245UTC");
    properties.setThrough(LocalDate.of(2003, 12, 17));

    List<IemMosChunk> chunks = planner.plan(
        properties,
        List.of(new IemMosStation("KLGA", "LGA")));

    assertThat(chunks)
        .extracting(chunk -> chunk.product().productCode())
        .containsExactly("MAV");
    IemMosChunk chunk = chunks.get(0);
    assertThat(chunk.startDate()).isEqualTo(LocalDate.of(2003, 12, 16));
    assertThat(chunk.endDateInclusive()).isEqualTo(LocalDate.of(2003, 12, 17));
    assertThat(chunk.requestJson()).contains("\"cutoffId\":\"T_1245UTC\"");
    assertThat(chunk.requestSha256()).hasSize(64);
  }

  @Test
  void requestHashChangesWhenCutoffChanges() {
    IemMosBackfillProperties first = new IemMosBackfillProperties();
    first.setJobId("test_job");
    first.setCutoffId("T_1245UTC");
    first.setStart(LocalDate.of(2026, 6, 1));
    first.setThrough(LocalDate.of(2026, 6, 1));

    IemMosBackfillProperties second = new IemMosBackfillProperties();
    second.setJobId("test_job");
    second.setCutoffId("T_MINUS_1_2045UTC");
    second.setStart(LocalDate.of(2026, 6, 1));
    second.setThrough(LocalDate.of(2026, 6, 1));

    IemMosStation station = new IemMosStation("KLGA", "LGA");
    IemMosChunk firstChunk = planner.plan(first, List.of(station)).get(0);
    IemMosChunk secondChunk = planner.plan(second, List.of(station)).get(0);

    assertThat(firstChunk.requestSha256()).isNotEqualTo(secondChunk.requestSha256());
  }
}
