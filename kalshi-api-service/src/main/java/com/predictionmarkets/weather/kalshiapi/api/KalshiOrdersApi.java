package com.predictionmarkets.weather.kalshiapi.api;

import com.predictionmarkets.weather.kalshiapi.http.KalshiHttpClient;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.BatchCancelOrdersRequest;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.BatchCancelOrdersResponse;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.CancelOrderResponse;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.CreateOrderRequest;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.CreateOrderResponse;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.GetOrderResponse;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.GetOrdersQuery;
import com.predictionmarkets.weather.kalshiapi.model.portfolio.GetOrdersResponse;
import org.springframework.stereotype.Component;

@Component
public class KalshiOrdersApi {

  private final KalshiHttpClient httpClient;

  public KalshiOrdersApi(KalshiHttpClient httpClient) {
    this.httpClient = httpClient;
  }

  public CreateOrderResponse createOrder(CreateOrderRequest request) {
    return httpClient.postAuth("/portfolio/orders", request, CreateOrderResponse.class);
  }

  public CancelOrderResponse cancelOrder(String orderId) {
    return httpClient.deleteAuth("/portfolio/orders/" + orderId, CancelOrderResponse.class);
  }

  public BatchCancelOrdersResponse batchCancelOrders(BatchCancelOrdersRequest request) {
    return httpClient.deleteAuth("/portfolio/orders/batched", request, BatchCancelOrdersResponse.class);
  }

  public GetOrdersResponse getOrders(GetOrdersQuery query) {
    return httpClient.getAuth("/portfolio/orders", query == null ? null : query::applyTo, GetOrdersResponse.class);
  }

  public GetOrderResponse getOrder(String orderId) {
    return httpClient.getAuth("/portfolio/orders/" + orderId, null, GetOrderResponse.class);
  }
}
