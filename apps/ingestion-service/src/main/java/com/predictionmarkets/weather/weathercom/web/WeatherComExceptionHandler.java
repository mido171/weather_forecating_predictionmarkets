package com.predictionmarkets.weather.weathercom.web;

import com.predictionmarkets.weather.weathercom.service.WeatherComNotFoundException;
import com.predictionmarkets.weather.weathercom.web.dto.WeatherComErrorResponse;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.ConstraintViolationException;
import java.time.Instant;
import java.time.format.DateTimeParseException;
import java.util.stream.Collectors;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
public class WeatherComExceptionHandler {
  @ExceptionHandler(WeatherComNotFoundException.class)
  public ResponseEntity<WeatherComErrorResponse> handleNotFound(
      WeatherComNotFoundException ex,
      HttpServletRequest request) {
    return build(HttpStatus.NOT_FOUND, ex.getMessage(), request);
  }

  @ExceptionHandler({
      IllegalArgumentException.class,
      DateTimeParseException.class,
      ConstraintViolationException.class,
      MethodArgumentNotValidException.class
  })
  public ResponseEntity<WeatherComErrorResponse> handleBadRequest(
      Exception ex,
      HttpServletRequest request) {
    String message = ex.getMessage();
    if (ex instanceof MethodArgumentNotValidException methodArgumentNotValidException) {
      message = methodArgumentNotValidException.getBindingResult().getFieldErrors().stream()
          .map(this::formatFieldError)
          .collect(Collectors.joining("; "));
    } else if (ex instanceof ConstraintViolationException constraintViolationException) {
      message = constraintViolationException.getConstraintViolations().stream()
          .map(violation -> violation.getPropertyPath() + " " + violation.getMessage())
          .collect(Collectors.joining("; "));
    }
    return build(HttpStatus.BAD_REQUEST, message, request);
  }

  @ExceptionHandler(Exception.class)
  public ResponseEntity<WeatherComErrorResponse> handleUnhandled(
      Exception ex,
      HttpServletRequest request) {
    return build(HttpStatus.INTERNAL_SERVER_ERROR, "Unexpected server error", request);
  }

  private ResponseEntity<WeatherComErrorResponse> build(
      HttpStatus status,
      String message,
      HttpServletRequest request) {
    WeatherComErrorResponse body = new WeatherComErrorResponse(
        Instant.now(),
        status.value(),
        status.getReasonPhrase(),
        message,
        request == null ? null : request.getRequestURI());
    return ResponseEntity.status(status).body(body);
  }

  private String formatFieldError(FieldError fieldError) {
    if (fieldError == null) {
      return "";
    }
    return fieldError.getField() + " " + fieldError.getDefaultMessage();
  }
}
