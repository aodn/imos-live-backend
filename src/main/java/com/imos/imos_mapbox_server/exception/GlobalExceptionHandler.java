package com.imos.imos_mapbox_server.exception;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<ExceptionResponse> handleException(IllegalArgumentException e) {
        return ResponseEntity.
                status(HttpStatus.BAD_REQUEST).
                body(
                        ExceptionResponse.builder()
                                .error(e.getMessage())
                                .build()
                );
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ExceptionResponse> handleException(Exception e) {
        e.printStackTrace();
        return ResponseEntity.
                status(HttpStatus.INTERNAL_SERVER_ERROR).
                body(
                        ExceptionResponse.builder()
                                .error(e.getMessage())
                                .build()
                );
    }
}
