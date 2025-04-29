package com.imos.imos_mapbox_server.controller;

import com.imos.imos_mapbox_server.dto.DatasetResponse;
import com.imos.imos_mapbox_server.dto.OceanCurrentResponse;
import com.imos.imos_mapbox_server.service.DatasetService;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import org.jetbrains.annotations.NotNull;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;

import java.io.IOException;
import java.util.List;

@Controller
@RequiredArgsConstructor
@RequestMapping("/api/v1/dataset")
public class DatasetController {
    private final DatasetService datasetService;

    @GetMapping("/last-seven-days")
    public ResponseEntity<List<DatasetResponse>> getLastSevenDays(HttpServletRequest request) {
        String baseUrl = buildBaseUrl(request);
        return ResponseEntity.ok(datasetService.fetchLastSevenDays(baseUrl));
    }

    @GetMapping("/ocean-current/{date}")
    public ResponseEntity<OceanCurrentResponse> getOceanCurrentDetails(
            @PathVariable String date,
            @RequestParam(name = "lat") Double lat,
            @RequestParam(name = "lon") Double lon
    ) throws IOException {
        return ResponseEntity.ok(datasetService.fetchOceanCurrentDetails(date, lat, lon));
    }

    private String buildBaseUrl(@NotNull HttpServletRequest request) {
        return String.format("%s://%s:%d",
                request.getScheme(),
                request.getServerName(),
                request.getServerPort());
    }
}
