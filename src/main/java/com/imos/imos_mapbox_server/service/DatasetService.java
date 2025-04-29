package com.imos.imos_mapbox_server.service;

import com.imos.imos_mapbox_server.dto.DatasetResponse;
import com.imos.imos_mapbox_server.dto.OceanCurrentResponse;
import com.imos.imos_mapbox_server.mapper.DatasetMapper;
import com.imos.imos_mapbox_server.utils.DateUtils;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.List;

@Service
@RequiredArgsConstructor
public class DatasetService {

    private final DatasetMapper mapper;

    public List<DatasetResponse> fetchLastSevenDays(String baseUrl) {
        return DateUtils.getLastSevenDays().stream()
                .map(date -> mapper.toDatasetResponse(baseUrl, date))
                .toList();
    }

    public OceanCurrentResponse fetchOceanCurrentDetails(String date, Double lat, Double lon) throws IOException {
        return mapper.toOceanCurrentResponse(date,lat,lon);
    }

}
