package com.imos.imos_mapbox_server.mapper;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.imos.imos_mapbox_server.dto.DatasetMeta;
import com.imos.imos_mapbox_server.dto.DatasetResponse;
import com.imos.imos_mapbox_server.dto.OceanCurrentDetails;
import com.imos.imos_mapbox_server.dto.OceanCurrentResponse;
import com.imos.imos_mapbox_server.utils.OceanCurrentUtils;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

@Slf4j
@Component
@RequiredArgsConstructor
public class DatasetMapper {

    @Value("${app.storage.path}")
    private String storagePath;

    private final ObjectMapper objectMapper;

    public DatasetResponse toDatasetResponse(String baseUrl, String date) {
        try {
            return DatasetResponse.builder()
                    .particlePngUrl(String.format("%s/images/%s/gsla_input.png", baseUrl, date))
                    .overlayPngUrl(String.format("%s/images/%s/gsla_overlay.png", baseUrl, date))
                    .datasetMeta(readDatasetMeta(date))
                    .build();
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to read metadata for date: " + date, e);
        }
    }

    public OceanCurrentResponse toOceanCurrentResponse(String date, double lat, double lon) throws IOException {
        OceanCurrentDetails details=readOceanCurrentDetails(date);

        int[] index = OceanCurrentUtils.latLonToDataIndex(
                lat, lon,
                details.getLatRange(),
                details.getLonRange(),
                details.getWidth(),
                details.getHeight()
        );

        double[] values = details.getData()[index[1]][index[0]];
        double u = values[0];
        double v = values[1];
        boolean alpha = values[2]!= 0.0;
        double gsla = values[3];

        return OceanCurrentResponse.builder()
                .v(v)
                .u(u)
                .alpha(alpha)
                .gsla(gsla)
                .gslaUnit("m")
                .speed(OceanCurrentUtils.generateSpeed(u,v))
                .speedUnit("m/s")
                .degree(OceanCurrentUtils.generateDegree(u,v))
                .direction(OceanCurrentUtils.generateDirection(u,v))
                .build();
    }

    private DatasetMeta readDatasetMeta(String date) throws IOException {
        Path metaPath = Paths.get(storagePath, date, "gsla_meta.json").toAbsolutePath();
        return objectMapper.readValue(metaPath.toFile(), DatasetMeta.class);
    }

    public OceanCurrentDetails readOceanCurrentDetails(String date) throws IOException {
        Path oceanCurrentPath = Paths.get(storagePath, "GSLA", date, "gsla_data.json").toAbsolutePath();

        return objectMapper.readValue(oceanCurrentPath.toFile(), OceanCurrentDetails.class);
    }

}
