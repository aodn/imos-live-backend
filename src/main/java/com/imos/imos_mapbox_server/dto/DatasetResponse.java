package com.imos.imos_mapbox_server.dto;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@Builder
public class DatasetResponse {
    private DatasetMeta datasetMeta;
    private String particlePngUrl;
    private String overlayPngUrl;
}
