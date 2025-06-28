package com.imos.imos_mapbox_server.dto;

import com.imos.imos_mapbox_server.enums.UploadType;
import lombok.AllArgsConstructor;
import lombok.Data;

import java.nio.file.Path;

@Data
@AllArgsConstructor
public class UploadTask {
    private UploadType type;
    private Path outputDir;
}
