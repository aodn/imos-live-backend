package com.imos.imos_mapbox_server.dto;

import lombok.Data;

import java.nio.file.Path;

@Data
public class S3FileUploadRequest {
    private String key;
    private Path filePath;
}
