package com.imos.imos_mapbox_server.dto;

import jakarta.validation.constraints.Size;
import lombok.Data;

import java.util.List;

@Data
public class DatasetMeta {
    @Size(min = 2, max = 2, message = "latRange must contain exactly 2 elements")
    private List<Double> latRange;

    @Size(min = 2, max = 2, message = "lonRange must contain exactly 2 elements")
    private List<Double> lonRange;

    @Size(min = 2, max = 2, message = "uRange must contain exactly 2 elements")
    private List<Double> uRange;

    @Size(min = 2, max = 2, message = "vRange must contain exactly 2 elements")
    private List<Double> vRange;
}
