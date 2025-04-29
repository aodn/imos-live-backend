package com.imos.imos_mapbox_server.dto;

import lombok.Data;

@Data
public class OceanCurrentDetails {
    private Integer width;
    private Integer height;
    private double[] latRange;
    private double[] lonRange;
    private double[][][] data;
}
