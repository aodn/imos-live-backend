package com.imos.imos_mapbox_server.dto;

import com.imos.imos_mapbox_server.enums.Direction;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@Builder
public class OceanCurrentResponse {
    private Double speed;
    private String speedUnit;
    private Double degree;
    private Direction direction;
    private Double u;
    private Double v;
    private boolean alpha;
    private Double gsla;
    private String gslaUnit;
}
