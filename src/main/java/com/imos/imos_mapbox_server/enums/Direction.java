package com.imos.imos_mapbox_server.enums;

import lombok.Getter;

@Getter
public enum Direction {
    E(0),
    NE(45),
    N(90),
    NW(135),
    W(180),
    SW(225),
    S(270),
    SE(315);

    private final int angle;

    Direction(int angle) {
        this.angle = angle;
    }

    public static Direction fromDegree(double degree) {
        // Normalize degree to [0, 360)
        double normalized = (degree % 360 + 360) % 360;

        int index = (int) Math.round(normalized / 45.0) % 8;

        return Direction.values()[index];
    }
}

