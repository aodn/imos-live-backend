package com.imos.imos_mapbox_server.utils;

import com.imos.imos_mapbox_server.enums.Direction;

public class OceanCurrentUtils {
    /**
     * round lat and lon to the grid point(lat, lon) existing in gsla-data.json.
     * @return rounded index list points to value in gsla-data.json.
     */
    public static int[] latLonToDataIndex(
            double lat,double lon,
            double[] latRange, double[] lonRange,
            int width, int height
    ) {
        double minLon = lonRange[0];
        double maxLat = latRange[1];
        double maxLon = lonRange[1];
        double minLat = latRange[0];
        if(lat < minLat || lat > maxLat || lon < minLon || lon>maxLon) {
            throw new IllegalArgumentException("Latitude or Longitude out of range");
        }

        double x = ((lon - minLon) / (maxLon - minLon)) * width;
        double y = ((maxLat - lat) / (maxLat - minLat)) * height;

        return new int[]{(int) Math.floor(x), (int) Math.floor(y)};
    }

    public static double generateSpeed(double u, double v) {
        return Math.sqrt(u * u + v * v);
    }

    public static double generateDegree(double u, double v) {
        double degree = Math.atan2(v,u)*180/Math.PI;
        if(degree < 0) degree = 360+degree;
        return degree;
    }

    public static Direction generateDirection(double u, double v) {
        double degree = generateDegree(u,v);
        return Direction.fromDegree(degree);
    }

}
