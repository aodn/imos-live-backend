package com.imos.imos_mapbox_server.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;

@Configuration
public class S3Config {
    @Value("${cloud.aws.region.static}")
    private String region;
//    this region needs to be added in configuration env variables, as AWS_REGION = ap-southeast-2. Otherwise, it will
//    cannot get the region, even though added in application.yml.

    @Bean
    public S3Client s3Client() {
        return S3Client.builder()
                .region(Region.of(region))
                .credentialsProvider(DefaultCredentialsProvider.create())
                .build();
//        the Access key and Secret access key are put in run configuration as env variables which will be read by s3client automatically.
    }
}
