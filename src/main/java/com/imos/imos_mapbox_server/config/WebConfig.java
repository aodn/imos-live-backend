package com.imos.imos_mapbox_server.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;


import java.nio.file.Paths;


@Configuration
@EnableScheduling
public class WebConfig implements WebMvcConfigurer {
    @Value("${app.storage.path}")
    private String storagePath;
    //expose files inside storagePath folder to downloadable url like files inside resources.
    //TODO this is only for development, the images and other files should be better to be saved in S3.
    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        String path = Paths.get(storagePath).toUri().toString();
        registry.addResourceHandler("/images/**")
                .addResourceLocations(path);

    }


}
