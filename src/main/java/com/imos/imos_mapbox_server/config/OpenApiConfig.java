package com.imos.imos_mapbox_server.config;


import io.swagger.v3.oas.annotations.OpenAPIDefinition;
import io.swagger.v3.oas.annotations.enums.SecuritySchemeIn;
import io.swagger.v3.oas.annotations.enums.SecuritySchemeType;
import io.swagger.v3.oas.annotations.info.Contact;
import io.swagger.v3.oas.annotations.info.Info;
import io.swagger.v3.oas.annotations.info.License;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import io.swagger.v3.oas.annotations.security.SecurityScheme;
import io.swagger.v3.oas.annotations.servers.Server;

/*
http://localhost:8080/swagger-ui/index.html#/
https://springdoc.org/
 */

@OpenAPIDefinition(
        info = @Info(
                contact = @Contact(
                        name = "Leslie",
                        email = "leslied41@gmail.com",
                        url = "https://www.linkedin.com/in/leslie-duan-641853121/"
                ),
                description = "OpenApi documentation for spring security",
                title = "OpenApi specification - Leslie",
                version = "1.0",
                license = @License(
                        name = "License name"
                )
        ),
        servers = {
                @Server(
                        description = "local env",
                        url = "http://localhost:8080"
                )
        }

)
public class OpenApiConfig {
}