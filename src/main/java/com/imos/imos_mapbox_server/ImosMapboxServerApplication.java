package com.imos.imos_mapbox_server;

import com.imos.imos_mapbox_server.service.PythonRunner;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@Slf4j
@SpringBootApplication
public class ImosMapboxServerApplication {

	public static void main(String[] args) {
		SpringApplication.run(ImosMapboxServerApplication.class, args);
	}

	//enable scheduled task work when application starts.
	@Bean
	public CommandLineRunner commandLineRunner(PythonRunner pythonRunner) {
		return args -> {
			try {
				pythonRunner.runScript();
			} catch (Exception e) {
				log.error("Failed to run Python script", e);
			}
		};
	}

}
