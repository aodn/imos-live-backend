package com.imos.imos_mapbox_server;

import com.imos.imos_mapbox_server.scheduler.S3UploadScheduler;
import com.imos.imos_mapbox_server.service.PythonRunner;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.scheduling.annotation.EnableScheduling;

@Slf4j
@EnableScheduling
@SpringBootApplication
public class ImosMapboxServerApplication {

	public static void main(String[] args) {
		SpringApplication.run(ImosMapboxServerApplication.class, args);
	}

	//enable scheduled task work once immediately after application starts. S3 task need wait python runner finish.
	@Bean
	public CommandLineRunner commandLineRunner(PythonRunner pythonRunner, S3UploadScheduler s3UploadScheduler) {
		return args -> {

			try {
				log.info("Running Python script on application startup");
				pythonRunner.runScript();
			} catch (Exception e) {
				log.error("Failed to run Python script", e);
			}

			try {
				log.info("Running s3UploadScheduler on application startup");
				s3UploadScheduler.processAndUploadFiles();
			} catch (Exception e) {
				log.error("Failed to run S3 upload on startup", e);
			}
		};
	}

}
