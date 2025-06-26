package com.imos.imos_mapbox_server;

import com.imos.imos_mapbox_server.constant.DataProcessingConstants;
import com.imos.imos_mapbox_server.scheduler.S3UploadScheduler;
import com.imos.imos_mapbox_server.service.PythonRunner;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.scheduling.annotation.EnableScheduling;

import java.util.concurrent.CompletableFuture;

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
		String GSLA_PROCESSING_SCRIPT = DataProcessingConstants.GSLA_PROCESSING_SCRIPT;
		String WAVE_BUOYS_PROCESSING_SCRIPT = DataProcessingConstants.WAVE_BUOYS_PROCESSING_SCRIPT;
		return args -> {

			log.info("Starting initial data processing pipeline...");
			//start these two simultaneously
			CompletableFuture<Void>	waveBuoysTask =	runScriptAsync(WAVE_BUOYS_PROCESSING_SCRIPT,pythonRunner::runWaveBuoysScript);
			CompletableFuture<Void>	gslaTask =	runScriptAsync(GSLA_PROCESSING_SCRIPT,pythonRunner::runGslaScript);

			CompletableFuture.allOf(waveBuoysTask,gslaTask).join();
			log.info("Both Python scripts completed. Starting S3 upload...");

			try {
				s3UploadScheduler.processAndUploadFiles();
			} catch (Exception e) {
				log.error("Failed to run S3 upload on startup", e);
			}

			log.info("All initial data processing completed successfully");
		};
	}

	private CompletableFuture<Void> runScriptAsync(String scriptName, Runnable scriptRunner) {
		return CompletableFuture.runAsync(() -> {
			try {
				log.info("Starting {} script (parallel)", scriptName);
				scriptRunner.run();
				log.info("{} script completed successfully", scriptName);
			} catch (Exception e) {
				log.error("{} script failed", scriptName, e);
				throw new RuntimeException(e);
			}
		});
	}
}
