package com.imos.imos_mapbox_server.service;


import com.imos.imos_mapbox_server.utils.DateUtils;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static com.imos.imos_mapbox_server.constant.BuoyConstants.BUOYS;
import static com.imos.imos_mapbox_server.constant.DataProcessingConstants.*;
import static com.imos.imos_mapbox_server.enums.UploadType.BUOY;
import static com.imos.imos_mapbox_server.enums.UploadType.GSLA;

@Service
@Slf4j
@RequiredArgsConstructor
public class PythonRunner {
    private final S3Service s3Service;
    private final S3UploadQueue s3UploadQueue;

    @Value("${app.storage.path}")
    private String storagePath;


    // Daily GSLA processing at 9:40 AM
    @Scheduled(cron = "0 40 09 * * ?")
    public void runGslaScript() {
        log.info("Starting daily GSLA script execution");
        try {
            Path outputDir = Paths.get(storagePath, GSLA.name());
            List<String> missingDates = s3Service.findGslaMissingFiles(outputDir, DateUtils.getLastSevenDays());

            if (!missingDates.isEmpty()) {
                runScript(outputDir, GSLA_PROCESSING_SCRIPT, missingDates, null);
                s3UploadQueue.queueGslaUpload(outputDir);
            }

        } catch (Exception e) {
            log.error("GSLA script execution failed", e);
        }
    }

    // Hourly wave buoys processing, this should process current month wave buoy files per hour because each hour the existing current files will be updated, until new month comes then new files generated.
    @Scheduled(cron = "0 30 * * * ?")
    public void runWaveBuoysScript() {
        log.info("Starting monthly wave buoys script execution");
        try {
            Path outputDir = Paths.get(storagePath, BUOY.name());
            List<String> missingDates =s3Service.findBuoyMissingFiles(outputDir, DateUtils.getCurrentMonthsInCurrentYear());

            if(!missingDates.isEmpty()) {
                runScript(outputDir, WAVE_BUOYS_PROCESSING_SCRIPT, missingDates, BUOYS);
                s3UploadQueue.queueBuoyUpload(outputDir);
            }

        } catch (Exception e) {
            log.error("Wave buoys script execution failed", e);
        }
    }


    private void runScript(Path outputDir, String scriptName, List<String> dates, List<String> buoys) {
        StringBuilder output = new StringBuilder();

        try {
            ProcessBuilder builder = buildDockerProcess(outputDir, scriptName, dates,buoys);
            builder.redirectErrorStream(true);
            Process process = builder.start();

            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line).append(System.lineSeparator());
                    log.debug("Python script output: {}", line);
                }
            }

            int exitCode = process.waitFor();
            if (exitCode == 139) {
                log.warn("Python script exited with code 139 but all files were created successfully");
            } else if (exitCode != 0) {
                log.error("Python script failed with exit code {} and output: {}", exitCode, output);
                throw new RuntimeException("Python script exited with code: " + exitCode);
            } else {
                log.info("Python script executed successfully");
            }

        } catch (Exception e) {
            log.error("Failed to execute Python script {} for dates: {}", scriptName, dates, e);
            throw new RuntimeException("Failed to run Python script", e);
        }
    }


    private ProcessBuilder buildDockerProcess(
            Path outputDir,
            String scriptName,
            List<String> dates,
            List<String> buoys
    ) {
        List<String> command = new ArrayList<>();
        command.add("docker");
        command.add("run");
        command.add("--rm");
        command.add("-v");
        command.add(outputDir.toAbsolutePath().toString().replace("\\", "/") + ":/data");
        command.add(DATA_PROCESSING_IMAGE);
        command.add(scriptName);
        command.add("--output_base_dir");
        command.add("/data");
        command.add("--dates");
        command.addAll(dates);

        if (WAVE_BUOYS_PROCESSING_SCRIPT.equals(scriptName)) {
            if (buoys != null && !buoys.isEmpty()) {
                command.add("--buoys");
                command.addAll(buoys);
            }
        }

        return new ProcessBuilder(command);
    }
}