package com.imos.imos_mapbox_server.service;

import com.imos.imos_mapbox_server.constant.DataProcessingConstants;
import com.imos.imos_mapbox_server.utils.DateUtils;
import com.imos.imos_mapbox_server.utils.FileUtils;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

@Service
@Slf4j
public class PythonRunner {

    @Value("${app.storage.path}")
    private String storagePath;

    // Daily GSLA processing at 9:40 AM
    @Scheduled(cron = "0 40 09 * * ?")
    public void runGslaScript() {
        log.info("Starting daily GSLA script execution");
        try {
            List<String> missingDates = FileUtils.findMissingDirectories(storagePath, DateUtils.getLastSevenDays());
            if (!missingDates.isEmpty()) {
                String GSLA_PROCESSING_SCRIPT = DataProcessingConstants.GSLA_PROCESSING_SCRIPT;
                runScript(GSLA_PROCESSING_SCRIPT, missingDates);
            } else {
                log.info("No missing GSLA date files found, skipping script execution");
            }
        } catch (Exception e) {
            log.error("GSLA script execution failed", e);
        }
    }

    // Monthly wave buoys processing on 1st day at 2:00 AM
    @Scheduled(cron = "0 0 02 1 * ?")
    public void runWaveBuoysScript() {
        log.info("Starting monthly wave buoys script execution");
        try {
            //TODO add condition check, if data already processed then skip.
            String WAVE_BUOYS_PROCESSING_SCRIPT = DataProcessingConstants.WAVE_BUOYS_PROCESSING_SCRIPT;
            runScript(WAVE_BUOYS_PROCESSING_SCRIPT, List.of("2025-01-01"));
        } catch (Exception e) {
            log.error("Wave buoys script execution failed", e);
        }
    }


    private void runScript(String scriptName, List<String> dates) {
        StringBuilder output = new StringBuilder();
        String outputDir = new File(storagePath).getAbsolutePath().replace("\\", "/");

        try {
            ProcessBuilder builder = buildDockerProcess(outputDir, scriptName, dates);
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
                // Check if files are still missing after execution
                List<String> stillMissingFiles = FileUtils.findMissingDirectories(storagePath, DateUtils.getLastSevenDays());
                if(!stillMissingFiles.isEmpty()) {
                    throw new RuntimeException("Python script exited with segmentation fault (139), files still missing");
                }
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

    private ProcessBuilder buildDockerProcess(String outputDir, String scriptName, List<String> dates) {
        String dockerImageName = DataProcessingConstants.DATA_PROCESSING_IMAGE;
        List<String> command = new ArrayList<>();
        command.add("docker");
        command.add("run");
        command.add("--rm");
        command.add("-v");
        command.add(outputDir + ":/data");
        command.add(dockerImageName);
        command.add(scriptName);
        command.add("--output_base_dir");
        command.add("/data");
        command.add("--dates");
        command.addAll(dates);

        log.info("Executing Docker command: {}", String.join(" ", command));
        return new ProcessBuilder(command);
    }
}