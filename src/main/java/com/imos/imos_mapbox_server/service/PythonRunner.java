package com.imos.imos_mapbox_server.service;

import com.imos.imos_mapbox_server.utils.DateUtils;
import com.imos.imos_mapbox_server.utils.FileUtils;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.scheduling.annotation.EnableAsync;
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

    @Scheduled(cron = "0 40 09 * * ?")
    public void runScheduledScript() {  // Better method name
        try {
            runScript();
        } catch (Exception e) {
            log.error("Scheduled Python script execution failed", e);
        }
    }

    public void runScript() {
        StringBuilder output = new StringBuilder();
        String outputDir = new File(storagePath).getAbsolutePath().replace("\\", "/");

        List<String> missingDateFiles = FileUtils.findMissingDirectories(storagePath, DateUtils.getLastSevenDays());

        if(missingDateFiles.isEmpty()) {
            log.info("No missing date files found, skipping script execution");
            return;
        }

        try {
            ProcessBuilder builder = buildDockerProcess(outputDir, missingDateFiles);
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
            log.error("Failed to execute Python script for dates: {}", missingDateFiles, e);
            throw new RuntimeException("Failed to run Python script", e);
        }
    }

    private ProcessBuilder buildDockerProcess(String outputDir, List<String> missingDates) {
        List<String> command = new ArrayList<>();
        command.add("docker");
        command.add("run");
        command.add("--rm");
        command.add("-v");
        command.add(outputDir + ":/data");
        command.add("gsla-py-script");
        command.add("--output_base_dir");
        command.add("/data");
        command.add("--dates");
        command.addAll(missingDates);

        log.info("Executing Docker command: {}", String.join(" ", command));
        return new ProcessBuilder(command);
    }
}