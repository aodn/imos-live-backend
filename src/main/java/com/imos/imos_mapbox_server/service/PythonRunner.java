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
@EnableAsync
public class PythonRunner {

    @Value("${app.storage.path}")
    private String storagePath;

    @Async
    @Scheduled(cron = "0 40 09 * * ?")
    public void runScript()  {
        StringBuilder output = new StringBuilder();
        String outputDir = new File(storagePath).getAbsolutePath().replace("\\", "/");

        List<String> missingDateFiles = FileUtils.findMissingDirectories(storagePath, DateUtils.getLastSevenDays());

        if(missingDateFiles.isEmpty()) return;

        try {
            ProcessBuilder builder = buildDockerProcess(outputDir, missingDateFiles);

            builder.redirectErrorStream(true);
            Process process = builder.start();

            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line).append(System.lineSeparator());
                }
            }

            int exitCode = process.waitFor();
            if (exitCode == 139) {
                List<String> dateFiles = FileUtils.findMissingDirectories(storagePath, DateUtils.getLastSevenDays());
                if(!dateFiles.isEmpty()) throw new RuntimeException("Python script exited with code: " + exitCode);
            } else if (exitCode != 0) {
                log.error("Python script failed with output:{}", output);
                throw new RuntimeException("Python script exited with code: " + exitCode);
            }

        } catch (Exception e) {
            throw new RuntimeException("Failed to run Python script", e);
        }
    }

    private ProcessBuilder buildDockerProcess(String outputDir, List<String> dynamicArgs) {
        List<String> command = new ArrayList<>();
        command.add("docker");
        command.add("run");
        command.add("--rm");
        command.add("-v");
        command.add(outputDir + ":/data");
        command.add("gsla-py-script");
        command.add("/data");
        command.addAll(dynamicArgs);
        return new ProcessBuilder(command);
    }
}
