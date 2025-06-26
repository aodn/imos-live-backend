package com.imos.imos_mapbox_server.scheduler;

import com.imos.imos_mapbox_server.service.S3Service;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalTime;
import java.util.List;

@Component
@Slf4j
@RequiredArgsConstructor
@ConditionalOnProperty(
        name = "app.scheduling.taskS3Upload.enabled",
        havingValue = "true",
        matchIfMissing = true
)
public class S3UploadScheduler {
    private final S3Service s3Service;
    @Value("${app.storage.path}")
    private String sourceDirectory;

    @Scheduled(cron = "0 0 * * * *")
    public void uploadFilesHourly() {
        if(shouldRunUpload()) {
            log.info("Starting hourly scheduled file upload");
            processAndUploadFiles();
        }
    }
    //TODO: upload wave buoys data to S3 and do not upload repeated ones.
    public void processAndUploadFiles() {
        try {
            Path sourceDir = Paths.get(sourceDirectory);

            if (!Files.exists(sourceDir)) {
                log.warn("Source directory does not exist: {}", sourceDirectory);
                return;
            }

            List<Path> filesToUpload = Files.walk(sourceDir)//recursively get all files
                    .filter(Files::isRegularFile)
                    .filter(path->shouldUploadFile(path, sourceDir))
                    .toList();

            if (filesToUpload.isEmpty()) {
                log.info("No files found to upload");
                return;
            }

            for (Path file : filesToUpload) {
                String key = generateS3Key(file,sourceDir);
                s3Service.uploadFile(key, file);
                handleFileAfterUpload(file);
            }

            log.info("Completed uploading {} files", filesToUpload.size());

        } catch (Exception e) {
            log.error("Error during scheduled file upload: {}", e.getMessage(), e);
        }
    }

    private boolean shouldRunUpload() {
        LocalTime now = LocalTime.now();
        return now.isAfter(LocalTime.of(9, 0)) && now.isBefore(LocalTime.of(11, 0));
    }

    private boolean shouldUploadFile(Path file,Path sourceDir) {
        String fileName = file.getFileName().toString();
        String s3Key = generateS3Key(file, sourceDir);
        // Skip hidden files and system files
        if (fileName.startsWith(".")) {
            return false;
        }
        //if files already exited, skip.
        if(s3Service.objectExists(s3Key)) return false;

        return (fileName.endsWith(".png") || fileName.endsWith(".json"));
    }

    private String generateS3Key(Path file,Path baseDir) {
        Path relativePath = baseDir.relativize(file);
        // This will create keys like: uploads/GSLA/subdir/filename.txt
        return String.format("uploads/GSLA/%s", relativePath.toString().replace("\\", "/"));
    }

    private void handleFileAfterUpload(Path file) {
        try {
             Files.delete(file);
        } catch (IOException e) {
            log.error("Error handling file after upload: {}", e.getMessage(), e);
        }
    }
}
