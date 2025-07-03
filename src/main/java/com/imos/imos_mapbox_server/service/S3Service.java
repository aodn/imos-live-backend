package com.imos.imos_mapbox_server.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import static com.imos.imos_mapbox_server.constant.GslaFileConstants.*;

@Slf4j
@RequiredArgsConstructor
@Service
public class S3Service {
    private final static  String BUOY_LOCATION_DIR="buoy_locations";
    private final static  String BUOY_DETAILS_DIR="buoy_details";

    private final S3Client s3Client;

    @Value("${cloud.aws.s3.bucket}")
    private String bucketName;

    @Value("${app.storage.path}")
    private String storagePath;


    private static final Set<String> EXCLUDED_FILES = Set.of(
           GSLA_DATA
    );

    public  void uploadFile(String key, Path filePath) {
        try{
            PutObjectRequest putObjectRequest = PutObjectRequest.builder()
                    .bucket(bucketName)
                    .key(key)
                    .build();

            s3Client.putObject(putObjectRequest, RequestBody.fromFile(filePath));
            log.info("Successfully Uploaded File: {} to S3", key);
        } catch (Exception e) {
            log.error("Error uploading file to S3: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to upload file to S3", e);
        }
    }


    public void processAndUploadFiles(Path sourceDir) {
        try {

            if (!Files.exists(sourceDir)) {
                log.warn("Source directory does not exist: {}", sourceDir);
                return;
            }

            List<Path> filesToUpload = Files.walk(sourceDir)//recursively get all files
                    .filter(Files::isRegularFile)
                    .filter(this::shouldUploadFile)
                    .toList();

            if (filesToUpload.isEmpty()) {
                log.info("No files found to upload");
                return;
            }

            for (Path file : filesToUpload) {
                String key = generateS3Key(file);
                uploadFile(key, file);
                handleFileAfterUpload(file);
            }

            log.info("Completed uploading {} files", filesToUpload.size());

        } catch (Exception e) {
            log.error("Error during scheduled file upload: {}", e.getMessage(), e);
        }
    }


    //List<String> dates expect to be YYYY-mm-dd
    public List<String> findGslaMissingFiles(Path sourceDir, List<String> dates) {
        List<String> missingFiles = new ArrayList<>();

        for (String date : dates) {
            try {
                Path overlayFile = sourceDir.resolve(date).resolve(GSLA_OVERLAY);
                String overlayKey = generateS3Key(overlayFile);

                Path inputFile = sourceDir.resolve(date).resolve(GSLA_INPUT);
                String inputKey = generateS3Key(inputFile);

                Path metaFile = sourceDir.resolve(date).resolve(GSLA_META);
                String metaKey = generateS3Key(metaFile);

                if (!objectExists(overlayKey) || !objectExists(inputKey) || !objectExists(metaKey)) {
                    missingFiles.add(date);
                    log.debug("Missing files for date: {}", date);
                }
            } catch (Exception e) {
                log.error("Error checking files for date {}: {}", date, e.getMessage());
                missingFiles.add(date);
            }
        }

        return missingFiles;
    }

    //List<String> dates expect to be YYYY-mm-dd
    public List<String> findBuoyMissingFiles(Path sourceDir, List<String> dates) {
        String locationPrefix = generateS3Key(sourceDir.resolve(BUOY_LOCATION_DIR)) ;
        String detailsPrefix = generateS3Key(sourceDir.resolve(BUOY_DETAILS_DIR));
        List<String> missingFiles = new ArrayList<>();
        for (String date : dates) {
            if(!objectExists(locationPrefix, date) || !objectExists(detailsPrefix, date)) {
                missingFiles.add(date);
            }
        }

        String extraDate=getExtraDate();
        missingFiles.add(extraDate);

        return missingFiles;
    }

    private String getExtraDate() {
        LocalDate now = LocalDate.now();
        LocalDate date = now.getDayOfMonth() == 1
                ? now.minusMonths(1).withDayOfMonth(1)
                : now.withDayOfMonth(1);
        return date.format(DateTimeFormatter.ISO_LOCAL_DATE);
    }

    private boolean shouldUploadFile(Path file) {
        String fileName = file.getFileName().toString();

        if(EXCLUDED_FILES.contains(fileName)) {
            return false;
        }

        // Skip hidden files and system files
        if (fileName.startsWith(".")) {
            return false;
        }
        return (fileName.endsWith(".png") || fileName.endsWith(".json")|| fileName.endsWith(".geojson"));
    }


    private String generateS3Key(Path file) {
        Path relativePath = Paths.get(storagePath).relativize(file);
        // This will create keys like: uploads/subdir/filename.txt
        return String.format("uploads/%s", relativePath.toString().replace("\\", "/"));
    }


    private void handleFileAfterUpload(Path file) {
        try {
            Files.delete(file);
        } catch (IOException e) {
            log.error("Error handling file after upload: {}", e.getMessage(), e);
        }
    }


    private boolean objectExists(String key) {
        try {
            s3Client.headObject(HeadObjectRequest.builder()
                    .bucket(bucketName)
                    .key(key)
                    .build());
            return true;
        }catch (NoSuchKeyException e) {
            return false;
        }catch (Exception e) {
            log.warn("Error checking if object exists in S3: {}", e.getMessage());
            return false;
        }
    }


    private boolean objectExists(String prefix, String searchText) {
        try {
            ListObjectsV2Request listRequest = ListObjectsV2Request.builder()
                    .bucket(bucketName)
                    .prefix(prefix)
                    .build();

            ListObjectsV2Response listResponse = s3Client.listObjectsV2(listRequest);

            return listResponse.contents().stream()
                    .anyMatch(s3Object -> s3Object.key().contains(searchText));

        } catch (Exception e) {
            log.error("Error searching for keys with prefix containing text: {}", e.getMessage(), e);
            return false;
        }
    }
}
