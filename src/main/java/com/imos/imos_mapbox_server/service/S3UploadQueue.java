package com.imos.imos_mapbox_server.service;

import com.imos.imos_mapbox_server.dto.UploadTask;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.nio.file.Path;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;

import static com.imos.imos_mapbox_server.enums.UploadType.BUOY;
import static com.imos.imos_mapbox_server.enums.UploadType.GSLA;

@Service
@Slf4j
@RequiredArgsConstructor
public class S3UploadQueue {
    private final S3Service s3Service;
    private final BlockingQueue<UploadTask> uploadQueue = new LinkedBlockingQueue<>();
    private final ExecutorService uploadExecutor = Executors.newSingleThreadExecutor();

    @PostConstruct
    public void startUploadProcessor() {
        uploadExecutor.submit(this::processUploads);
    }

    public void queueGslaUpload(Path outputDir) {
        boolean added = uploadQueue.offer(new UploadTask(GSLA, outputDir));
        if (added) {
            log.info("Queued GSLA upload for: {}", outputDir);
        } else {
            log.error("Failed to queue GSLA upload for {}", outputDir);
        }
    }

    public void queueBuoyUpload(Path outputDir) {
        boolean added = uploadQueue.offer(new UploadTask(BUOY, outputDir));
        if (added) {
            log.info("Queued BUOY upload for: {}", outputDir);
        } else {
            log.error("Failed to queue BUOY upload for {}", outputDir);
        }
    }

    private void processUploads() {
        while (!Thread.currentThread().isInterrupted()) {
            try {
                UploadTask task = uploadQueue.take();
                s3Service.processAndUploadFiles(task.getOutputDir());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                log.info("Upload processor interrupted");
                break;
            } catch (Exception e) {
                log.error("Error processing upload task", e);
            }
        }
    }

    @PreDestroy
    public void shutdown() {
        uploadExecutor.shutdown();
    }
}


