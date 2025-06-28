# Ocean Data Processor Docker Container

This Docker container processes Australian ocean data from IMOS (Integrated Marine Observing System), including GSLA (Gridded Sea Level Anomaly) and wave buoy data. Designed for integration with Spring Boot applications.

## Features

- **GSLA Processing**: Downloads and processes sea level anomaly data with ocean currents
- **Wave Buoy Processing**: Processes wave buoy data from 20 locations around Australia
- **Concurrent Processing**: Optimized with multi-threading for faster data processing
- **Multiple Output Formats**: Generates PNG visualizations, JSON data files, and GeoJSON formats
- **Robust Error Handling**: Comprehensive logging and error recovery
- **Spring Boot Ready**: Designed for integration with Java applications via ProcessBuilder

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t ocean-data-processor .
```

### 2. Run GSLA Processing

Process sea level anomaly and ocean current data:

```bash
docker run --rm -v "/path/to/output:/data" ocean-data-processor \
    gsla_processing_script.py \
    --output_base_dir /data \
    --dates 2025-06-19
```

### 3. Run Wave Buoy Processing

Process wave buoy data from all Australian locations:

```bash
docker run --rm -v "/path/to/output:/data" ocean-data-processor \
    wave_buoys_processing_script.py \
    --output_base_dir /data \
    --dates 2025-06-19
```

## Spring Boot Integration

This container is designed to work seamlessly with Spring Boot applications using ProcessBuilder:

```java
private ProcessBuilder buildDockerProcess(String outputDir, String scriptName, List<String> dates) {
    List<String> command = new ArrayList<>();
    command.add("docker");
    command.add("run");
    command.add("--rm");
    command.add("-v");
    command.add(outputDir + ":/data");
    command.add("ocean-data-processor");  // Your image name
    command.add(scriptName);
    command.add("--output_base_dir");
    command.add("/data");
    command.add("--dates");
    command.addAll(dates);

    log.info("Executing Docker command: {}", String.join(" ", command));
    return new ProcessBuilder(command);
}
```

### Example Spring Boot Usage

```java
// Process GSLA data
List<String> dates = Arrays.asList("2025-06-19", "2025-06-20");
ProcessBuilder pb = buildDockerProcess("/tmp/ocean-data", "gsla_processing_script.py", dates);
Process process = pb.start();

// Process wave buoy data
ProcessBuilder pb2 = buildDockerProcess("/tmp/ocean-data", "wave_buoys_processing_script.py", dates);
Process process2 = pb2.start();
```

## Container Exit Codes

The container returns specific exit codes for Spring Boot error handling:

- **0**: Success - All processing completed successfully
- **1**: Error - Processing failed or was interrupted

### Monitoring Progress

You can monitor container output in Spring Boot:

```java
ProcessBuilder pb = buildDockerProcess(outputDir, scriptName, dates);
pb.redirectErrorStream(true);
Process process = pb.start();

try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
    String line;
    while ((line = reader.readLine()) != null) {
        log.info("Docker output: {}", line);
    }
}

int exitCode = process.waitFor();
if (exitCode == 0) {
    log.info("Docker processing completed successfully");
} else {
    log.error("Docker processing failed with exit code: {}", exitCode);
}
```

### GSLA Processing Script

Processes gridded sea level anomaly data with ocean currents.

**Command:**

```bash
docker run --rm -v "${PWD}/generated-images:/data" ocean-data-processor \
    gsla_processing_script.py \
    --output_base_dir /data \
    --dates 2025-06-19 2025-06-20 \
    --log_level INFO
```

**Output Files (per date):**

- `gsla_overlay.png` - Visualization overlay
- `gsla_input.png` - Input data as PNG with current vectors
- `gsla_meta.json` - Metadata including coordinate ranges
- `gsla_data.json` - Raw data in JSON format

### Wave Buoys Processing Script

Processes wave data from Australian coastal buoys.

**Command:**

```bash
docker run --rm -v "${PWD}/generated-images:/data" ocean-data-processor \
    wave_buoys_processing_script.py \
    --output_base_dir /data \
    --dates 2025-06-19 \
    --buoys APOLLO-BAY STORM-BAY \
    --max_workers 3 \
    --log_level INFO
```

**Available Buoys:**

- APOLLO-BAY, BENGELLO, BOB, BRIGHTON, CAPE-BRIDGEWATER
- CEDUNA, CENTRAL, COLLAROY, CORAL-BAY, HILLARYS
- KARUMBA, NORTH-KANGAROO-ISLAND, OCEAN-BEACH, ROBE
- SHARK-BAY, STORM-BAY, TANTABIDDI, TATHRA, TORBAY-WEST, WOOLI

**Output Files:**

- `buoy_locations/` - Daily GeoJSON files with buoy positions
- `buoy_details/` - Detailed hourly time series data per buoy per day

## Script Parameters

### Common Parameters

- `--output_base_dir`: Directory where files will be saved (required)
- `--dates`: Space-separated list of dates in YYYY-MM-DD format (required)
- `--log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Wave Buoys Specific Parameters

- `--buoys`: Space-separated list of buoy names (optional, defaults to all buoys)
- `--max_workers`: Maximum concurrent processing threads (default: 5)

## Data Sources

- **GSLA Data**: IMOS Ocean Current GSLA NRT from AWS S3
- **Wave Buoy Data**: IMOS Coastal Wave Buoys from THREDDS servers

## Output Directory Structure

```
generated-images/
├── 2025-06-19/                    # GSLA outputs
│   ├── gsla_overlay.png
│   ├── gsla_input.png
│   ├── gsla_meta.json
│   └── gsla_data.json
├── buoy_locations/                # Wave buoy locations
│   └── buoy_locations_2025-06-19.geojson
└── buoy_details/           # Detailed wave data
    ├── APOLLO-BAY_2025-06-19.geojson
    ├── STORM-BAY_2025-06-19.geojson
    └── ...
```

## Troubleshooting

### Common Issues

1. **Permission Errors**: Ensure the output directory is writable

   ```bash
   chmod 755 generated-images
   ```

2. **Memory Issues**: Reduce `--max_workers` for wave buoy processing

   ```bash
   --max_workers 2
   ```

3. **Network Timeouts**: Check internet connection and retry

### Debug Mode

Run with debug logging for detailed information:

```bash
docker run --rm -v "${PWD}/generated-images:/data" ocean-data-processor \
    gsla_processing_script.py \
    --output_base_dir /data \
    --dates 2025-06-19 \
    --log_level DEBUG
```

### Interactive Container

For debugging, run an interactive container:

```bash
docker run --rm -it -v "${PWD}/generated-images:/data" ocean-data-processor bash
```

## Development

### File Structure

```
.
├── Dockerfile
├── requirements.txt
├── entrypoint.sh
├── scripts/
│   ├── gsla_processing_script.py
│   └── wave_buoys_processing_script.py
└── README.md
```

### Building for Production

1. Build the container:

   ```bash
   docker build -t ocean-data-processor .
   ```

2. Tag for your registry:

   ```bash
   docker tag ocean-data-processor your-registry/ocean-data-processor:latest
   ```

3. Push to registry:
   ```bash
   docker push your-registry/ocean-data-processor:latest
   ```

### Spring Boot Configuration

Add to your `application.yml`:

```yaml
ocean:
  processing:
    docker:
      image: ocean-data-processor:latest
      timeout: 3600000 # 1 hour timeout
    output:
      base-dir: /tmp/ocean-data
```

### Testing

Test with a single date first:

```bash
docker run --rm -v "${PWD}/test-output:/data" ocean-data-processor \
    gsla_processing_script.py \
    --output_base_dir /data \
    --dates $(date +%Y-%m-%d)
```

## Performance Notes

- GSLA processing: ~2-5 minutes per date
- Wave buoy processing: ~10-30 minutes depending on number of buoys and dates
- Concurrent processing can significantly speed up wave buoy data processing
- Memory usage scales with the number of concurrent workers and data size

## License

This container processes publicly available data from IMOS. Please refer to IMOS data usage policies for the underlying datasets.
