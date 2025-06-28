#!/bin/bash

# Enable error handling
set -e

# Get the script name from first argument
SCRIPT_NAME=$1

if [ -z "$SCRIPT_NAME" ]; then
    echo "Error: No script specified"
    echo "Usage: docker run <image> <script_name> [script_arguments...]"
    echo ""
    echo "Available scripts:"
    echo "  gsla_processing_script.py         - Process GSLA ocean current data"
    echo "  wave_buoys_processing_script.py   - Process wave buoy data"
    echo ""
    echo "Examples:"
    echo "  docker run --rm -v \"\${PWD}/output:/data\" ocean-data-processor gsla_processing_script.py --output_base_dir /data --dates 2025-01-01"
    echo "  docker run --rm -v \"\${PWD}/output:/data\" ocean-data-processor wave_buoys_processing_script.py --output_base_dir /data --dates 2025-01-01"
    exit 1
fi

# Check if script exists
SCRIPT_PATH="scripts/$SCRIPT_NAME"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script '$SCRIPT_NAME' not found in scripts directory"
    echo "Available scripts:"
    ls -1 scripts/*.py 2>/dev/null || echo "  No Python scripts found in scripts directory"
    exit 1
fi

# Remove script name from arguments and pass the rest to the Python script
shift

echo "Running: python $SCRIPT_PATH $@"
echo "Working directory: $(pwd)"
echo "Available space in /data: $(df -h /data | tail -1 | awk '{print $4}')"

# Run the specified Python script with any additional arguments passed to the Docker container
python "$SCRIPT_PATH" "$@"

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Script completed successfully!"
    echo "Generated files in /data:"
    find /data -type f -name "*.png" -o -name "*.json" -o -name "*.geojson" | head -20
    TOTAL_FILES=$(find /data -type f | wc -l)
    echo "Total files generated: $TOTAL_FILES"
else
    echo "Script failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE