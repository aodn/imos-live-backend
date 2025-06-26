#!/bin/bash

# Get the script name from first argument
SCRIPT_NAME=$1

if [ -z "$SCRIPT_NAME" ]; then
    echo "Error: No script specified"
    echo "Usage: docker run <image> <script_name> [script_arguments...]"
    exit 1
fi

# Remove script name from arguments and pass the rest to the Python script
shift

# this will run the specified Python script with any additional arguments passed to the Docker container
python "scripts/$SCRIPT_NAME" "$@"

