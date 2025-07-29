FROM python:3.12

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy Python scripts
COPY ./scripts/wave_buoys_processing_script.py .
COPY ./scripts/gsla_processing_script.py .

# Copy entrypoint script
COPY entrypoint.sh .
RUN dos2unix entrypoint.sh && chmod +x entrypoint.sh

# Create data directory
RUN mkdir -p /data

# Set the entrypoint
ENTRYPOINT ["./entrypoint.sh"]