FROM python:3.12

WORKDIR /app

# Install system dependencies required for geospatial and scientific computing
RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libproj-dev \
    libeccodes-dev \
    gcc \
    g++ \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libnetcdf-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL
ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/share/proj

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install Python packages with specific versions to avoid conflicts
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy scripts folder and entrypoint
COPY scripts/ ./scripts/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Create data directory
RUN mkdir -p /data

ENTRYPOINT ["./entrypoint.sh"]