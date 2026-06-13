# Dockerfile for Symbiotic AI with HTK HMM State Detection
# This container includes Python, all dependencies, and HTK toolkit

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libgl1 \
    libglx0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Build HTK 3.4.1 from source
COPY HTK/ /tmp/HTK/

# Configure HTK (specify build type for ARM64)
RUN cd /tmp/HTK && \
    ./configure --build=aarch64-unknown-linux-gnu --prefix=/usr/local --disable-hlmtools 2>&1 | tee /tmp/configure.log

# Build HTK
RUN cd /tmp/HTK && \
    make all 2>&1 | tee /tmp/make.log && \
    echo "=== Build complete, looking for binaries ===" && \
    find . -type f -executable -name "H*" | head -20

# Install HTK binaries
RUN cd /tmp/HTK && \
    echo "=== Looking for HTK binaries ===" && \
    find . -type f -name "HVite" -o -name "HERest" -o -name "HCompV" -o -name "HParse" -o -name "HCopy" && \
    echo "=== Installing binaries ===" && \
    (make install 2>&1 || \
     find ./HTKTools -type f -executable ! -name "*.o" -exec cp {} /usr/local/bin/ \; ) && \
    ls -la /usr/local/bin/H* && \
    cd / && rm -rf /tmp/HTK

# Add HTK to PATH (once installed)
ENV PATH="/usr/local/bin:${PATH}"

# Copy only requirements first for better caching
COPY requirements_docker.txt /app/requirements_docker.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements_docker.txt

# Copy the symbiote codebases
COPY symbiote/ /app/symbiote/
COPY symbiote_weak/ /app/symbiote_weak/

# Create directories for data, models, and outputs
RUN mkdir -p /data/videos /data/annotations /data/aruco_config /models /outputs

# Set Python path
ENV PYTHONPATH="/app"

# Default command shows help
CMD ["python", "-m", "symbiote_weak.cli.main", "--help"]
