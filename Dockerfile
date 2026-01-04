# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Expose models volume
VOLUME /app/models

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
# We copy README.md just in case it's needed for installation, though usually pyproject.toml is enough
COPY README.md .

# Install python dependencies first to leverage caching
RUN pip install --no-cache-dir .

# Copy source code after dependencies are installed
COPY src/ ./src/

# Install the package in editable mode to ensure it's in the python path and dependencies are linked if needed
RUN pip install --no-deps -e .

# Default command using module execution to resolve imports correctly
CMD ["python", "-m", "src.train"]
