# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Expose models volume
VOLUME /app/models

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
# We copy README.md just in case it's needed for installation, though usually pyproject.toml is enough
COPY README.md .

# Install python dependencies first to leverage caching
RUN pip install --no-cache-dir .

# Copy source code after dependencies are installed
COPY src/ ./src/

# Default command
CMD ["python", "src/train.py"]
