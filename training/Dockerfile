FROM nvidia/cuda:12.0.1-base-ubuntu22.04

# Build arguments
ARG MODE=prod

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODE=${MODE}

# Install Python and development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements.txt (if it exists)
COPY requirements_training.txt* .

# Install Python dependencies (if requirements.txt exists)
RUN if [ -f requirements_training.txt ]; then pip install --no-cache-dir -r requirements_training.txt; fi

# Copy the rest of the application (for production mode)
# In development mode, this will be overridden by volume mounting
COPY . .

# Build the Docker image without running any command by default
CMD ["echo", "Use 'docker run' with a command to start a container."]
