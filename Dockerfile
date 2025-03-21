FROM python:3.12-slim

WORKDIR /app

# Install dependencies required for packages
RUN apt-get update && apt-get install -y \
    gcc \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype-dev \
    libffi-dev \
    libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory
RUN mkdir -p models

# Copy application files
COPY app.py .
COPY predict.py .
COPY tests.py .
COPY models/model.onnx models/

# Expose port
EXPOSE 8000

# Run tests before starting the application
RUN pytest tests.py -v

# Run the application
CMD ["python", "app.py"]
