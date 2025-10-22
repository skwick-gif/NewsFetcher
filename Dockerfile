# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies - retry with different approach
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./
COPY config.yaml ./

# Create necessary directories
RUN mkdir -p logs data temp

# Expose port
EXPOSE 8000

# Use the production application with working imports
CMD ["uvicorn", "main_realtime:app", "--host", "0.0.0.0", "--port", "8000"]