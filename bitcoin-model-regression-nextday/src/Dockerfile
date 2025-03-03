# Use Ubuntu with amd64 architecture
FROM --platform=linux/amd64 ubuntu:22.04

# Set working directory
WORKDIR /app

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip is upgraded
RUN pip3 install --upgrade pip

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy everything inside 'src/' into '/app/'
COPY . /app/

# Ensure output directory exists
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the prediction script
CMD ["python3", "/app/main.py"]
