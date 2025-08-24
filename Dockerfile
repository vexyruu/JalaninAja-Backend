# Use lightweight Python base
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies for Pillow, numpy, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install all dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model
COPY main.py .
COPY best.pt ./best.pt

# Expose port for Cloud Run
EXPOSE 8080

# Add this line to force CPU usage and prevent CUDA errors
ENV NVIDIA_VISIBLE_DEVICES=""

# Your command to start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
