# Use the same base as before
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies remain the same
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- EXPLICITLY COPY ALL NECESSARY FILES ---
COPY main.py .
COPY worker_app.py .
COPY best.onnx .
# This specifically copies the shared directory and its contents
COPY shared/ ./shared/

EXPOSE 8080
ENV NVIDIA_VISIBLE_DEVICES=""

# This command dynamically uses the port from Cloud Run
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}

