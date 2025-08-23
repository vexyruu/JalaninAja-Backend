# Dockerfile

# Gunakan base image resmi dari Python
FROM python:3.11-slim

# --- PERUBAHAN: Install SEMUA library sistem yang hilang ---
# Menambahkan libglib2.0-0 untuk mengatasi error libgthread
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Set working directory di dalam container
WORKDIR /app

# Salin file requirements dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi ke dalam container
COPY . .

# Jalankan server Uvicorn saat container dimulai
# Port 8080 adalah port default yang didengarkan oleh Cloud Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
