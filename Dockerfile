# Gunakan base image Python
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Buat direktori kerja
WORKDIR /app

# Salin semua file ke dalam container
COPY . /app

# Install pip dan dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port untuk Cloud Run
EXPOSE 8080

# Jalankan aplikasi Flask (gunakan uvicorn untuk production)
CMD ["uvicorn", "app.flask_api:app", "--host", "0.0.0.0", "--port", "8080"]
