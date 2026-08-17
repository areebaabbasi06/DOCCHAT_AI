FROM python:3.12-slim

WORKDIR /app

# System dependencies required by Docling, PyTorch and PDF processing
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Railway provides PORT
EXPOSE 8080

# Start FastAPI backend
CMD ["sh", "-c", "uvicorn application_backend.api:app --host 0.0.0.0 --port ${PORT}"]