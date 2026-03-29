# CASA-Net Streamlit Application — Docker Image
#
# Build:  docker build -t casa-net-app .
# Run:    docker run -p 8501:8501 casa-net-app
# GPU:    docker run --gpus all -p 8501:8501 casa-net-app

# ---------------------------------------------------------------------------
# Base image — use the slim CPU build; override FROM for GPU deployments
# ---------------------------------------------------------------------------
FROM python:3.10-slim

# Allow custom model-weights path and environment flags to be passed at runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    CASA_FORCE_CPU=1 \
    MULTI_HAZARD_MODE=flood,erosion,landslide

WORKDIR /app

# ---------------------------------------------------------------------------
# System dependencies (GDAL / rasterio runtime libs)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgdal-dev \
        gdal-bin \
        libgeos-dev \
        libproj-dev \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Application source
# ---------------------------------------------------------------------------
COPY app.py ./
COPY .streamlit/ ./.streamlit/

# Optional: copy pre-trained model weights if present
# COPY models/casa_net.pth ./models/

# ---------------------------------------------------------------------------
# Streamlit runs on port 8501 by default
# ---------------------------------------------------------------------------
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.headless=true"]
