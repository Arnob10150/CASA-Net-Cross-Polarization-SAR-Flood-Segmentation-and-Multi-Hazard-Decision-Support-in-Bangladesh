# CASA-Net Deployment Guide

This guide covers every supported deployment path for the CASA-Net Streamlit application.

---

## Table of Contents

1. [Local Development](#1-local-development)
2. [Streamlit Cloud (free, recommended)](#2-streamlit-cloud)
3. [Hugging Face Spaces](#3-hugging-face-spaces)
4. [Docker (self-hosted / any cloud)](#4-docker)
5. [Cloud Platforms (AWS / GCP / Azure)](#5-cloud-platforms)
6. [Environment Variables Reference](#6-environment-variables-reference)
7. [Providing Model Weights](#7-providing-model-weights)

---

## 1. Local Development

### Prerequisites
- Python 3.10+
- (Optional) NVIDIA GPU + CUDA 11.8+

### Steps

```bash
# Clone the repository
git clone https://github.com/Arnob10150/CASA-Net-Cross-Polarization-SAR-Flood-Segmentation-and-Multi-Hazard-Decision-Support-in-Bangladesh.git
cd CASA-Net-Cross-Polarization-SAR-Flood-Segmentation-and-Multi-Hazard-Decision-Support-in-Bangladesh

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) copy and edit environment variables
cp .env.example .env
# edit .env as needed

# Run the app
streamlit run streamlit_app.py
```

Open your browser at **http://localhost:8501**.

---

## 2. Streamlit Cloud

[Streamlit Cloud](https://streamlit.io/cloud) offers free public hosting for open-source
repositories.

### Steps

1. Push this repository to **GitHub** (public or private).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **"New app"** → select the repository, branch, and set:
   - **Main file path**: `streamlit_app.py`
4. Under **Advanced settings → Secrets**, add any required environment variables
   (see [Section 6](#6-environment-variables-reference)).
5. Click **Deploy**.

> **Note**: Streamlit Cloud free tier uses CPU-only containers. The app automatically
> falls back to CPU when no GPU is available, or when `CASA_FORCE_CPU=1`.

### Secrets format (Streamlit Cloud)

```toml
# .streamlit/secrets.toml  (do NOT commit this file)
CASA_FORCE_CPU = "1"
MULTI_HAZARD_MODE = "flood,erosion,landslide"
```

---

## 3. Hugging Face Spaces

[Hugging Face Spaces](https://huggingface.co/spaces) supports Streamlit apps natively.

### Steps

1. Create a new Space on <https://huggingface.co/new-space>:
   - **SDK**: Streamlit
   - **Visibility**: Public or Private
2. Push this repository to the Space:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/<space-name>
   git push hf main
   ```
3. The Space will build automatically using `requirements.txt`.
4. Add environment variables under **Settings → Variables and secrets**.

> Hugging Face Spaces free tier is CPU-only. Upgrade to a GPU Space for faster inference.

---

## 4. Docker

### Build and run locally

```bash
# Build the image
docker build -t casa-net-app .

# CPU-only run
docker run -p 8501:8501 casa-net-app

# With a pre-trained weights file mounted
docker run -p 8501:8501 \
  -v /path/to/models:/app/models \
  -e CASA_FORCE_CPU=0 \
  casa-net-app

# GPU-enabled run (requires nvidia-container-toolkit)
docker run --gpus all -p 8501:8501 \
  -e CASA_FORCE_CPU=0 \
  casa-net-app
```

Open **http://localhost:8501** in your browser.

### Docker Compose (with environment file)

Create a `docker-compose.yml`:

```yaml
version: "3.9"
services:
  casa-net:
    build: .
    ports:
      - "8501:8501"
    env_file:
      - .env
    volumes:
      - ./models:/app/models   # mount trained weights
    restart: unless-stopped
```

Then run:

```bash
docker compose up -d
```

---

## 5. Cloud Platforms

### AWS (Elastic Container Service / App Runner)

1. Build and push the Docker image to **Amazon ECR**:
   ```bash
   aws ecr get-login-password --region us-east-1 \
     | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
   docker tag casa-net-app:latest <account>.dkr.ecr.us-east-1.amazonaws.com/casa-net-app:latest
   docker push <account>.dkr.ecr.us-east-1.amazonaws.com/casa-net-app:latest
   ```
2. Create an **App Runner** service pointing to the ECR image.  
   Set environment variables and expose port `8501`.

### Google Cloud Run

```bash
# Authenticate and build
gcloud auth configure-docker
docker tag casa-net-app gcr.io/<project-id>/casa-net-app
docker push gcr.io/<project-id>/casa-net-app

# Deploy
gcloud run deploy casa-net-app \
  --image gcr.io/<project-id>/casa-net-app \
  --platform managed \
  --region us-central1 \
  --port 8501 \
  --allow-unauthenticated \
  --set-env-vars CASA_FORCE_CPU=1,MULTI_HAZARD_MODE=flood
```

### Azure Container Instances

```bash
az container create \
  --resource-group myResourceGroup \
  --name casa-net-app \
  --image <registry>/casa-net-app:latest \
  --ports 8501 \
  --environment-variables CASA_FORCE_CPU=1 MULTI_HAZARD_MODE=flood \
  --dns-name-label casa-net-demo
```

---

## 6. Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `CASA_FORCE_CPU` | `0` | Set to `1` to force CPU-only inference (no GPU required) |
| `MULTI_HAZARD_MODE` | `flood` | Comma-separated active hazard types: `flood`, `erosion`, `landslide` |

Additional variables used by data-download notebooks (not required for the Streamlit app)
are documented in `.env.example`.

---

## 7. Providing Model Weights

The Streamlit app supports uploading weights directly through the UI:

1. On the **Flood Inference** page, click **"Model weights (optional)"**.
2. Upload a `.pth` or `.pt` checkpoint file saved with `torch.save(model.state_dict(), ...)`.
3. The app will load the weights and run inference immediately.

To bake weights into the Docker image, place the file at `models/casa_net.pth` and
uncomment the `COPY` line in the `Dockerfile`.

For Streamlit Cloud / Hugging Face Spaces, use
[Git LFS](https://git-lfs.github.com/) to track large checkpoint files:

```bash
git lfs install
git lfs track "models/*.pth"
git add .gitattributes models/casa_net.pth
git commit -m "add model weights via LFS"
git push
```
