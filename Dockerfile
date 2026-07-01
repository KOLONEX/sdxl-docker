FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    OUTPUT_DIR=/outputs \
    LORA_DIR=/models/loras \
    MODEL_DIR=/opt/models/sdxl-base \
    REFINER_DIR=/opt/models/sdxl-refiner \
    VAE_DIR=/opt/models/sdxl-vae \
    MAX_UPLOAD_MB=10 \
    MAX_BATCH=4 \
    DEFAULT_USE_REFINER=true

# 1. System deps + Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip libgl1 libglib2.0-0 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# 2. Torch (cu121) first
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 \
        --index-url https://download.pytorch.org/whl/cu121

# 3. App runtime deps
COPY requirements-api.txt /tmp/requirements-api.txt
RUN pip install --no-cache-dir -r /tmp/requirements-api.txt

# 4. Bake models (base + refiner + vae)
COPY scripts/download_models.py /opt/scripts/download_models.py
RUN python /opt/scripts/download_models.py

# 5. App code
WORKDIR /app
COPY app/ /app/app/

RUN mkdir -p /outputs /models/loras
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
