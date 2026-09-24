FROM python:3.12-slim

WORKDIR /app

# System deps: libgl/libglib needed by Pillow/torchvision image codecs on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch/torchvision from PyTorch's CPU wheel index.
# This is NOT the same as the project's local pyproject.toml, which points
# at the cu128 (GPU) index for local training -- the deployed container has
# no GPU, so CPU wheels are both correct and dramatically smaller (~200MB
# vs several GB for CUDA wheels).
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only the two files inference actually needs -- not features.pkl or the
# train/val/test CSVs, which are training-time artifacts.
COPY artifacts/best_model.pt artifacts/best_model.pt
COPY artifacts/vocab.pkl artifacts/vocab.pkl

COPY src/ src/
COPY app.py .

# Pre-download ResNet-50's ImageNet weights at build time, not at container
# startup -- otherwise every fresh container's first request stalls on a
# network fetch, and a container with no internet egress would fail outright.
RUN python -c "from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.DEFAULT)"

ENV IN_DOCKER=1
EXPOSE 7860

CMD ["python", "app.py"]
