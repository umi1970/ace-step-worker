FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone ACE-Step v1.5 repo
RUN git clone --recurse-submodules https://github.com/ace-step/ACE-Step-1.5.git /app/ace-step-repo

# Install ACE-Step v1.5 dependencies manually (keep existing PyTorch from base image)
# The repo's requirements.txt wants PyTorch 2.10+cu128, but 2.4.0+cu124 works fine for inference
RUN pip install --no-cache-dir \
    "transformers>=4.51.0,<4.58.0" \
    "diffusers>=0.32.0" \
    "peft>=0.18.0" \
    "safetensors>=0.7.0" \
    "accelerate>=0.30.0" \
    "soundfile>=0.12.0" \
    "scipy>=1.10.0" \
    "einops>=0.7.0" \
    "loguru>=0.7.0" \
    "vector-quantize-pytorch>=1.14.0" \
    "toml>=0.10.0" \
    "numba>=0.59.0"

# Install the acestep package itself (no deps — already installed above)
RUN cd /app/ace-step-repo && pip install --no-cache-dir --no-deps -e .

# Install RunPod + Supabase
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# Default env vars (override in RunPod Template)
ENV LORA_FILENAME=adapter_model.safetensors
ENV LORA_SCALE=0.2

CMD ["python", "-u", "handler.py"]
