FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone ACE-Step v1.5 repo and install (LoRA uses v1.5 target modules!)
RUN git clone https://github.com/ace-step/ACE-Step-1.5.git /app/ace-step-repo && \
    cd /app/ace-step-repo && \
    pip install --no-cache-dir -e .

# Copy handler
COPY handler.py .

# Default env vars (override in RunPod Template)
ENV LORA_FILENAME=adapter_model.safetensors
ENV LORA_SCALE=0.2
ENV OUTPUT_FORMAT=mp3

CMD ["python", "-u", "handler.py"]
