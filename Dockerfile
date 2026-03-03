FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (the package manager ACE-Step v1.5 uses)
RUN pip install --no-cache-dir uv

# Clone ACE-Step v1.5 repo
RUN git clone --recurse-submodules https://github.com/ace-step/ACE-Step-1.5.git /app/ace-step-repo

# Install all ACE-Step deps via uv (same as the working RunPod pod)
WORKDIR /app/ace-step-repo
RUN uv sync

# Install RunPod + Supabase into the uv environment
RUN uv pip install runpod>=1.7.0 supabase>=2.0.0

# Pre-download LLM model (4B, needed for auto-duration, CoT & melody copying)
# 4B is a separate HuggingFace repo (not a subfolder of Ace-Step1.5)
RUN uv pip install huggingface_hub && \
    uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('ACE-Step/acestep-5Hz-lm-4B', local_dir='/app/ace-step-repo/acestep-5Hz-lm-4B')"

# Copy handler
COPY handler.py /app/handler.py

WORKDIR /app
CMD ["uv", "run", "--project", "/app/ace-step-repo", "python", "-u", "/app/handler.py"]
