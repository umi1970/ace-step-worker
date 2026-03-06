FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (the package manager ACE-Step v1.5 uses)
RUN pip install --no-cache-dir uv

# Clone ACE-Step v1.5 repo — pinned to 45d9783 (SFT support + multi-LoRA + LoRA UUID fix)
RUN git clone --recurse-submodules https://github.com/ace-step/ACE-Step-1.5.git /app/ace-step-repo && \
    cd /app/ace-step-repo && git checkout 45d9783

# Install all ACE-Step deps via uv (same as the working RunPod pod)
WORKDIR /app/ace-step-repo
RUN uv sync

# Install RunPod + Supabase into the uv environment
RUN uv pip install runpod>=1.7.0 supabase>=2.0.0 huggingface_hub

# ============================================================================
# Pre-download ALL models at build time (no cold-start downloads)
# ============================================================================

# 1. Main model (VAE + Qwen3 Embedding — shared by turbo and SFT)
RUN uv run python -c "\
from acestep.model_downloader import download_main_model; \
from pathlib import Path; \
download_main_model(Path('/app/ace-step-repo/checkpoints'))"

# 2. SFT DiT model (primary model)
RUN uv run python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('ACE-Step/acestep-v15-sft', local_dir='/app/ace-step-repo/checkpoints/acestep-v15-sft')"

# 3. 4B LLM (best CoT quality)
RUN uv run python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('ACE-Step/acestep-5Hz-lm-4B', local_dir='/app/ace-step-repo/acestep-5Hz-lm-4B')"

# Copy handler (last layer for fast rebuilds)
COPY handler.py /app/handler.py

WORKDIR /app
CMD ["uv", "run", "--project", "/app/ace-step-repo", "python", "-u", "/app/handler.py"]
