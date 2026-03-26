FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (the package manager ACE-Step v1.5 uses)
RUN pip install --no-cache-dir uv

# Clone ACE-Step v1.5 repo — always latest (unpinned)
RUN git clone --recurse-submodules https://github.com/ace-step/ACE-Step-1.5.git /app/ace-step-repo

# Install all ACE-Step deps via uv (same as the working RunPod pod)
WORKDIR /app/ace-step-repo
RUN uv sync

# Install nano-vllm (bundled in third_parts, needed for LLM backend="vllm")
RUN if [ -d "third_parts/nano-vllm" ]; then uv pip install -e third_parts/nano-vllm; fi

# Install RunPod + Supabase into the uv environment
RUN uv pip install runpod>=1.7.0 supabase>=2.0.0 huggingface_hub

# Copy handler (last layer for fast rebuilds)
COPY handler.py /app/handler.py

# ============================================================================
# OPTIMIZED: Models downloaded on-demand in cold start (not at build)
# This reduces image size and startup time dramatically
# First worker request will take time, but subsequent jobs are fast
# ============================================================================

WORKDIR /app

# Expose handler port for health checks
ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "--project", "/app/ace-step-repo", "python", "-u", "/app/handler.py"]
