"""
RunPod Serverless Handler for ACE-Step v1.5 Music Generation

Uses ACE-Step v1.5 programmatic API (AceStepHandler + generate_music).
Loads model + Turkish LoRA at cold start.
LoRA is downloaded from Supabase Storage (ace-lora bucket) at startup.
Generates audio, uploads to Supabase Storage, returns URLs.

Env vars (set in RunPod Template):
  SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
  LORA_FILENAME (default: adapter_model.safetensors)
  LORA_SCALE (default: 0.2)
"""

import os
import uuid
import time
import tempfile
import subprocess
import glob as glob_mod

import runpod
from supabase import create_client

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LORA_FILENAME = os.environ.get("LORA_FILENAME", "adapter_model.safetensors")
LORA_SCALE = float(os.environ.get("LORA_SCALE", "0.2"))
LORA_DIR = "/tmp/lora"
LORA_LOCAL_PATH = os.path.join(LORA_DIR, LORA_FILENAME)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
BUCKET_NAME = "mastered-songs"
LORA_BUCKET = "ace-lora"

# ACE-Step v1.5 repo is cloned to /app/ace-step-repo by Dockerfile
PROJECT_ROOT = "/app/ace-step-repo"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------------------------------------------------------
# Download LoRA from Supabase at cold start
# ---------------------------------------------------------------------------
def download_lora():
    """Download LoRA weights + config from Supabase ace-lora bucket."""
    os.makedirs(LORA_DIR, exist_ok=True)

    config_path = os.path.join(LORA_DIR, "adapter_config.json")

    # Download adapter_model.safetensors
    if not os.path.exists(LORA_LOCAL_PATH):
        print(f"[ACE-Step] Downloading LoRA '{LORA_FILENAME}' from Supabase...")
        dl_start = time.time()
        data = supabase.storage.from_(LORA_BUCKET).download(LORA_FILENAME)
        with open(LORA_LOCAL_PATH, "wb") as f:
            f.write(data)
        print(f"[ACE-Step] LoRA weights downloaded in {time.time() - dl_start:.1f}s "
              f"({len(data) / 1024 / 1024:.1f}MB)")
    else:
        print(f"[ACE-Step] LoRA weights already cached at {LORA_LOCAL_PATH}")

    # Download adapter_config.json (v1.5 REQUIRES this file!)
    if not os.path.exists(config_path):
        print("[ACE-Step] Downloading adapter_config.json from Supabase...")
        config_data = supabase.storage.from_(LORA_BUCKET).download("adapter_config.json")
        with open(config_path, "wb") as f:
            f.write(config_data)
        print("[ACE-Step] adapter_config.json downloaded")
    else:
        print("[ACE-Step] adapter_config.json already cached")


# ---------------------------------------------------------------------------
# Load model (runs once at cold start) — ACE-Step v1.5 API
# ---------------------------------------------------------------------------
print("[ACE-Step] Loading model (v1.5 API)...")
load_start = time.time()

from acestep.handler import AceStepHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

# Step 1: Create DiT handler and initialize the service
dit_handler = AceStepHandler()
init_status, init_success = dit_handler.initialize_service(
    project_root=PROJECT_ROOT,
    config_path="acestep-v15-turbo",  # Turbo model (8-step inference)
    device="cuda",
    use_flash_attention=False,
    compile_model=False,
    offload_to_cpu=False,
    offload_dit_to_cpu=False,
    quantization=None,
    use_mlx_dit=False,  # Not on Apple Silicon
)
print(f"[ACE-Step] Init status: {init_status}")
if not init_success:
    raise RuntimeError(f"Failed to initialize ACE-Step: {init_status}")

# Step 2: Download and load LoRA (MUST succeed — no silent fallback!)
download_lora()

print(f"[ACE-Step] Loading LoRA from {LORA_DIR} (scale={LORA_SCALE})")
lora_msg = dit_handler.load_lora(LORA_DIR)
print(f"[ACE-Step] LoRA load result: {lora_msg}")

# v1.5 returns "❌ ..." on failure — crash if LoRA didn't load
if "❌" in str(lora_msg):
    raise RuntimeError(f"LoRA loading failed: {lora_msg}")

dit_handler.set_lora_scale(LORA_SCALE)
dit_handler.set_use_lora(True)
print(f"[ACE-Step] LoRA active with scale={LORA_SCALE}")

print(f"[ACE-Step] Model loaded in {time.time() - load_start:.1f}s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def convert_to_mp3(audio_path: str) -> str:
    """Convert any audio file to MP3 using ffmpeg."""
    mp3_path = os.path.splitext(audio_path)[0] + ".mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-b:a", "320k", "-q:a", "0", mp3_path],
        capture_output=True,
        check=True,
    )
    return mp3_path


def upload_to_supabase(file_path: str, filename: str) -> str:
    """Upload a file to Supabase Storage and return the public URL."""
    with open(file_path, "rb") as f:
        content_type = "audio/mpeg" if filename.endswith(".mp3") else "audio/wav"
        supabase.storage.from_(BUCKET_NAME).upload(
            filename,
            f.read(),
            {"content-type": content_type},
        )
    return f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{filename}"


# ---------------------------------------------------------------------------
# RunPod Handler
# ---------------------------------------------------------------------------
def handler(job):
    """
    Input:
      - lyrics (str): Song lyrics
      - caption (str): Style/prompt description
      - duration (int): Target duration in seconds (default 120)
      - num_songs (int): Number of songs to generate (default 2)

    Output:
      - songs: list of { url, duration, title }
    """
    input_data = job["input"]

    lyrics = input_data.get("lyrics", "")
    caption = input_data.get("caption", "")
    duration = input_data.get("duration", 120)
    num_songs = min(input_data.get("num_songs", 2), 4)

    songs = []
    job_id = job.get("id", str(uuid.uuid4())[:8])

    # Create temp output directory
    save_dir = tempfile.mkdtemp(prefix="ace_")

    for i in range(num_songs):
        gen_start = time.time()
        print(f"[ACE-Step] Generating song {i+1}/{num_songs}...")

        try:
            # Build generation parameters (v1.5 dataclass API)
            seed = int(time.time() * 1000 + i) % (2**31)
            params = GenerationParams(
                task_type="text2music",
                caption=caption,
                lyrics=lyrics,
                duration=float(duration),
                vocal_language="tr",  # Turkish LoRA
                inference_steps=8,    # Turbo model = 8 steps
                guidance_scale=7.0,
                seed=seed,
                shift=3.0,            # Recommended for turbo
                infer_method="ode",   # Deterministic, faster
            )

            config = GenerationConfig(
                batch_size=1,
                audio_format="wav",   # We convert to MP3 ourselves
            )

            # Generate using v1.5 inference API
            # llm_handler=None means no LM chain-of-thought (DiT-only, faster)
            result = generate_music(
                dit_handler=dit_handler,
                llm_handler=None,
                params=params,
                config=config,
                save_dir=save_dir,
            )

            if not result.success:
                raise RuntimeError(f"Generation failed: {result.error or result.status_message}")

            # result.audios is a list of dicts with 'path', 'tensor', 'key', etc.
            if not result.audios or len(result.audios) == 0:
                raise RuntimeError("No audio files in result")

            audio_info = result.audios[0]
            audio_path = audio_info.get("path")

            # If no path saved, try to find generated files in save_dir
            if not audio_path or not os.path.isfile(audio_path):
                found = glob_mod.glob(os.path.join(save_dir, "*.wav")) + \
                        glob_mod.glob(os.path.join(save_dir, "*.flac")) + \
                        glob_mod.glob(os.path.join(save_dir, "*.mp3"))
                if found:
                    audio_path = sorted(found)[-1]  # Most recent
                else:
                    raise RuntimeError("No audio file found in output directory")

            # Convert to MP3
            mp3_path = convert_to_mp3(audio_path)
            if os.path.exists(audio_path) and audio_path != mp3_path:
                os.unlink(audio_path)

            # Get actual duration via ffprobe
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", mp3_path],
                capture_output=True, text=True,
            )
            actual_duration = float(probe.stdout.strip()) if probe.stdout.strip() else duration

            # Upload to Supabase
            filename = f"ace-{job_id}_{i+1}.mp3"
            url = upload_to_supabase(mp3_path, filename)
            os.unlink(mp3_path)

            songs.append({
                "url": url,
                "duration": round(actual_duration, 1),
            })

            gen_time = time.time() - gen_start
            print(f"[ACE-Step] Song {i+1} done in {gen_time:.1f}s -> {url}")

        except Exception as e:
            print(f"[ACE-Step] Error generating song {i+1}: {e}")
            import traceback
            traceback.print_exc()
            songs.append({
                "url": None,
                "duration": 0,
                "error": str(e),
            })

    return {"songs": songs}


runpod.serverless.start({"handler": handler})
