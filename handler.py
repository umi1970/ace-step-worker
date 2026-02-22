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
# Config — check ALL required env vars upfront
# ---------------------------------------------------------------------------
REQUIRED_ENV = ["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"]
missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
if missing:
    raise RuntimeError(
        f"[ACE-Step] Missing required environment variables: {', '.join(missing)}\n"
        f"Set these in your RunPod Template under Environment Variables."
    )

LORA_FILENAME = os.environ.get("LORA_FILENAME", "adapter_model.safetensors")
LORA_SCALE = float(os.environ.get("LORA_SCALE", "1.0"))
LORA_DIR = "/tmp/lora"
LORA_ADAPTER_DIR = os.path.join(LORA_DIR, "adapter")  # load_lora expects adapter/ subdir
LORA_LOCAL_PATH = os.path.join(LORA_ADAPTER_DIR, LORA_FILENAME)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

print(f"[ACE-Step] Config: SUPABASE_URL={SUPABASE_URL[:30]}..., LORA={LORA_FILENAME}, SCALE={LORA_SCALE}")
BUCKET_NAME = "mastered-songs"
LORA_BUCKET = "ace-lora"

# ACE-Step v1.5 repo is cloned to /app/ace-step-repo by Dockerfile
PROJECT_ROOT = "/app/ace-step-repo"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------------------------------------------------------
# Download LoRA from Supabase at cold start
# ---------------------------------------------------------------------------
def download_lora():
    """Download LoRA weights + config from Supabase ace-lora bucket.

    Uses a version.txt file in the bucket to detect LoRA updates.
    If the remote version differs from the cached version, re-downloads everything.
    """
    os.makedirs(LORA_ADAPTER_DIR, exist_ok=True)

    config_path = os.path.join(LORA_ADAPTER_DIR, "adapter_config.json")
    version_path = os.path.join(LORA_DIR, "version.txt")

    # Check remote version to detect LoRA updates
    needs_download = not os.path.exists(LORA_LOCAL_PATH)
    remote_version = None

    try:
        remote_version = supabase.storage.from_(LORA_BUCKET).download("version.txt").decode("utf-8").strip()
        local_version = None
        if os.path.exists(version_path):
            with open(version_path, "r") as f:
                local_version = f.read().strip()

        if remote_version != local_version:
            print(f"[ACE-Step] LoRA version changed: '{local_version}' -> '{remote_version}' — re-downloading")
            needs_download = True
        else:
            print(f"[ACE-Step] LoRA version matches: '{remote_version}'")
    except Exception as e:
        print(f"[ACE-Step] No version.txt in bucket (ok, using file-exists check): {e}")

    # Download adapter_model.safetensors
    if needs_download:
        print(f"[ACE-Step] Downloading LoRA '{LORA_FILENAME}' from Supabase...")
        dl_start = time.time()
        data = supabase.storage.from_(LORA_BUCKET).download(LORA_FILENAME)
        with open(LORA_LOCAL_PATH, "wb") as f:
            f.write(data)
        print(f"[ACE-Step] LoRA weights downloaded in {time.time() - dl_start:.1f}s "
              f"({len(data) / 1024 / 1024:.1f}MB)")

        # Download adapter_config.json (v1.5 REQUIRES this file!)
        print("[ACE-Step] Downloading adapter_config.json from Supabase...")
        config_data = supabase.storage.from_(LORA_BUCKET).download("adapter_config.json")
        with open(config_path, "wb") as f:
            f.write(config_data)
        print("[ACE-Step] adapter_config.json downloaded")

        # Save version marker
        if remote_version:
            with open(version_path, "w") as f:
                f.write(remote_version)
            print(f"[ACE-Step] Version marker saved: '{remote_version}'")
    else:
        print(f"[ACE-Step] LoRA weights already cached at {LORA_LOCAL_PATH}")
        # Still ensure config exists
        if not os.path.exists(config_path):
            print("[ACE-Step] Downloading adapter_config.json from Supabase...")
            config_data = supabase.storage.from_(LORA_BUCKET).download("adapter_config.json")
            with open(config_path, "wb") as f:
                f.write(config_data)
            print("[ACE-Step] adapter_config.json downloaded")


# ---------------------------------------------------------------------------
# Load model (runs once at cold start) — ACE-Step v1.5 API
# ---------------------------------------------------------------------------
print("[ACE-Step] Loading model (v1.5 API)...")
load_start = time.time()

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
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

# Step 1b: Initialize LLM handler (0.6B model, needed for auto-duration & CoT)
llm_handler = LLMHandler()
llm_status, llm_success = llm_handler.initialize(
    checkpoint_dir=PROJECT_ROOT,
    lm_model_path="acestep-5Hz-lm-0.6B",
    backend="vllm",
    device="cuda",
)
print(f"[ACE-Step] LLM init status: {llm_status}")
print(f"[ACE-Step] LLM init success: {llm_success}, llm_initialized={llm_handler.llm_initialized}")
if not llm_handler.llm_initialized:
    print("[ACE-Step] WARNING: LLM failed to initialize! Duration auto-detection will NOT work.")
    print("[ACE-Step] Songs will default to 30 seconds without LLM.")

# Step 2: Download and load LoRA (MUST succeed — no silent fallback!)
download_lora()

print(f"[ACE-Step] Loading LoRA from {LORA_DIR} (scale={LORA_SCALE})")
print(f"[ACE-Step] LoRA dir contents: {os.listdir(LORA_DIR)}")
print(f"[ACE-Step] LoRA adapter dir contents: {os.listdir(LORA_ADAPTER_DIR)}")
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
    duration = input_data.get("duration", -1)  # -1 = auto (requires LLM for inference)
    num_songs = min(input_data.get("num_songs", 2), 4)

    # Cover mode parameters
    task_type = input_data.get("task_type", "text2music")
    cover_audio_url = input_data.get("cover_audio_url")
    audio_cover_strength = float(input_data.get("audio_cover_strength", 0.5))

    # Download cover audio from Supabase if cover mode
    reference_audio_path = None
    if task_type == "cover" and cover_audio_url:
        import requests as req
        print(f"[ACE-Step] Downloading cover audio from {cover_audio_url[:80]}...")
        resp = req.get(cover_audio_url)
        reference_audio_path = f"/tmp/cover_{uuid.uuid4().hex[:8]}.mp3"
        with open(reference_audio_path, "wb") as f:
            f.write(resp.content)
        print(f"[ACE-Step] Cover audio downloaded: {len(resp.content)} bytes")

    # Per-request LoRA scale override (from UI slider)
    request_lora_scale = input_data.get("lora_scale")
    if request_lora_scale is not None:
        scale = float(request_lora_scale)
        dit_handler.set_lora_scale(scale)
        print(f"[ACE-Step] Using per-request lora_scale={scale}")
    else:
        dit_handler.set_lora_scale(LORA_SCALE)
        print(f"[ACE-Step] Using default lora_scale={LORA_SCALE}")

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
                task_type=task_type,
                caption=caption,
                lyrics=lyrics,
                duration=float(duration),
                vocal_language="tr",  # Turkish LoRA
                inference_steps=8,    # Turbo model = 8 steps
                guidance_scale=7.0,
                seed=seed,
                shift=3.0,            # Recommended for turbo
                infer_method="ode",   # Deterministic, faster
                **({"reference_audio": reference_audio_path, "audio_cover_strength": audio_cover_strength} if reference_audio_path else {}),
            )

            config = GenerationConfig(
                batch_size=1,
                audio_format="wav",   # We convert to MP3 ourselves
            )

            # Generate using v1.5 inference API
            # LLM handler enables Chain-of-Thought for auto-duration & metadata
            result = generate_music(
                dit_handler=dit_handler,
                llm_handler=llm_handler,
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

    # Cleanup temp cover audio file
    if reference_audio_path and os.path.exists(reference_audio_path):
        os.unlink(reference_audio_path)
        print("[ACE-Step] Cleaned up temp cover audio file")

    return {"songs": songs}


runpod.serverless.start({"handler": handler})
