"""
RunPod Serverless Handler for ACE-Step Music Generation

Loads ACE-Step model + Turkish LoRA at cold start.
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
import base64
import tempfile
import subprocess

import runpod
from supabase import create_client

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LORA_FILENAME = os.environ.get("LORA_FILENAME", "adapter_model.safetensors")
LORA_SCALE = float(os.environ.get("LORA_SCALE", "0.2"))
LORA_DIR = "/tmp/lora"
LORA_LOCAL_PATH = os.path.join(LORA_DIR, "pytorch_lora_weights.safetensors")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
BUCKET_NAME = "ace-songs"
LORA_BUCKET = "ace-lora"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------------------------------------------------------
# Download LoRA from Supabase at cold start
# ---------------------------------------------------------------------------
def download_lora():
    """Download LoRA weights from Supabase ace-lora bucket."""
    os.makedirs(LORA_DIR, exist_ok=True)
    if os.path.exists(LORA_LOCAL_PATH):
        print(f"[ACE-Step] LoRA already cached at {LORA_LOCAL_PATH}")
        return True
    try:
        print(f"[ACE-Step] Downloading LoRA '{LORA_FILENAME}' from Supabase...")
        dl_start = time.time()
        data = supabase.storage.from_(LORA_BUCKET).download(LORA_FILENAME)
        # ACE-Step expects 'pytorch_lora_weights.safetensors' in the directory
        with open(LORA_LOCAL_PATH, "wb") as f:
            f.write(data)
        print(f"[ACE-Step] LoRA downloaded in {time.time() - dl_start:.1f}s ({len(data) / 1024 / 1024:.1f}MB)")
        return True
    except Exception as e:
        print(f"[ACE-Step] Failed to download LoRA: {e}")
        return False


# ---------------------------------------------------------------------------
# Load model (runs once at cold start)
# ---------------------------------------------------------------------------
print("[ACE-Step] Loading model...")
load_start = time.time()

from acestep.pipeline_ace_step import ACEStepPipeline

pipeline = ACEStepPipeline()

# Download and load LoRA
lora_available = download_lora()
if lora_available:
    try:
        print(f"[ACE-Step] Loading LoRA from {LORA_DIR} (weight={LORA_SCALE})")
        pipeline.load_lora(LORA_DIR, lora_weight=LORA_SCALE)
        print("[ACE-Step] LoRA loaded successfully")
    except Exception as e:
        print(f"[ACE-Step] LoRA loading failed, continuing with base model: {e}")
        lora_available = False
else:
    print("[ACE-Step] Running base model (no LoRA)")

print(f"[ACE-Step] Model loaded in {time.time() - load_start:.1f}s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def convert_wav_to_mp3(wav_path: str) -> str:
    """Convert WAV to MP3 using ffmpeg."""
    mp3_path = wav_path.replace(".wav", ".mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-b:a", "320k", "-q:a", "0", mp3_path],
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
    output_dir = tempfile.mkdtemp(prefix="ace_")

    for i in range(num_songs):
        gen_start = time.time()
        print(f"[ACE-Step] Generating song {i+1}/{num_songs}...")

        try:
            # ACE-Step pipeline returns list of file paths + metadata dict
            results = pipeline(
                format="wav",
                audio_duration=float(duration),
                prompt=caption,
                lyrics=lyrics,
                infer_step=60,
                guidance_scale=15.0,
                scheduler_type="euler",
                cfg_type="apg",
                manual_seeds=[int(time.time() * 1000 + i) % (2**31)],
                save_path=output_dir,
            )

            # Results = [path1, path2, ..., metadata_dict]
            # We generate one song at a time, so expect [path, metadata]
            audio_path = None
            for r in results:
                if isinstance(r, str) and os.path.isfile(r):
                    audio_path = r
                    break

            if not audio_path:
                raise RuntimeError("No audio file generated")

            # Convert WAV to MP3
            mp3_path = convert_wav_to_mp3(audio_path)
            os.unlink(audio_path)

            # Get actual duration
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", mp3_path],
                capture_output=True, text=True,
            )
            actual_duration = float(probe.stdout.strip()) if probe.stdout.strip() else duration

            # Upload to Supabase
            filename = f"{job_id}_{i+1}.mp3"
            url = upload_to_supabase(mp3_path, filename)
            os.unlink(mp3_path)

            songs.append({
                "url": url,
                "duration": round(actual_duration, 1),
                "title": f"ACE Song {i+1}",
            })

            gen_time = time.time() - gen_start
            print(f"[ACE-Step] Song {i+1} done in {gen_time:.1f}s → {url}")

        except Exception as e:
            print(f"[ACE-Step] Error generating song {i+1}: {e}")
            songs.append({
                "url": None,
                "duration": 0,
                "title": f"ACE Song {i+1} (failed)",
                "error": str(e),
            })

    return {"songs": songs}


runpod.serverless.start({"handler": handler})
