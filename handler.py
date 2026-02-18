"""
RunPod Serverless Handler for ACE-Step Music Generation

Loads ACE-Step model + Turkish LoRA at cold start.
LoRA is downloaded from Supabase Storage (ace-lora bucket) at startup.
Accepts generation requests, produces WAV/MP3, uploads to Supabase Storage.
Returns URLs to the generated audio files.

Env vars (set in RunPod Template):
  SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
  LORA_FILENAME (default: adapter_model.safetensors)
  LORA_SCALE (default: 0.2)
  OUTPUT_FORMAT (default: mp3)
"""

import os
import uuid
import time
import base64
import tempfile
import subprocess

import runpod
import torch
from supabase import create_client

# ---------------------------------------------------------------------------
# Global model loading (runs once at cold start)
# ---------------------------------------------------------------------------
LORA_FILENAME = os.environ.get("LORA_FILENAME", "adapter_model.safetensors")
LORA_SCALE = float(os.environ.get("LORA_SCALE", "0.2"))
OUTPUT_FORMAT = os.environ.get("OUTPUT_FORMAT", "mp3")
LORA_LOCAL_PATH = f"/tmp/lora/{LORA_FILENAME}"

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
BUCKET_NAME = "ace-songs"
LORA_BUCKET = "ace-lora"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Download LoRA from Supabase Storage at cold start
def download_lora():
    """Download LoRA weights from Supabase ace-lora bucket."""
    os.makedirs(os.path.dirname(LORA_LOCAL_PATH), exist_ok=True)
    if os.path.exists(LORA_LOCAL_PATH):
        print(f"[ACE-Step] LoRA already cached at {LORA_LOCAL_PATH}")
        return True
    try:
        print(f"[ACE-Step] Downloading LoRA '{LORA_FILENAME}' from Supabase...")
        dl_start = time.time()
        data = supabase.storage.from_(LORA_BUCKET).download(LORA_FILENAME)
        with open(LORA_LOCAL_PATH, "wb") as f:
            f.write(data)
        print(f"[ACE-Step] LoRA downloaded in {time.time() - dl_start:.1f}s ({len(data) / 1024 / 1024:.1f}MB)")
        return True
    except Exception as e:
        print(f"[ACE-Step] Failed to download LoRA: {e}")
        return False

print("[ACE-Step] Loading model...")
load_start = time.time()

from acestep.pipeline import ACEStepPipeline

pipeline = ACEStepPipeline()

# Download and load LoRA
lora_available = download_lora()
if lora_available:
    print(f"[ACE-Step] Loading LoRA from {LORA_LOCAL_PATH} (scale={LORA_SCALE})")
    pipeline.load_lora(LORA_LOCAL_PATH, lora_scale=LORA_SCALE)
else:
    print("[ACE-Step] Running base model (no LoRA)")

print(f"[ACE-Step] Model loaded in {time.time() - load_start:.1f}s")


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


def handler(job):
    """
    RunPod handler function.

    Input:
      - lyrics (str): Song lyrics
      - caption (str): Style/prompt description
      - duration (int): Target duration in seconds (default 120)
      - num_songs (int): Number of songs to generate (default 2)
      - lora_scale (float): Override LoRA scale (optional)
      - reference_audio_b64 (str): Base64-encoded reference audio (optional)

    Output:
      - songs: list of { url, duration, title }
    """
    input_data = job["input"]

    lyrics = input_data.get("lyrics", "")
    caption = input_data.get("caption", "")
    duration = input_data.get("duration", 120)
    num_songs = min(input_data.get("num_songs", 2), 4)
    lora_scale_override = input_data.get("lora_scale")
    reference_audio_b64 = input_data.get("reference_audio_b64")

    # Override LoRA scale if provided
    if lora_scale_override is not None and lora_scale_override != LORA_SCALE and lora_available:
        pipeline.load_lora(LORA_LOCAL_PATH, lora_scale=float(lora_scale_override))

    # Handle reference audio
    reference_audio_path = None
    if reference_audio_b64:
        try:
            audio_bytes = base64.b64decode(reference_audio_b64)
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp.write(audio_bytes)
            tmp.close()
            reference_audio_path = tmp.name
        except Exception as e:
            print(f"[ACE-Step] Failed to decode reference audio: {e}")

    songs = []
    job_id = job.get("id", str(uuid.uuid4())[:8])

    for i in range(num_songs):
        gen_start = time.time()
        print(f"[ACE-Step] Generating song {i+1}/{num_songs}...")

        try:
            # Generate audio
            gen_kwargs = {
                "lyrics": lyrics,
                "caption": caption,
                "duration": duration,
            }
            if reference_audio_path:
                gen_kwargs["reference_audio"] = reference_audio_path

            audio_output = pipeline.generate(**gen_kwargs)

            # Save WAV to temp file
            wav_path = tempfile.mktemp(suffix=".wav")
            pipeline.save_audio(audio_output, wav_path)

            # Convert to MP3 if requested
            if OUTPUT_FORMAT == "mp3":
                final_path = convert_wav_to_mp3(wav_path)
                os.unlink(wav_path)
            else:
                final_path = wav_path

            # Get actual duration from file
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", final_path],
                capture_output=True, text=True,
            )
            actual_duration = float(probe.stdout.strip()) if probe.stdout.strip() else duration

            # Upload to Supabase
            ext = "mp3" if OUTPUT_FORMAT == "mp3" else "wav"
            filename = f"{job_id}_{i+1}.{ext}"
            url = upload_to_supabase(final_path, filename)

            songs.append({
                "url": url,
                "duration": round(actual_duration, 1),
                "title": f"ACE Song {i+1}",
            })

            # Cleanup temp file
            os.unlink(final_path)

            gen_time = time.time() - gen_start
            print(f"[ACE-Step] Song {i+1} done in {gen_time:.1f}s")

        except Exception as e:
            print(f"[ACE-Step] Error generating song {i+1}: {e}")
            songs.append({
                "url": None,
                "duration": 0,
                "title": f"ACE Song {i+1} (failed)",
                "error": str(e),
            })

    # Cleanup reference audio temp file
    if reference_audio_path and os.path.exists(reference_audio_path):
        os.unlink(reference_audio_path)

    return {"songs": songs}


runpod.serverless.start({"handler": handler})
