"""
RunPod Serverless Handler for ACE-Step v1.5 Music Generation

Uses ACE-Step v1.5 programmatic API (AceStepHandler + generate_music).
Loads model + Turkish LoRA at cold start.
LoRA is downloaded from Supabase Storage (ace-lora bucket) at startup.
Generates audio, uploads to Supabase Storage, returns URLs.

Env vars (set in RunPod Template):
  SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
  LORA_FILENAME (default: adapter_model.safetensors)
  LORA_SCALE (default: 0.45)
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
LORA_SCALE = float(os.environ.get("LORA_SCALE", "0.45"))
ACE_MODEL_CONFIG = os.environ.get("ACE_MODEL_CONFIG", "acestep-v15-turbo")
ACE_LM_MODEL = os.environ.get("ACE_LM_MODEL", "acestep-5Hz-lm-4B")  # 4B (best CoT quality)
LORA_DIR = "/tmp/lora"
LORA_LOCAL_PATH = os.path.join(LORA_DIR, LORA_FILENAME)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

print(f"[ACE-Step] Config: SUPABASE_URL={SUPABASE_URL[:30]}..., LORA={LORA_FILENAME}, SCALE={LORA_SCALE}, MODEL_CONFIG={ACE_MODEL_CONFIG}, LM_MODEL={ACE_LM_MODEL}")
BUCKET_NAME = "mastered-songs"
LORA_BUCKET = "ace-lora"

# ACE-Step v1.5 repo is cloned to /app/ace-step-repo by Dockerfile
PROJECT_ROOT = "/app/ace-step-repo"
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------------------------------------------------------
# Download LoRA from Supabase at cold start
# ---------------------------------------------------------------------------
def download_lora():
    """Download LoRA weights + config from Supabase ace-lora bucket.

    Uses a version.txt file in the bucket to detect LoRA updates.
    If the remote version differs from the cached version, re-downloads everything.
    """
    os.makedirs(LORA_DIR, exist_ok=True)

    config_path = os.path.join(LORA_DIR, "adapter_config.json")
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
    config_path=ACE_MODEL_CONFIG,  # Set via ACE_MODEL_CONFIG env (default: acestep-v15-turbo)
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

# Step 1b: Initialize LLM handler (configurable via ACE_LM_MODEL env, default 1.7B)
llm_handler = LLMHandler()
llm_status, llm_success = llm_handler.initialize(
    checkpoint_dir=CHECKPOINTS_DIR,
    lm_model_path=ACE_LM_MODEL,
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
# Quality Check — reject broken songs (cold-start artifacts, noise)
# ---------------------------------------------------------------------------
MAX_DURATION_HARD_CAP = 420  # 7 min absolute maximum
MAX_DURATION_AUTO = 360      # 6 min for auto-duration mode


def check_song_quality(actual_duration, requested_duration):
    """Check if generated song duration is within acceptable bounds.

    Returns (is_valid, reason).
    """
    if actual_duration > MAX_DURATION_HARD_CAP:
        return False, f"Duration {actual_duration:.1f}s exceeds hard cap {MAX_DURATION_HARD_CAP}s"
    if requested_duration > 0:
        max_allowed = min(requested_duration * 1.5, MAX_DURATION_AUTO)
        if actual_duration > max_allowed:
            return False, f"Duration {actual_duration:.1f}s exceeds limit {max_allowed:.0f}s (requested {requested_duration}s)"
    elif actual_duration > MAX_DURATION_AUTO:
        return False, f"Duration {actual_duration:.1f}s exceeds auto max {MAX_DURATION_AUTO}s"
    return True, ""


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

    # Cover mode parameters (verified from Gradio testing)
    task_type = input_data.get("task_type", "text2music")
    cover_audio_url = input_data.get("cover_audio_url")
    audio_cover_strength = float(input_data.get("audio_cover_strength", 0.06))
    cover_noise_strength = float(input_data.get("cover_noise_strength", 0.1))

    # Download cover source audio from Supabase if cover mode
    # IMPORTANT: Cover/Remix uses `src_audio` (NOT `reference_audio`!)
    # reference_audio = style reference for Custom mode only
    # src_audio = the song to cover/remix
    src_audio_path = None
    if task_type == "cover" and cover_audio_url:
        import requests as req
        print(f"[ACE-Step] Downloading cover source audio from {cover_audio_url}...")
        resp = req.get(cover_audio_url)
        print(f"[ACE-Step] Cover source audio downloaded: {len(resp.content)} bytes (HTTP {resp.status_code})")

        if resp.status_code != 200 or len(resp.content) < 1000:
            raise RuntimeError(
                f"Cover audio download failed: HTTP {resp.status_code}, {len(resp.content)} bytes. "
                f"URL: {cover_audio_url} | Body: {resp.content[:200]}"
            )

        tmp_dl = f"/tmp/cover_{uuid.uuid4().hex[:8]}_dl.mp3"
        with open(tmp_dl, "wb") as f:
            f.write(resp.content)

        # Convert to WAV — torchaudio.load() may fail on certain MP3 encodings
        src_audio_path = tmp_dl.replace("_dl.mp3", ".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_dl, "-ar", "48000", "-ac", "2", src_audio_path],
            capture_output=True, check=True,
        )
        os.unlink(tmp_dl)
        print(f"[ACE-Step] Cover source audio converted to WAV for torchaudio")

    # -----------------------------------------------------------------------
    # DIAGNOSTIC: Log raw input keys to see exactly what frontend sends
    # -----------------------------------------------------------------------
    input_keys = sorted(input_data.keys())
    print(f"[ACE-Step] Raw input keys: {input_keys}")

    # Core generation parameters (always read from request)
    guidance_scale = float(input_data.get("guidance_scale", 3.0))
    lm_cfg_scale = float(input_data.get("lm_cfg_scale", 2.0))
    shift_val = float(input_data.get("shift", 3.0))
    inference_steps = int(input_data.get("inference_steps", 8))

    # Extended parameters — ONLY applied when use_all_params=true
    # Default: use dataclass defaults (same as old working handler)
    use_all_params = bool(input_data.get("use_all_params", False))

    if use_all_params:
        lm_temperature = float(input_data.get("lm_temperature", 0.85))
        latent_shift = float(input_data.get("latent_shift", 0.0))
        latent_rescale = float(input_data.get("latent_rescale", 1.0))
        use_adg = bool(input_data.get("use_adg", False))
        cfg_interval_start = float(input_data.get("cfg_interval_start", 0.0))
        cfg_interval_end = float(input_data.get("cfg_interval_end", 1.0))
        infer_method = input_data.get("infer_method", "ode")
        custom_timesteps_str = input_data.get("timesteps", "")
        parsed_timesteps = None
        if custom_timesteps_str and str(custom_timesteps_str).strip():
            try:
                parsed_timesteps = [float(t.strip()) for t in str(custom_timesteps_str).split(",") if t.strip()]
                if parsed_timesteps:
                    inference_steps = len(parsed_timesteps) - 1
            except (ValueError, TypeError):
                parsed_timesteps = None
        lm_top_k = int(input_data.get("lm_top_k", 0))
        lm_top_p = float(input_data.get("lm_top_p", 0.9))
        lm_negative_prompt = input_data.get("lm_negative_prompt", "NO USER INPUT") or "NO USER INPUT"
        thinking = bool(input_data.get("thinking", True))
        use_cot_metas = bool(input_data.get("use_cot_metas", True))
        use_cot_caption = bool(input_data.get("use_cot_caption", True))
        use_cot_language = bool(input_data.get("use_cot_language", True))
        enable_normalization = bool(input_data.get("enable_normalization", True))
        normalization_db = float(input_data.get("normalization_db", -1.0))
        bpm_raw = input_data.get("bpm")
        bpm = int(bpm_raw) if bpm_raw is not None and str(bpm_raw).strip() else None
        keyscale = input_data.get("keyscale", "")
        timesignature = input_data.get("timesignature", "")
        vocal_language = input_data.get("vocal_language", "unknown")
        instrumental = bool(input_data.get("instrumental", False))
        print(f"[ACE-Step] MODE: use_all_params=True — ALL params from request")
    else:
        # MINIMAL MODE: use dataclass defaults for everything except core 4 params
        # This matches the old working handler behavior
        lm_temperature = 0.85
        latent_shift = 0.0
        latent_rescale = 1.0
        use_adg = False
        cfg_interval_start = 0.0
        cfg_interval_end = 1.0
        infer_method = "ode"
        parsed_timesteps = None
        lm_top_k = 0
        lm_top_p = 0.9
        lm_negative_prompt = "NO USER INPUT"
        thinking = True
        use_cot_metas = True
        use_cot_caption = True
        use_cot_language = True
        enable_normalization = True
        normalization_db = -1.0
        bpm = None
        keyscale = ""
        timesignature = ""  # IMPORTANT: empty = CoT decides (dataclass default)
        vocal_language = "unknown"
        instrumental = False
        print(f"[ACE-Step] MODE: MINIMAL — only core 4 params from request, rest = dataclass defaults")

    print(f"[ACE-Step] Params: cfg={guidance_scale}, lm={lm_cfg_scale}, shift={shift_val}, steps={inference_steps}"
          f", lm_temp={lm_temperature}, lat_shift={latent_shift}, lat_rescale={latent_rescale}"
          f", infer={infer_method}, adg={use_adg}, thinking={thinking}"
          f", timesteps={'custom' if parsed_timesteps else 'auto'}"
          f", norm={enable_normalization}, norm_db={normalization_db}"
          f", bpm={bpm}, key={keyscale}, timesig={timesignature}, lang={vocal_language}"
          f", instrumental={instrumental}")

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

            if use_all_params:
                # FULL MODE: set ALL params explicitly
                params = GenerationParams(
                    task_type=task_type,
                    caption=caption,
                    lyrics=lyrics,
                    instrumental=instrumental,
                    vocal_language=vocal_language,
                    bpm=bpm,
                    keyscale=keyscale,
                    timesignature=timesignature,
                    duration=float(duration),
                    seed=seed,
                    inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    shift=shift_val,
                    infer_method=infer_method,
                    timesteps=parsed_timesteps,
                    use_adg=use_adg,
                    cfg_interval_start=cfg_interval_start,
                    cfg_interval_end=cfg_interval_end,
                    lm_cfg_scale=lm_cfg_scale,
                    lm_temperature=lm_temperature,
                    lm_top_k=lm_top_k,
                    lm_top_p=lm_top_p,
                    lm_negative_prompt=lm_negative_prompt,
                    thinking=thinking,
                    use_cot_metas=use_cot_metas,
                    use_cot_caption=use_cot_caption,
                    use_cot_language=use_cot_language,
                    use_constrained_decoding=True,
                    enable_normalization=enable_normalization,
                    normalization_db=normalization_db,
                    latent_shift=latent_shift,
                    latent_rescale=latent_rescale,
                    **({"src_audio": src_audio_path,
                        "audio_cover_strength": audio_cover_strength,
                        "cover_noise_strength": cover_noise_strength} if src_audio_path else {}),
                )
            else:
                # MINIMAL MODE: only set essential params, rest = dataclass defaults
                # This matches the old working handler (pre parameter-additions)
                params = GenerationParams(
                    task_type=task_type,
                    caption=caption,
                    lyrics=lyrics,
                    duration=float(duration),
                    inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    lm_cfg_scale=lm_cfg_scale,
                    seed=seed,
                    shift=shift_val,
                    infer_method="ode",
                    **({"src_audio": src_audio_path,
                        "audio_cover_strength": audio_cover_strength,
                        "cover_noise_strength": cover_noise_strength} if src_audio_path else {}),
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

            # Quality check — reject broken songs before uploading
            is_valid, reject_reason = check_song_quality(actual_duration, duration)
            if not is_valid:
                print(f"[ACE-Step] Song {i+1} REJECTED: {reject_reason}")
                os.unlink(mp3_path)
                songs.append({
                    "url": None,
                    "duration": round(actual_duration, 1),
                    "rejected": True,
                    "error": reject_reason,
                })
                gen_time = time.time() - gen_start
                print(f"[ACE-Step] Song {i+1} rejected in {gen_time:.1f}s")
                continue

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

    # Cleanup temp cover source audio file
    if src_audio_path and os.path.exists(src_audio_path):
        os.unlink(src_audio_path)
        print("[ACE-Step] Cleaned up temp cover source audio file")

    valid_count = sum(1 for s in songs if s.get("url"))
    rejected_count = sum(1 for s in songs if s.get("rejected"))
    print(f"[ACE-Step] Done: {valid_count} valid, {rejected_count} rejected out of {len(songs)}")

    return {
        "songs": songs,
        "valid_count": valid_count,
        "rejected_count": rejected_count,
        # Diagnostic info — visible in RunPod response
        "diag": {
            "llm_initialized": llm_handler.llm_initialized if llm_handler else False,
            "llm_backend": getattr(llm_handler, 'llm_backend', None),
            "lora_loaded": dit_handler.lora_loaded,
            "use_lora": dit_handler.use_lora,
            "lora_scale": dit_handler.lora_scale,
            "model_config": ACE_MODEL_CONFIG,
            "lm_model": ACE_LM_MODEL,
            "use_all_params": use_all_params,
            "input_keys": input_keys,
        },
    }


runpod.serverless.start({"handler": handler})
