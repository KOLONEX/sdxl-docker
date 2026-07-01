"""Bake SDXL base + refiner + VAE at build time into their MODEL dirs.

Two sources, in order of preference:

1. A local cache (LOCAL_CACHE, default /opt/models-cache) — during `docker build` the
   project's ./models folder is bind-mounted here. If a model is already present there,
   it is copied in with no network access. This is the offline path: run
   scripts/export-models.sh once to populate ./models from a built image, and every later
   build (even without internet or Docker layer cache) reuses those weights.

2. HuggingFace — if the local cache does not have the model, it is downloaded. Uses
   hf_transfer (HF_HUB_ENABLE_HF_TRANSFER=1 in the Dockerfile) for robust, parallel,
   resumable downloads, and restricts base/refiner to the fp16 weights + configs (the
   pipelines load variant="fp16"; the full-precision weights are dead weight).
"""
import os
import shutil
import sys
import time
from pathlib import Path

# fp16 weights + everything needed to load a diffusers pipeline (configs, tokenizers).
# Excludes the full-precision *.safetensors / *.bin weights we never load.
FP16_PATTERNS = [
    "**/*.fp16.safetensors",
    "**/*.json",
    "**/*.txt",
    "**/*.model",
    "*.json",
]

TARGETS = [
    ("stabilityai/stable-diffusion-xl-base-1.0",
     os.environ.get("MODEL_DIR", "/opt/models/sdxl-base"), FP16_PATTERNS),
    ("stabilityai/stable-diffusion-xl-refiner-1.0",
     os.environ.get("REFINER_DIR", "/opt/models/sdxl-refiner"), FP16_PATTERNS),
    # VAE has no fp16-suffixed file (it IS the fp16-fix); fetch it whole. It's ~320MB.
    ("madebyollin/sdxl-vae-fp16-fix",
     os.environ.get("VAE_DIR", "/opt/models/sdxl-vae"), None),
]

LOCAL_CACHE = os.environ.get("LOCAL_CACHE", "/opt/models-cache")
MAX_ATTEMPTS = 6


def from_local(dst: str) -> bool:
    """If the local cache holds this model, copy it into dst and return True.

    The cache sub-folder is matched by dst's basename (sdxl-base / sdxl-refiner /
    sdxl-vae), which is how scripts/export-models.sh lays them out. A folder only counts
    as a valid cache if it actually contains weights, so a stray empty dir falls through
    to the network download."""
    src = Path(LOCAL_CACHE) / Path(dst).name
    if not src.is_dir():
        return False
    if not any(src.rglob("*.safetensors")):
        return False
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return True


def fetch(repo: str, dst: str, patterns) -> None:
    """Download a repo from HuggingFace with retries. snapshot_download resumes partial
    files, so each retry picks up where the last left off."""
    from huggingface_hub import snapshot_download  # lazy: only needed for the network path
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            snapshot_download(
                repo_id=repo,
                local_dir=dst,
                allow_patterns=patterns,
                max_workers=8,
            )
            return
        except Exception as e:  # network hiccups, throttling, partial reads
            print(f"  attempt {attempt}/{MAX_ATTEMPTS} for {repo} failed: {e}", flush=True)
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(5 * attempt)


def main() -> int:
    for repo, dst, patterns in TARGETS:
        if from_local(dst):
            print(f"Using local cache for {repo} -> {dst}", flush=True)
            continue
        print(f"Downloading {repo} -> {dst}", flush=True)
        fetch(repo, dst, patterns)
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
