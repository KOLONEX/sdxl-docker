"""Bake SDXL base + refiner + VAE at build time into their MODEL dirs.

Uses hf_transfer (enabled via HF_HUB_ENABLE_HF_TRANSFER=1 in the Dockerfile) for
robust, parallel, resumable downloads, and restricts the base/refiner repos to the
fp16 weights + configs — the pipelines load with variant="fp16", so the full-precision
weights are dead weight and roughly double the download. The VAE repo (sdxl-vae-fp16-fix)
has no fp16-suffixed file, so it is fetched in full (it is small).
"""
import os
import sys
import time

from huggingface_hub import snapshot_download

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

MAX_ATTEMPTS = 6


def fetch(repo: str, dst: str, patterns) -> None:
    """Download a repo with retries. snapshot_download resumes partial files, so each
    retry picks up where the last left off."""
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
        print(f"Downloading {repo} -> {dst}", flush=True)
        fetch(repo, dst, patterns)
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
