"""Bake SDXL base + refiner + VAE at build time into their MODEL dirs."""
import os
import sys
from huggingface_hub import snapshot_download

TARGETS = {
    "stabilityai/stable-diffusion-xl-base-1.0": os.environ.get("MODEL_DIR", "/opt/models/sdxl-base"),
    "stabilityai/stable-diffusion-xl-refiner-1.0": os.environ.get("REFINER_DIR", "/opt/models/sdxl-refiner"),
    "madebyollin/sdxl-vae-fp16-fix": os.environ.get("VAE_DIR", "/opt/models/sdxl-vae"),
}


def main() -> int:
    for repo, dst in TARGETS.items():
        print(f"Downloading {repo} -> {dst}", flush=True)
        snapshot_download(repo_id=repo, local_dir=dst)
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
