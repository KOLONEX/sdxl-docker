#!/usr/bin/env bash
# Save the SDXL weights baked into a built image to ./models/, so later builds can reuse
# them offline. download_models.py copies from ./models (bind-mounted at build time) when
# the weights are present there, instead of downloading from HuggingFace.
#
# Usage:  scripts/export-models.sh [IMAGE]      (IMAGE defaults to sdxl-api)
set -euo pipefail

IMAGE="${1:-sdxl-api}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/models"

echo "Exporting SDXL weights from image '$IMAGE' -> $DEST"
mkdir -p "$DEST"

cid="$(docker create "$IMAGE")"
trap 'docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT

# /opt/models holds sdxl-base/ sdxl-refiner/ sdxl-vae/ — copy them all.
docker cp "$cid:/opt/models/." "$DEST/"

echo "Done. Local offline backup:"
du -sh "$DEST"/*/ 2>/dev/null || ls -la "$DEST"
