<div align="center">

# SDXL API — Generate images from text or image on GPU

**Text or image in → PNG(s) out.** A self-contained, GPU-powered Docker service that
runs SDXL base + refiner + VAE offline, built to feed the asset pipeline of
[**KOLONEX**](https://kolonex.net) — a real-time 4X space-empire strategy game.

[![Play KOLONEX](https://img.shields.io/badge/🎮%20Play-KOLONEX.NET-6c3ce9)](https://kolonex.net)

*English · [Español ↓](#-español)*

</div>

---

## Why this exists

[**KOLONEX**](https://kolonex.net) needs a constant stream of concept-art variants — ship skins,
planetary structures, UI panels. SDXL's quality and speed make it the right tool for that job.
This service wraps the **base + refiner ensemble** behind a clean HTTP API, with LoRA support for
on-demand style adaptation, packaged as a **single Docker image that boots 100% offline**.

---

## Highlights

- **Text → PNG** and **Image → PNG** in one request, with the base + refiner ensemble.
- **Fully self-contained & offline.** Base, refiner, and VAE weights are **baked into the image**. No internet needed at runtime.
- **LoRA drop-in.** Mount a host folder to `/models/loras`; any `.safetensors` file placed there is auto-detected and selectable per request.
- **GPU-selectable.** Pin to a specific GPU with `--gpus '"device=1"'` or `CUDA_VISIBLE_DEVICES`.
- **Testable without a GPU.** The full API is covered by tests that run on any machine via `FakeBackend`.
- **Hardened.** Path-traversal-safe job IDs, upload-size limits, blocking mutex so the GPU handles one job at a time.

---

## Quick start

```bash
# build (downloads + bakes base + refiner + VAE: ~13 GB models, ~18-22 GB image)
docker build -t sdxl-api ./sdxl-docker

# run on GPU 1, LoRAs from a host folder, outputs persisted
docker run --gpus '"device=1"' -p 5082:8000 \
  -v "$PWD/loras:/models/loras" -v sdxl-outputs:/outputs sdxl-api
# → http://localhost:5082
```

Or with Docker Compose (GPU 1 pre-configured):

```bash
cd sdxl-docker
docker compose up
# → http://localhost:5082
```

**Requirements:** NVIDIA GPU with **≥16 GB VRAM**, CUDA 12.1-compatible drivers, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

---

## API

| Method | Path | Description |
|---|---|---|
| `GET`    | `/`                | bilingual web frontend |
| `GET`    | `/health`          | model status, GPU name, free VRAM |
| `GET`    | `/loras`           | list available LoRAs |
| `POST`   | `/txt2img`         | JSON body → `{images:[{job_id, url, seed}], duration_ms, count}` |
| `POST`   | `/img2img`         | multipart `file` + form fields → same |
| `POST`   | `/img2img/json`    | JSON body with `image_base64` → same |
| `GET`    | `/files/{id}.png`  | download a generated image |
| `DELETE` | `/files/{id}.png`  | delete a generated image |

### Parameters (txt2img / img2img)

| Param | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | — | required, min 1 char |
| `negative_prompt` | string | `""` | |
| `steps` | int | `30` | 1–150 |
| `cfg` | float | `7.0` | 0–30 |
| `sampler` | string | `dpmpp_2m` | `euler`, `euler_a`, `dpmpp_2m`, `dpmpp_sde`, `ddim` |
| `width` | int | `1024` | 512–2048, multiple of 8 (txt2img only) |
| `height` | int | `1024` | 512–2048, multiple of 8 (txt2img only) |
| `seed` | int | `0` | 0 = random |
| `batch` | int | `1` | 1–8; requests above `MAX_BATCH` are rejected with HTTP 400 |
| `loras` | list | `[]` | `[{"name": "my-lora", "weight": 0.8}]` |
| `vae` | string | `fp16-fix` | VAE selection (txt2img only; not applied on img2img in v1) |
| `use_refiner` | bool | `true` | run the refiner pass (txt2img only; not applied on img2img in v1) |
| `refiner_switch` | float | `0.8` | base/refiner handoff (0–1) |
| `remove_background` | bool | `false` | rembg post-process |
| `inline` | bool | `false` | return PNG bytes directly (batch=1 only) |
| `denoise` | float | `0.6` | img2img denoising strength (0–1) |

---

## LoRA drop-in workflow

1. Mount a host directory as `/models/loras`:
   ```bash
   docker run --gpus '"device=1"' -p 5082:8000 \
     -v /path/to/my-loras:/models/loras -v sdxl-outputs:/outputs sdxl-api
   ```
2. Drop any `.safetensors` file into that folder — no restart needed; the registry rescans on each request.
3. Check available LoRAs:
   ```bash
   curl http://localhost:5082/loras
   # {"loras":[{"name":"my-style","filename":"my-style.safetensors"}]}
   ```
4. Use it in a generation:
   ```bash
   curl -s -X POST http://localhost:5082/txt2img \
     -H 'Content-Type: application/json' \
     -d '{"prompt":"a sci-fi cargo ship, concept art","loras":[{"name":"my-style","weight":0.8}]}'
   ```

---

## How the image is built

The `Dockerfile` uses `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` as the base. The pipeline:

1. **System deps** — Python 3.10, GL libs.
2. **Torch first** — `torch==2.4.0 + torchvision==0.19.0` from the **cu121** index.
3. **App runtime deps** — `diffusers`, `transformers`, `fastapi`, `uvicorn`, `rembg`, etc.
4. **Bake models** — `scripts/download_models.py` downloads base + refiner + VAE via `huggingface_hub` into `/opt/models/`. The container needs no internet at runtime.
5. **App code** — copied into `/app`.

```bash
docker build -t sdxl-api ./sdxl-docker
```

---

## Architecture

SDXL is isolated behind a `Backend` protocol (`load()` / `txt2img()` / `img2img()`). A `PipelineManager`
wraps any backend with an `asyncio.Lock` (one GPU job at a time), timing, and persistence. The API
(`create_app` factory) and `storage.py` **know nothing about torch or CUDA**, so they are tested
anywhere with a `FakeBackend`. Only `SdxlBackend` and `docker build` need a GPU.

```
app/
├── schemas.py    # Pydantic v2 request/response validation
├── storage.py    # the only place that touches .png paths (traversal-safe job_id)
├── loras.py      # LoraRegistry — scans /models/loras, resolves by name
├── pipeline.py   # Backend protocol · PipelineManager (mutex) · FakeBackend · SdxlBackend
├── api.py        # create_app factory + endpoints
└── web/          # bilingual ES/EN frontend (no build step)
```

**Dev tests (no GPU, no torch):**

```bash
cd sdxl-docker
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements-dev.txt
pytest
```

---

## Configuration

| Env var | Default | Description |
|---|---|---|
| `OUTPUT_DIR`          | `/outputs`              | where PNG files are stored |
| `LORA_DIR`            | `/models/loras`         | LoRA `.safetensors` scan directory |
| `MODEL_DIR`           | `/opt/models/sdxl-base` | SDXL base model path |
| `REFINER_DIR`         | `/opt/models/sdxl-refiner` | SDXL refiner model path |
| `VAE_DIR`             | `/opt/models/sdxl-vae`  | VAE (fp16-fix) path |
| `MAX_UPLOAD_MB`       | `10`                    | max input image size (MB) |
| `MAX_BATCH`           | `4`                     | max images per request |
| `DEFAULT_USE_REFINER` | `true`                  | whether the refiner runs by default |

---

## License

This wrapper — API and Docker packaging — is part of the [KOLONEX](https://kolonex.net) toolchain,
released under the MIT License. SDXL base and refiner weights are subject to the
[CreativeML Open RAIL++-M License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md).

<div align="center">

### [Play KOLONEX — free, in early access →](https://kolonex.net)

</div>

---
---

<a name="-español"></a>

<div align="center">

# SDXL API — Generá imágenes desde texto o imagen en GPU

**Texto o imagen entran → PNG(s) salen.** Un servicio Docker autocontenido y acelerado por GPU que
corre SDXL base + refiner + VAE offline, creado para alimentar el pipeline de assets de
[**KOLONEX**](https://kolonex.net) — un juego de estrategia espacial 4X en tiempo real.

*[English ↑](#sdxl-api--generate-images-from-text-or-image-on-gpu) · Español*

</div>

## Por qué existe

[**KOLONEX**](https://kolonex.net) necesita un flujo constante de variantes de arte conceptual —
skins de naves, estructuras planetarias, paneles de UI. La calidad y velocidad de SDXL lo hacen la
herramienta ideal. Este servicio envuelve el **ensemble base + refiner** detrás de una API HTTP
limpia, con soporte de LoRA para adaptación de estilo por request, empaquetado como una **imagen
Docker que arranca 100% offline**.

## Inicio rápido

```bash
# build (descarga + hornea base + refiner + VAE: ~13 GB de modelos, imagen ~18-22 GB)
docker build -t sdxl-api ./sdxl-docker

# correr en GPU 1, LoRAs desde una carpeta del host, outputs persistidos
docker run --gpus '"device=1"' -p 5082:8000 \
  -v "$PWD/loras:/models/loras" -v sdxl-outputs:/outputs sdxl-api
# → http://localhost:5082
```

Con Docker Compose (GPU 1 preconfigurada):

```bash
cd sdxl-docker
docker compose up
# → http://localhost:5082
```

## LoRAs — flujo de trabajo

1. Montá una carpeta del host como `/models/loras`.
2. Copiá cualquier archivo `.safetensors` a esa carpeta — sin reiniciar; el registry rescannea en cada request.
3. Verificá los LoRAs disponibles: `curl http://localhost:5082/loras`
4. Usalo en una generación: `{"prompt":"...", "loras":[{"name":"mi-estilo","weight":0.8}]}`

## Arquitectura

SDXL queda aislado detrás de un `Backend` protocol. Un `PipelineManager` envuelve cualquier backend
con un `asyncio.Lock` (un job de GPU a la vez), timing y persistencia. La API y `storage.py` **no
conocen torch ni CUDA**, por eso se testean en cualquier máquina con `FakeBackend`.

## Licencia

Este wrapper es parte del toolchain de [KOLONEX](https://kolonex.net), publicado bajo MIT.
Los pesos de SDXL base y refiner están sujetos a la
[CreativeML Open RAIL++-M License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md).

<div align="center">

### [Jugá KOLONEX — gratis, en acceso anticipado →](https://kolonex.net)

</div>
