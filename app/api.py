import base64
import binascii
import io
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image, UnidentifiedImageError

from app.loras import LoraRegistry
from app.pipeline import PipelineManager
from app.schemas import (GeneratedImage, GenerateResponse, HealthResponse,
                         Img2ImgJsonRequest, Img2ImgParams,
                         LorasResponse, Txt2ImgParams)
from app.storage import Storage


def _gpu_stats():
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None
        free, _ = torch.cuda.mem_get_info()
        return torch.cuda.get_device_name(0), int(free // (1024 * 1024))
    except Exception:
        return None, None


def create_app(manager: PipelineManager, storage: Storage, registry: LoraRegistry,
               web_dir: str, max_upload_mb: int = 10, max_batch: int = 4,
               lifespan=None) -> FastAPI:
    app = FastAPI(title="SDXL API", version="1.0", lifespan=lifespan)
    index_path = Path(web_dir) / "index.html"

    @app.get("/", response_class=HTMLResponse)
    async def root():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    @app.get("/health", response_model=HealthResponse)
    async def health():
        gpu, vram = _gpu_stats()
        return HealthResponse(
            status="ok" if manager.model_loaded else "loading",
            gpu=gpu, vram_free_mb=vram,
            model_loaded=manager.model_loaded, busy=manager.busy)

    @app.get("/loras", response_model=LorasResponse)
    async def loras():
        return LorasResponse(loras=registry.list())

    def _validate(params):
        if params.batch > max_batch:
            raise HTTPException(status_code=400, detail=f"batch exceeds max {max_batch}")
        for l in params.loras:
            try:
                registry.resolve(l.name)
            except KeyError:
                raise HTTPException(status_code=400, detail=f"unknown lora: {l.name}")
        if not manager.model_loaded:
            raise HTTPException(status_code=503, detail="model not loaded")

    def _respond(result, inline: bool):
        if inline and result.count == 1:
            return Response(content=result.paths[0].read_bytes(), media_type="image/png",
                            headers={"X-Job-Id": result.job_ids[0],
                                     "X-Seed": str(result.seeds[0])})
        images = [GeneratedImage(job_id=j, url=storage.url_for(j), seed=s)
                  for j, s in zip(result.job_ids, result.seeds)]
        return JSONResponse(GenerateResponse(
            images=images, duration_ms=result.duration_ms, count=result.count).model_dump())

    def _decode_image(data: bytes) -> Image.Image:
        if len(data) > max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="image too large")
        try:
            img = Image.open(io.BytesIO(data)); img.load()
        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(status_code=400, detail="invalid image")
        return img.convert("RGB")

    @app.post("/txt2img")
    async def txt2img(params: Txt2ImgParams):
        _validate(params)
        result = await manager.generate_txt2img(params)
        return _respond(result, params.inline)

    @app.post("/img2img")
    async def img2img(
        file: UploadFile = File(...),
        prompt: str = Form(...),
        negative_prompt: str = Form(""),
        steps: int = Form(30),
        cfg: float = Form(7.0),
        sampler: str = Form("dpmpp_2m"),
        seed: int = Form(0),
        batch: int = Form(1),
        vae: str = Form("fp16-fix"),
        use_refiner: bool = Form(True),
        refiner_switch: float = Form(0.8),
        remove_background: bool = Form(False),
        denoise: float = Form(0.6),
        inline: bool = Form(False),
    ):
        data = await file.read()
        image = _decode_image(data)
        try:
            params = Img2ImgParams(
                prompt=prompt, negative_prompt=negative_prompt, steps=steps, cfg=cfg,
                sampler=sampler, seed=seed, batch=batch, vae=vae, use_refiner=use_refiner,
                refiner_switch=refiner_switch, remove_background=remove_background,
                denoise=denoise, inline=inline)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        _validate(params)
        result = await manager.generate_img2img(params, image)
        return _respond(result, params.inline)

    @app.post("/img2img/json")
    async def img2img_json(req: Img2ImgJsonRequest):
        try:
            data = base64.b64decode(req.image_base64, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=400, detail="invalid base64")
        image = _decode_image(data)
        params = Img2ImgParams(**req.model_dump(exclude={"image_base64"}))
        _validate(params)
        result = await manager.generate_img2img(params, image)
        return _respond(result, params.inline)

    @app.get("/files/{job_id}.png")
    async def get_file(job_id: str):
        if not storage.is_valid_job_id(job_id):
            raise HTTPException(status_code=400, detail="invalid job id")
        path = storage.path_for(job_id)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return Response(content=path.read_bytes(), media_type="image/png")

    @app.delete("/files/{job_id}.png", status_code=204)
    async def delete_file(job_id: str):
        if not storage.is_valid_job_id(job_id):
            raise HTTPException(status_code=400, detail="invalid job id")
        if not storage.delete(job_id):
            raise HTTPException(status_code=404, detail="not found")
        return Response(status_code=204)

    return app


_OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/outputs")
_LORA_DIR = os.environ.get("LORA_DIR", "/models/loras")
_MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/models/sdxl-base")
_REFINER_DIR = os.environ.get("REFINER_DIR", "/opt/models/sdxl-refiner")
_VAE_DIR = os.environ.get("VAE_DIR", "/opt/models/sdxl-vae")
_MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))
_MAX_BATCH = int(os.environ.get("MAX_BATCH", "4"))
_DEFAULT_USE_REFINER = os.environ.get("DEFAULT_USE_REFINER", "true").lower() == "true"
_WEB_DIR = str(Path(__file__).resolve().parent / "web")


def _build_default_app() -> FastAPI:
    from contextlib import asynccontextmanager
    from app.pipeline import SdxlBackend

    storage = Storage(_OUTPUT_DIR)
    registry = LoraRegistry(_LORA_DIR)
    backend = SdxlBackend(_MODEL_DIR, _REFINER_DIR, _VAE_DIR, _LORA_DIR, _DEFAULT_USE_REFINER)
    manager = PipelineManager(backend, storage)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await manager.load()
        yield

    return create_app(manager, storage, registry, _WEB_DIR, _MAX_UPLOAD_MB, _MAX_BATCH,
                      lifespan=lifespan)


app = _build_default_app()
