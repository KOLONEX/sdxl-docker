import asyncio
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol

from app.schemas import Img2ImgParams, Txt2ImgParams
from app.storage import Storage


class Backend(Protocol):
    def load(self) -> None: ...
    def txt2img(self, params: Txt2ImgParams, seeds: List[int]) -> list: ...
    def img2img(self, params: Img2ImgParams, init_image, seeds: List[int]) -> list: ...


@dataclass
class GenerationResult:
    job_ids: List[str]
    paths: List[Path]
    seeds: List[int]
    duration_ms: int
    count: int


class PipelineManager:
    def __init__(self, backend: Backend, storage: Storage):
        self._backend = backend
        self._storage = storage
        self._lock = asyncio.Lock()
        self._busy = False
        self._loaded = False

    async def load(self) -> None:
        await asyncio.to_thread(self._backend.load)
        self._loaded = True

    @property
    def model_loaded(self) -> bool:
        return self._loaded

    @property
    def busy(self) -> bool:
        return self._busy

    def _seeds(self, params) -> List[int]:
        base = params.seed or random.randint(1, 2_147_483_647)
        return [base + i for i in range(params.batch)]

    async def _run(self, fn, params, *args) -> GenerationResult:
        async with self._lock:
            self._busy = True
            try:
                start = time.monotonic()
                seeds = self._seeds(params)
                images = await asyncio.to_thread(fn, params, *args, seeds)
                job_ids, paths = [], []
                for img in images:
                    jid = self._storage.new_job_id()
                    paths.append(self._storage.save_png(jid, img))
                    job_ids.append(jid)
                duration_ms = int((time.monotonic() - start) * 1000)
                return GenerationResult(
                    job_ids=job_ids, paths=paths, seeds=seeds,
                    duration_ms=duration_ms, count=len(images))
            finally:
                self._busy = False

    async def generate_txt2img(self, params: Txt2ImgParams) -> GenerationResult:
        return await self._run(self._backend.txt2img, params)

    async def generate_img2img(self, params: Img2ImgParams, init_image) -> GenerationResult:
        return await self._run(self._backend.img2img, params, init_image)


class FakeBackend:
    """In-memory backend for host tests (no GPU/torch)."""

    def __init__(self):
        self.loaded = False
        self.last_seeds = None
        self.last_params = None
        self._in_flight = 0
        self.max_in_flight = 0

    def load(self) -> None:
        self.loaded = True

    def _make(self, params, seeds):
        from PIL import Image
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            time.sleep(0.01)
            self.last_seeds = seeds
            self.last_params = params
            return [Image.new("RGB", (params.width, params.height), "gray")
                    for _ in seeds]
        finally:
            self._in_flight -= 1

    def txt2img(self, params, seeds):
        return self._make(params, seeds)

    def img2img(self, params, init_image, seeds):
        return self._make(params, seeds)


class SdxlBackend:
    """Real Stable Diffusion XL backend. Requires GPU. All torch/diffusers
    imports are lazy so this module still imports on a plain host."""

    _SCHEDULERS = {
        "euler": "EulerDiscreteScheduler",
        "euler_a": "EulerAncestralDiscreteScheduler",
        "dpmpp_2m": "DPMSolverMultistepScheduler",
        "dpmpp_sde": "DPMSolverSDEScheduler",
        "ddim": "DDIMScheduler",
    }

    def __init__(self, model_dir: str, refiner_dir: str, vae_dir: str,
                 lora_dir: str, default_use_refiner: bool = True):
        self.model_dir = model_dir
        self.refiner_dir = refiner_dir
        self.vae_dir = vae_dir
        self.lora_dir = lora_dir
        self.default_use_refiner = default_use_refiner
        self._base = None
        self._refiner = None

    def load(self) -> None:
        import torch
        from diffusers import (AutoencoderKL, StableDiffusionXLImg2ImgPipeline,
                               StableDiffusionXLPipeline)
        vae = AutoencoderKL.from_pretrained(self.vae_dir, torch_dtype=torch.float16)
        self._base = StableDiffusionXLPipeline.from_pretrained(
            self.model_dir, vae=vae, torch_dtype=torch.float16,
            variant="fp16", use_safetensors=True).to("cuda")
        self._refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.refiner_dir, vae=vae, text_encoder_2=self._base.text_encoder_2,
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
        try:
            self._base.enable_xformers_memory_efficient_attention()
            self._refiner.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    def _apply_scheduler(self, pipe, sampler: str):
        import diffusers
        cls = getattr(diffusers, self._SCHEDULERS[sampler])
        pipe.scheduler = cls.from_config(pipe.scheduler.config)

    def _apply_loras(self, params):
        from pathlib import Path
        if not params.loras:
            return
        names, weights = [], []
        for i, spec in enumerate(params.loras):
            path = Path(self.lora_dir) / f"{spec.name}.safetensors"
            adapter = f"a{i}"
            self._base.load_lora_weights(str(path.parent), weight_name=path.name,
                                         adapter_name=adapter)
            names.append(adapter); weights.append(spec.weight)
        self._base.set_adapters(names, adapter_weights=weights)

    def _reset_loras(self):
        try:
            self._base.unload_lora_weights()
        except Exception:
            pass

    def _generators(self, seeds):
        import torch
        return [torch.Generator(device="cuda").manual_seed(int(s)) for s in seeds]

    def _postprocess(self, images, params):
        if params.remove_background:
            import rembg
            images = [rembg.remove(im) for im in images]
        return images

    def txt2img(self, params, seeds):
        self._apply_scheduler(self._base, params.sampler)
        self._apply_loras(params)
        try:
            gens = self._generators(seeds)
            use_ref = params.use_refiner
            common = dict(prompt=[params.prompt] * len(seeds),
                          negative_prompt=[params.negative_prompt] * len(seeds),
                          num_inference_steps=params.steps, guidance_scale=params.cfg,
                          width=params.width, height=params.height, generator=gens)
            if use_ref:
                latents = self._base(**common, denoising_end=params.refiner_switch,
                                     output_type="latent").images
                self._apply_scheduler(self._refiner, params.sampler)
                images = self._refiner(
                    prompt=[params.prompt] * len(seeds),
                    negative_prompt=[params.negative_prompt] * len(seeds),
                    num_inference_steps=params.steps, guidance_scale=params.cfg,
                    denoising_start=params.refiner_switch, image=latents,
                    generator=gens).images
            else:
                images = self._base(**common).images
            return self._postprocess(images, params)
        finally:
            self._reset_loras()

    def img2img(self, params, init_image, seeds):
        from diffusers import StableDiffusionXLImg2ImgPipeline
        # Reuse the base weights as an img2img pipeline (shares components, no reload).
        img_pipe = StableDiffusionXLImg2ImgPipeline(**self._base.components)
        self._apply_scheduler(img_pipe, params.sampler)
        self._apply_loras(params)
        try:
            gens = self._generators(seeds)
            images = img_pipe(
                prompt=[params.prompt] * len(seeds),
                negative_prompt=[params.negative_prompt] * len(seeds),
                image=init_image.convert("RGB"), strength=params.denoise,
                num_inference_steps=params.steps, guidance_scale=params.cfg,
                generator=gens).images
            return self._postprocess(images, params)
        finally:
            self._reset_loras()
