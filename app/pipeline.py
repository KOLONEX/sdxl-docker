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
