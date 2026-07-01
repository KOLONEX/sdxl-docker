from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

ALLOWED_SAMPLERS = {"euler", "euler_a", "dpmpp_2m", "dpmpp_sde", "ddim"}


class LoraSpec(BaseModel):
    name: str
    weight: float = Field(default=1.0, ge=-2.0, le=2.0)


class Txt2ImgParams(BaseModel):
    prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    steps: int = Field(default=30, ge=1, le=150)
    cfg: float = Field(default=7.0, ge=0.0, le=30.0)
    sampler: str = "dpmpp_2m"
    width: int = 1024
    height: int = 1024
    seed: int = Field(default=0, ge=0)
    batch: int = Field(default=1, ge=1, le=8)
    loras: List[LoraSpec] = Field(default_factory=list)
    vae: str = "fp16-fix"
    use_refiner: bool = True
    refiner_switch: float = Field(default=0.8, ge=0.0, le=1.0)
    remove_background: bool = False
    inline: bool = False

    @field_validator("sampler", mode="after")
    @classmethod
    def _check_sampler(cls, v: str) -> str:
        if v not in ALLOWED_SAMPLERS:
            raise ValueError(f"sampler must be one of {sorted(ALLOWED_SAMPLERS)}")
        return v

    @field_validator("width", "height", mode="after")
    @classmethod
    def _check_dims(cls, v: int) -> int:
        if not (512 <= v <= 2048):
            raise ValueError("dimension must be in [512, 2048]")
        if v % 8 != 0:
            raise ValueError("dimension must be a multiple of 8")
        return v


class Img2ImgParams(Txt2ImgParams):
    denoise: float = Field(default=0.6, ge=0.0, le=1.0)


class Img2ImgJsonRequest(Img2ImgParams):
    image_base64: str


class GeneratedImage(BaseModel):
    job_id: str
    url: str
    seed: int


class GenerateResponse(BaseModel):
    images: List[GeneratedImage]
    duration_ms: int
    count: int


class LoraInfo(BaseModel):
    name: str
    filename: str


class LorasResponse(BaseModel):
    loras: List[LoraInfo]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    gpu: Optional[str] = None
    vram_free_mb: Optional[int] = None
    model_loaded: bool = False
    busy: bool = False
