import pytest
from pydantic import ValidationError
from app.schemas import Txt2ImgParams, Img2ImgParams, LoraSpec, ALLOWED_SAMPLERS


def test_defaults():
    p = Txt2ImgParams(prompt="a spaceship")
    assert p.steps == 30 and p.width == 1024 and p.batch == 1
    assert p.use_refiner is None and p.sampler in ALLOWED_SAMPLERS


def test_rejects_unknown_sampler():
    with pytest.raises(ValidationError):
        Txt2ImgParams(prompt="x", sampler="nonsense")


def test_rejects_non_multiple_of_8_dimensions():
    with pytest.raises(ValidationError):
        Txt2ImgParams(prompt="x", width=1001)


def test_rejects_empty_prompt():
    with pytest.raises(ValidationError):
        Txt2ImgParams(prompt="")


def test_lora_spec_weight_bounds():
    with pytest.raises(ValidationError):
        LoraSpec(name="x", weight=5.0)


def test_img2img_adds_denoise():
    p = Img2ImgParams(prompt="x", denoise=0.4)
    assert p.denoise == 0.4
