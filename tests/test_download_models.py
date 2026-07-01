"""Tests for the offline local-cache path of scripts/download_models.py.

Only from_local is exercised — it must not import huggingface_hub (that stays lazy inside
fetch), so these run on a plain host with no ML deps installed."""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "download_models", Path(__file__).resolve().parent.parent / "scripts" / "download_models.py")
dm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dm)


def test_from_local_copies_when_weights_present(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    (cache / "sdxl-base").mkdir(parents=True)
    (cache / "sdxl-base" / "model.fp16.safetensors").write_bytes(b"weights")
    (cache / "sdxl-base" / "config.json").write_text("{}")
    monkeypatch.setattr(dm, "LOCAL_CACHE", str(cache))

    dst = tmp_path / "opt" / "sdxl-base"
    assert dm.from_local(str(dst)) is True
    assert (dst / "model.fp16.safetensors").read_bytes() == b"weights"
    assert (dst / "config.json").exists()


def test_from_local_false_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "LOCAL_CACHE", str(tmp_path / "cache"))  # does not exist
    assert dm.from_local(str(tmp_path / "opt" / "sdxl-base")) is False


def test_from_local_false_when_no_weights(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    (cache / "sdxl-vae").mkdir(parents=True)
    (cache / "sdxl-vae" / "config.json").write_text("{}")  # configs but no *.safetensors
    monkeypatch.setattr(dm, "LOCAL_CACHE", str(cache))
    assert dm.from_local(str(tmp_path / "opt" / "sdxl-vae")) is False


def test_target_keys_map_one_to_one():
    # The Dockerfile bakes each model in its own layer via `download_models.py <key>`;
    # each key must select exactly one target.
    for key in ("base", "refiner", "vae"):
        matched = [t for t in dm.TARGETS if Path(t[1]).name.endswith(key)]
        assert len(matched) == 1, (key, matched)
