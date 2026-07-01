import asyncio
from pathlib import Path
from fastapi.testclient import TestClient
from app.api import create_app
from app.pipeline import PipelineManager, FakeBackend
from app.storage import Storage
from app.loras import LoraRegistry

WEB = str(Path(__file__).resolve().parent.parent / "app" / "web")


def _client(tmp_path, loaded=True):
    loras = tmp_path / "loras"; loras.mkdir()
    (loras / "scifi.safetensors").write_bytes(b"x")
    storage = Storage(str(tmp_path / "out"))
    manager = PipelineManager(FakeBackend(), storage)
    if loaded:
        asyncio.run(manager.load())
    app = create_app(manager, storage, LoraRegistry(str(loras)), WEB, max_batch=4)
    return TestClient(app), storage


def test_txt2img_returns_batch_urls(tmp_path):
    client, storage = _client(tmp_path)
    r = client.post("/txt2img", json={"prompt": "a spaceship", "batch": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2 and len(body["images"]) == 2
    # each URL is fetchable
    for img in body["images"]:
        assert client.get(img["url"]).status_code == 200


def test_txt2img_rejects_unknown_lora(tmp_path):
    client, _ = _client(tmp_path)
    r = client.post("/txt2img", json={"prompt": "x", "loras": [{"name": "ghost"}]})
    assert r.status_code == 400


def test_txt2img_rejects_batch_over_max(tmp_path):
    client, _ = _client(tmp_path)
    r = client.post("/txt2img", json={"prompt": "x", "batch": 8})
    assert r.status_code == 400


def test_txt2img_503_when_not_loaded(tmp_path):
    client, _ = _client(tmp_path, loaded=False)
    r = client.post("/txt2img", json={"prompt": "x"})
    assert r.status_code == 503


def test_txt2img_inline_single_returns_png(tmp_path):
    client, _ = _client(tmp_path)
    r = client.post("/txt2img", json={"prompt": "x", "inline": True})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
