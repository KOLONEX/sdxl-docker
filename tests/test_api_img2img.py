import asyncio, base64, io
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image
from app.api import create_app
from app.pipeline import PipelineManager, FakeBackend
from app.storage import Storage
from app.loras import LoraRegistry

WEB = str(Path(__file__).resolve().parent.parent / "app" / "web")


def _client(tmp_path):
    storage = Storage(str(tmp_path / "out"))
    manager = PipelineManager(FakeBackend(), storage)
    asyncio.run(manager.load())
    app = create_app(manager, storage, LoraRegistry(str(tmp_path / "loras")), WEB)
    return TestClient(app)


def _png_bytes():
    buf = io.BytesIO(); Image.new("RGB", (64, 64), "blue").save(buf, "PNG")
    return buf.getvalue()


def test_img2img_multipart(tmp_path):
    client = _client(tmp_path)
    r = client.post("/img2img", data={"prompt": "x", "denoise": "0.5"},
                    files={"file": ("in.png", _png_bytes(), "image/png")})
    assert r.status_code == 200 and r.json()["count"] == 1


def test_img2img_json_base64(tmp_path):
    client = _client(tmp_path)
    b64 = base64.b64encode(_png_bytes()).decode()
    r = client.post("/img2img/json", json={"prompt": "x", "image_base64": b64})
    assert r.status_code == 200 and r.json()["count"] == 1


def test_img2img_rejects_invalid_image(tmp_path):
    client = _client(tmp_path)
    r = client.post("/img2img", data={"prompt": "x"},
                    files={"file": ("in.png", b"not-an-image", "image/png")})
    assert r.status_code == 400
