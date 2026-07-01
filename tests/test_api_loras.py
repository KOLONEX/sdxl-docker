from pathlib import Path
from fastapi.testclient import TestClient
from app.api import create_app
from app.pipeline import PipelineManager, FakeBackend
from app.storage import Storage
from app.loras import LoraRegistry

WEB = str(Path(__file__).resolve().parent.parent / "app" / "web")


def test_lists_loras(tmp_path):
    loras = tmp_path / "loras"; loras.mkdir()
    (loras / "scifi.safetensors").write_bytes(b"x")
    storage = Storage(str(tmp_path / "out"))
    manager = PipelineManager(FakeBackend(), storage)
    app = create_app(manager, storage, LoraRegistry(str(loras)), WEB)
    client = TestClient(app)
    r = client.get("/loras")
    assert r.status_code == 200
    assert r.json()["loras"][0]["name"] == "scifi"
