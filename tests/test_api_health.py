from pathlib import Path
from fastapi.testclient import TestClient
from app.api import create_app
from app.pipeline import PipelineManager, FakeBackend
from app.storage import Storage
from app.loras import LoraRegistry

WEB = str(Path(__file__).resolve().parent.parent / "app" / "web")


def _client(tmp_path):
    storage = Storage(str(tmp_path / "out"))
    registry = LoraRegistry(str(tmp_path / "loras"))
    manager = PipelineManager(FakeBackend(), storage)
    app = create_app(manager, storage, registry, WEB)
    return TestClient(app), manager


def test_health_reports_loading(tmp_path):
    client, _ = _client(tmp_path)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is False
