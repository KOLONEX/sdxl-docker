import asyncio
from pathlib import Path
from fastapi.testclient import TestClient
from app.api import create_app
from app.pipeline import PipelineManager, FakeBackend
from app.storage import Storage
from app.loras import LoraRegistry
from app.history import History

WEB = str(Path(__file__).resolve().parent.parent / "app" / "web")


def _client(tmp_path):
    storage = Storage(str(tmp_path / "out"))
    manager = PipelineManager(FakeBackend(), storage)
    asyncio.run(manager.load())
    history = History(str(tmp_path / "hist.db"))
    app = create_app(manager, storage, LoraRegistry(str(tmp_path / "loras")), WEB,
                     max_batch=4, history=history)
    return TestClient(app)


def test_generation_is_recorded(tmp_path):
    client = _client(tmp_path)
    r = client.post("/txt2img", json={"prompt": "a ship", "batch": 2})
    assert r.status_code == 200
    hist = client.get("/history").json()["items"]
    assert len(hist) == 2  # one row per image
    assert hist[0]["mode"] == "txt2img"
    assert hist[0]["params"]["prompt"] == "a ship"
    assert "inline" not in hist[0]["params"]  # inline excluded from stored params


def test_history_limit(tmp_path):
    client = _client(tmp_path)
    client.post("/txt2img", json={"prompt": "x", "batch": 4})
    assert len(client.get("/history?limit=2").json()["items"]) == 2


def test_delete_history_entry(tmp_path):
    client = _client(tmp_path)
    client.post("/txt2img", json={"prompt": "x"})
    jid = client.get("/history").json()["items"][0]["job_id"]
    assert client.delete(f"/history/{jid}").status_code == 204
    assert client.get("/history").json()["items"] == []


def test_history_empty_without_history_backend(tmp_path):
    # create_app without a History still exposes /history (empty)
    storage = Storage(str(tmp_path / "out"))
    manager = PipelineManager(FakeBackend(), storage)
    asyncio.run(manager.load())
    app = create_app(manager, storage, LoraRegistry(str(tmp_path / "loras")), WEB)
    client = TestClient(app)
    assert client.get("/history").json() == {"items": []}
