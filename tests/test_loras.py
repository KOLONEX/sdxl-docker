import pytest
from app.loras import LoraRegistry


def _touch(d, name):
    (d / name).write_bytes(b"fake")


def test_lists_safetensors_only(tmp_path):
    _touch(tmp_path, "scifi.safetensors")
    _touch(tmp_path, "notes.txt")
    reg = LoraRegistry(str(tmp_path))
    names = [l.name for l in reg.list()]
    assert names == ["scifi"]


def test_resolve_known(tmp_path):
    _touch(tmp_path, "scifi.safetensors")
    reg = LoraRegistry(str(tmp_path))
    assert reg.resolve("scifi").name == "scifi.safetensors"


def test_resolve_unknown_raises(tmp_path):
    reg = LoraRegistry(str(tmp_path))
    with pytest.raises(KeyError):
        reg.resolve("does-not-exist")


def test_resolve_rejects_traversal(tmp_path):
    _touch(tmp_path, "scifi.safetensors")
    reg = LoraRegistry(str(tmp_path))
    with pytest.raises(KeyError):
        reg.resolve("../scifi")


def test_missing_dir_lists_empty(tmp_path):
    reg = LoraRegistry(str(tmp_path / "nope"))
    assert reg.list() == []
