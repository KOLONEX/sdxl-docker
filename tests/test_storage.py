from PIL import Image
from app.storage import Storage


def test_job_id_roundtrip(tmp_path):
    s = Storage(str(tmp_path))
    jid = s.new_job_id()
    assert Storage.is_valid_job_id(jid)
    assert s.url_for(jid) == f"/files/{jid}.png"


def test_rejects_bad_job_ids():
    assert not Storage.is_valid_job_id("../etc/passwd")
    assert not Storage.is_valid_job_id("XYZ")
    assert not Storage.is_valid_job_id("")


def test_save_and_delete(tmp_path):
    s = Storage(str(tmp_path))
    jid = s.new_job_id()
    path = s.save_png(jid, Image.new("RGB", (8, 8), "red"))
    assert path.is_file() and s.exists(jid)
    assert s.delete(jid) is True
    assert s.exists(jid) is False
    assert s.delete(jid) is False
