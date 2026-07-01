from app.history import History


def _params():
    return {"prompt": "a ship", "steps": 30, "cfg": 7.0, "loras": []}


def test_add_and_list_newest_first(tmp_path):
    h = History(str(tmp_path / "h.db"))
    h.add("a" * 32, "/files/" + "a" * 32 + ".png", "txt2img", 1, _params())
    h.add("b" * 32, "/files/" + "b" * 32 + ".png", "img2img", 2, _params())
    items = h.list()
    assert [i["job_id"] for i in items] == ["b" * 32, "a" * 32]  # newest first
    assert items[0]["mode"] == "img2img" and items[0]["seed"] == 2
    assert items[0]["params"]["prompt"] == "a ship"  # params round-trip as a dict


def test_limit(tmp_path):
    h = History(str(tmp_path / "h.db"))
    for i in range(5):
        h.add(f"{i:032x}", "/files/x.png", "txt2img", i, _params())
    assert len(h.list(limit=3)) == 3


def test_delete(tmp_path):
    h = History(str(tmp_path / "h.db"))
    h.add("c" * 32, "/files/c.png", "txt2img", 1, _params())
    assert h.delete("c" * 32) is True
    assert h.list() == []
    assert h.delete("c" * 32) is False


def test_lazy_init_no_fs_on_construct(tmp_path):
    # Constructing must NOT create the DB file (import-time safety).
    db = tmp_path / "sub" / "h.db"
    History(str(db))
    assert not db.exists()
