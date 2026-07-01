# tests/test_frontend.py
from pathlib import Path

INDEX = Path(__file__).resolve().parent.parent / "app" / "web" / "index.html"


def test_index_exists_and_has_key_hooks():
    html = INDEX.read_text(encoding="utf-8")
    # endpoints wired
    for token in ["/txt2img", "/img2img", "/loras"]:
        assert token in html
    # bilingual switcher present
    assert "data-i18n" in html or "lang-es" in html
    # core controls present
    for token in ["id=\"prompt\"", "id=\"batch\"", "id=\"sampler\"", "id=\"loras\""]:
        assert token in html
