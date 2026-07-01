import asyncio
from app.pipeline import PipelineManager, FakeBackend
from app.storage import Storage
from app.schemas import Txt2ImgParams, Img2ImgParams
from PIL import Image


def _mgr(tmp_path):
    return PipelineManager(FakeBackend(), Storage(str(tmp_path)))


def test_txt2img_saves_batch(tmp_path):
    m = _mgr(tmp_path)
    asyncio.run(m.load())
    res = asyncio.run(m.generate_txt2img(Txt2ImgParams(prompt="x", batch=3, seed=100)))
    assert res.count == 3 and len(res.job_ids) == 3
    assert res.seeds == [100, 101, 102]
    for p in res.paths:
        assert p.is_file()


def test_img2img_runs(tmp_path):
    m = _mgr(tmp_path)
    asyncio.run(m.load())
    img = Image.new("RGB", (64, 64), "blue")
    res = asyncio.run(m.generate_img2img(Img2ImgParams(prompt="x"), img))
    assert res.count == 1 and res.paths[0].is_file()


def test_mutex_serializes(tmp_path):
    backend = FakeBackend()
    m = PipelineManager(backend, Storage(str(tmp_path)))
    asyncio.run(m.load())

    async def hammer():
        await asyncio.gather(*[
            m.generate_txt2img(Txt2ImgParams(prompt="x")) for _ in range(5)
        ])

    asyncio.run(hammer())
    assert backend.max_in_flight == 1
