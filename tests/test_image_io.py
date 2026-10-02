"""image_io 模块测试：各种 URL 形态、跳过策略、max_images 截断、网络取图、sha1。"""

from __future__ import annotations

import base64
import hashlib

import cv2
import numpy as np
import pytest
from aiohttp import web
from core.config import Settings
from core.image_io import ImageCollector


def _png_bytes(size=(64, 48), color=(200, 100, 50)) -> bytes:
    img = np.zeros((*size, 3), dtype=np.uint8)
    img[:] = color
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.mark.asyncio
async def test_base64_uri():
    png = _png_bytes()
    b64 = base64.b64encode(png).decode()
    collector = ImageCollector(lambda: Settings())
    imgs = await collector.collect([f"base64://{b64}"])
    assert len(imgs) == 1
    assert imgs[0].sha1 == hashlib.sha1(png).hexdigest()
    assert imgs[0].size_bytes == len(png)


@pytest.mark.asyncio
async def test_data_uri():
    png = _png_bytes()
    b64 = base64.b64encode(png).decode()
    collector = ImageCollector(lambda: Settings())
    imgs = await collector.collect([f"data:image/png;base64,{b64}"])
    assert len(imgs) == 1
    assert imgs[0].sha1 == hashlib.sha1(png).hexdigest()


@pytest.mark.asyncio
async def test_local_file_forms(tmp_path):
    png = _png_bytes()
    p = tmp_path / "img.png"
    p.write_bytes(png)
    collector = ImageCollector(lambda: Settings())
    # 裸路径
    imgs = await collector.collect([str(p)])
    assert len(imgs) == 1
    assert imgs[0].sha1 == hashlib.sha1(png).hexdigest()
    # file://C:/...
    imgs = await collector.collect([f"file://{p.as_posix()}"])
    assert len(imgs) == 1
    # file:///C:/...（Windows 三斜杠）
    imgs = await collector.collect([f"file:///{p.as_posix()}"])
    assert len(imgs) == 1


@pytest.mark.asyncio
async def test_missing_path_skipped(tmp_path):
    collector = ImageCollector(lambda: Settings())
    assert await collector.collect([str(tmp_path / "nope.png")]) == []


@pytest.mark.asyncio
async def test_corrupt_bytes_skipped(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"definitely not an image")
    collector = ImageCollector(lambda: Settings())
    assert await collector.collect([str(bad)]) == []


@pytest.mark.asyncio
async def test_max_images_truncation_keeps_index():
    png = _png_bytes()
    b64 = base64.b64encode(png).decode()
    collector = ImageCollector(lambda: Settings(max_images=3))
    imgs = await collector.collect([f"base64://{b64}"] * 5)
    assert [i.index for i in imgs] == [0, 1, 2]
    # 第 2 张可解码时 SourceImage.index == 1（下标保留）
    collector2 = ImageCollector(lambda: Settings(max_images=2))
    imgs2 = await collector2.collect(["base64://!!!bad", f"base64://{b64}"])
    assert [i.index for i in imgs2] == [1]


@pytest.mark.asyncio
async def test_over_max_pixels_skipped(tmp_path, monkeypatch):
    import core.image_io as io

    png = _png_bytes(size=(200, 200))
    p = tmp_path / "big.png"
    p.write_bytes(png)
    monkeypatch.setattr(io, "_MAX_PIXELS", 100)
    collector = ImageCollector(lambda: Settings())
    assert await collector.collect([str(p)]) == []


@pytest.mark.asyncio
async def test_det_pre_max_side_scales_and_flags(tmp_path):
    """最长边超过 det_pre_max_side → INTER_AREA 整体缩小，scaled=True。"""
    png = _png_bytes(size=(64, 48))
    p = tmp_path / "img.png"
    p.write_bytes(png)
    collector = ImageCollector(lambda: Settings(det_pre_max_side=32))
    imgs = await collector.collect([str(p)])
    assert len(imgs) == 1
    assert imgs[0].scaled is True
    assert max(imgs[0].bgr.shape[:2]) == 32
    # 未超限 → scaled=False
    collector2 = ImageCollector(lambda: Settings(det_pre_max_side=2048))
    imgs2 = await collector2.collect([str(p)])
    assert imgs2[0].scaled is False
    assert imgs2[0].bgr.shape[:2] == (64, 48)


# ---------------- 网络 ----------------


@pytest.mark.asyncio
async def test_http_200(http_server):
    png = _png_bytes()
    app = web.Application()

    async def ok_handler(request):
        return web.Response(body=png, content_type="image/png")

    app.router.add_get("/ok.png", ok_handler)
    port = await http_server(app)
    collector = ImageCollector(lambda: Settings())
    imgs = await collector.collect([f"http://127.0.0.1:{port}/ok.png"])
    assert len(imgs) == 1
    assert imgs[0].sha1 == hashlib.sha1(png).hexdigest()


@pytest.mark.asyncio
async def test_http_404_skipped(http_server):
    app = web.Application()

    async def not_found(request):
        return web.Response(status=404)

    app.router.add_get("/missing.png", not_found)
    port = await http_server(app)
    collector = ImageCollector(lambda: Settings())
    assert await collector.collect([f"http://127.0.0.1:{port}/missing.png"]) == []


@pytest.mark.asyncio
async def test_http_text_content_type_skipped(http_server):
    app = web.Application()

    async def text_handler(request):
        return web.Response(body=b"not an image", content_type="text/plain")

    app.router.add_get("/text.txt", text_handler)
    port = await http_server(app)
    collector = ImageCollector(lambda: Settings())
    assert await collector.collect([f"http://127.0.0.1:{port}/text.txt"]) == []


@pytest.mark.asyncio
async def test_http_svg_skipped(http_server):
    app = web.Application()

    async def svg_handler(request):
        return web.Response(body=b"<svg/>", content_type="image/svg+xml")

    app.router.add_get("/a.svg", svg_handler)
    port = await http_server(app)
    collector = ImageCollector(lambda: Settings())
    assert await collector.collect([f"http://127.0.0.1:{port}/a.svg"]) == []


@pytest.mark.asyncio
async def test_mixed_urls_partial_success(http_server, tmp_path):
    """混合形态：本地好图 + 网络图 + 坏图 → 只收可用项且下标正确。"""
    png = _png_bytes()
    local = tmp_path / "ok.png"
    local.write_bytes(png)
    app = web.Application()

    async def ok_handler(request):
        return web.Response(body=png, content_type="image/png")

    app.router.add_get("/ok.png", ok_handler)
    port = await http_server(app)
    collector = ImageCollector(lambda: Settings())
    imgs = await collector.collect(
        [
            str(local),  # 0
            f"http://127.0.0.1:{port}/ok.png",  # 1
            "base64://!!!bad",  # 2
            str(tmp_path / "missing.png"),  # 3
        ]
    )
    assert [i.index for i in imgs] == [0, 1]
