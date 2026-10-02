"""downloader 模块测试：本机 aiohttp 假 HF 服务器（127.0.0.1 随机端口）。"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path

import pytest
from aiohttp import web
from core.config import Settings
from core.downloader import DownloadError, ModelDownloader
from core.registry import load_registry

PLUG_ROOT = Path(__file__).resolve().parents[1]
SHA = "deadbeef" * 5  # 40 位 hex

DET_REPO = "deepghs/anime_person_detection"
CLS_REPO = "ABCwewe/wuwa_playable_character_identifier"


@pytest.fixture
def registry():
    return load_registry(PLUG_ROOT / "data" / "registry.json")


def _make_app(repo_id, files, fail_paths=()):
    """构造假 HF 应用：sha 解析 + resolve 文件下载，并统计请求/授权头。"""
    counters = {"sha": 0, "files": 0, "requests": 0, "auth": None}

    async def handler(request):
        counters["requests"] += 1
        auth = request.headers.get("Authorization")
        if auth is not None:
            counters["auth"] = auth
        path = request.rel_url.path
        if path == f"/api/models/{repo_id}/revision/main":
            counters["sha"] += 1
            return web.json_response({"sha": SHA})
        prefix = f"/{repo_id}/resolve/"
        if path.startswith(prefix):
            counters["files"] += 1
            parts = path[len(prefix) :].split("/", 1)
            if len(parts) < 2:
                return web.Response(status=404)
            file_path = parts[1]
            if file_path in fail_paths:
                return web.Response(status=404)
            if file_path in files:
                return web.Response(
                    body=files[file_path], content_type="application/octet-stream"
                )
            return web.Response(status=404)
        return web.Response(status=404)

    app = web.Application()
    app.router.add_get("/{tail:.*}", handler)
    return app, counters


def _downloader(tmp_path, registry, port, **settings_kw):
    settings = Settings(hf_endpoint=f"http://127.0.0.1:{port}", **settings_kw)
    dl = ModelDownloader(
        get_settings=lambda: settings, models_root=tmp_path, registry=registry
    )
    return dl


@pytest.mark.asyncio
async def test_ensure_downloads_and_ready(http_server, registry, tmp_path):
    app, counters = _make_app(
        DET_REPO, {"person_detect_v1.1_n/model.onnx": b"MODELBIN" * 10}
    )
    port = await http_server(app)
    dl = _downloader(tmp_path, registry, port)
    spec = registry.detector("person_detect_v1.1_n")
    paths = await dl.ensure(spec)
    assert Path(paths["model"]).is_file()
    assert Path(paths["model"]).read_bytes() == b"MODELBIN" * 10
    # revision 目录用解析出的 sha
    assert paths["model"].parent.name == SHA
    st = dl.status(spec)
    assert st.ready is True
    assert st.revision == SHA
    assert counters["sha"] == 1
    assert counters["files"] == 1
    # 第二次 ensure 不再发请求
    before = counters["files"]
    paths2 = await dl.ensure(spec)
    assert counters["files"] == before
    assert paths2["model"] == paths["model"]


@pytest.mark.asyncio
async def test_sha256_mismatch_cleans_up(http_server, registry, tmp_path, monkeypatch):
    monkeypatch.setattr("core.downloader._RETRY_BACKOFF", (0.0, 0.0, 0.0))
    app, counters = _make_app(DET_REPO, {"person_detect_v1.1_n/model.onnx": b"WRONG"})
    port = await http_server(app)
    dl = _downloader(tmp_path, registry, port)
    spec = registry.detector("person_detect_v1.1_n")
    bad = dataclasses.replace(spec, sha256={"model": "0" * 64})
    with pytest.raises(DownloadError):
        await dl.ensure(bad)
    # 本轮文件被清理：models_root 下无任何非空文件
    leftover = [p for p in tmp_path.rglob("*") if p.is_file() and p.stat().st_size > 0]
    assert leftover == []


@pytest.mark.asyncio
async def test_classifier_pair_discard_on_prototypes_failure(
    http_server, registry, tmp_path, monkeypatch
):
    monkeypatch.setattr("core.downloader._RETRY_BACKOFF", (0.0, 0.0, 0.0))
    files = {
        "latest/mnv4l_448/wuwa_playable_character_latest_mnv4l_448_int8.onnx": b"MODEL",
    }
    app, counters = _make_app(
        CLS_REPO, files, fail_paths={"latest/mnv4l_448/prototypes_int8.npz"}
    )
    port = await http_server(app)
    dl = _downloader(tmp_path, registry, port)
    spec = registry.classifier("wuwa_mnv4l_448_int8")
    with pytest.raises(DownloadError):
        await dl.ensure(spec)
    # 成对丢弃：model 也被清理
    repo_dir = tmp_path / "ABCwewe--wuwa_playable_character_identifier"
    children = list(repo_dir.rglob("*")) if repo_dir.exists() else []
    assert children == []


@pytest.mark.asyncio
async def test_display_names_404_does_not_block_ready(
    http_server, registry, tmp_path, monkeypatch
):
    monkeypatch.setattr("core.downloader._RETRY_BACKOFF", (0.0, 0.0, 0.0))
    files = {
        "latest/mnv4l_448/wuwa_playable_character_latest_mnv4l_448_int8.onnx": b"MODEL",
        "latest/mnv4l_448/prototypes_int8.npz": b"PROTO",
    }
    app, counters = _make_app(CLS_REPO, files, fail_paths={"class_names_zh.yaml"})
    port = await http_server(app)
    dl = _downloader(tmp_path, registry, port)
    spec = registry.classifier("wuwa_mnv4l_448_int8")
    paths = await dl.ensure(spec)
    assert "display_names" not in paths
    assert "model" in paths and "prototypes" in paths
    assert dl.status(spec).ready is True


@pytest.mark.asyncio
async def test_offline_mode_missing_raises(registry, tmp_path):
    settings = Settings(offline_mode=True)
    dl = ModelDownloader(
        get_settings=lambda: settings, models_root=tmp_path, registry=registry
    )
    spec = registry.detector("person_detect_v1.1_n")
    with pytest.raises(DownloadError):
        await dl.ensure(spec)


@pytest.mark.asyncio
async def test_concurrent_ensure_single_download(http_server, registry, tmp_path):
    app, counters = _make_app(
        DET_REPO, {"person_detect_v1.1_n/model.onnx": b"MODELBIN" * 10}
    )
    port = await http_server(app)
    dl = _downloader(tmp_path, registry, port)
    spec = registry.detector("person_detect_v1.1_n")
    results = await asyncio.gather(*[dl.ensure(spec) for _ in range(5)])
    assert len({str(r["model"]) for r in results}) == 1
    assert counters["files"] == 1  # 锁生效：只触发一次文件下载


@pytest.mark.asyncio
async def test_token_sent_in_authorization(http_server, registry, tmp_path):
    app, counters = _make_app(
        DET_REPO, {"person_detect_v1.1_n/model.onnx": b"MODELBIN" * 10}
    )
    port = await http_server(app)
    settings = Settings(hf_endpoint=f"http://127.0.0.1:{port}", hf_token="tok123")
    dl = ModelDownloader(
        get_settings=lambda: settings, models_root=tmp_path, registry=registry
    )
    spec = registry.detector("person_detect_v1.1_n")
    await dl.ensure(spec)
    assert counters["auth"] == "Bearer tok123"


@pytest.mark.asyncio
async def test_status_before_download_is_not_ready(registry, tmp_path):
    dl = ModelDownloader(
        get_settings=lambda: Settings(), models_root=tmp_path, registry=registry
    )
    spec = registry.detector("person_detect_v1.1_n")
    st = dl.status(spec)
    assert st.ready is False
    assert all(f.path is None for f in st.files)


@pytest.mark.asyncio
async def test_disk_usage_counts_ready_files(http_server, registry, tmp_path):
    payload = b"X" * 100
    app, counters = _make_app(DET_REPO, {"person_detect_v1.1_n/model.onnx": payload})
    port = await http_server(app)
    dl = _downloader(tmp_path, registry, port)
    spec = registry.detector("person_detect_v1.1_n")
    await dl.ensure(spec)
    assert dl.disk_usage() == len(payload)
