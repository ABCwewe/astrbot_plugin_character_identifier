"""runtime 模块测试：monkeypatch 会话构建，验证引用计数、卸载、指纹重建、终止、快照。"""

from __future__ import annotations

from pathlib import Path

import pytest
from core.config import Settings
from core.registry import load_registry
from core.runtime import ModelRuntime

PLUG_ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    """递增 id 的假模型对象。"""

    _counter = 0

    def __init__(self):
        type(self)._counter += 1
        self.id = type(self)._counter


class FakeDownloader:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path

    async def ensure(self, spec):
        return {"model": self.tmp_path, "prototypes": self.tmp_path}


@pytest.fixture
def make_runtime(tmp_path, monkeypatch):
    registry = load_registry(PLUG_ROOT / "data" / "registry.json")

    def _make(get_settings):
        FakeModel._counter = 0

        def _build_model(self, kind, spec, paths, settings):
            return FakeModel(), {"npz_version": 1, "input_size": 448}

        monkeypatch.setattr(ModelRuntime, "_build_model", _build_model)
        monkeypatch.setattr(
            ModelRuntime, "_build_session", lambda self, path, settings: None
        )
        rt = ModelRuntime(
            get_settings=get_settings,
            registry=registry,
            downloader=FakeDownloader(tmp_path),
            models_root=tmp_path / "models",
        )
        return rt

    return _make


@pytest.mark.asyncio
async def test_acquire_builds_once_and_release_zeroes(make_runtime):
    rt = make_runtime(lambda: Settings(resident=False, idle_timeout_sec=30))
    try:
        det = await rt.acquire_detector()
        assert isinstance(det, FakeModel)
        assert FakeModel._counter == 1
        assert rt.snapshot()["kinds"]["detector"]["loaded"] is True
        # 已加载再次 acquire 不重建
        await rt.acquire_detector()
        assert FakeModel._counter == 1
        await rt.release_detector()
        await rt.release_detector()
        assert rt.snapshot()["kinds"]["detector"]["inflight"] == 0
    finally:
        await rt.terminate()


@pytest.mark.asyncio
async def test_unload_skips_inflight_and_keeps_model(make_runtime):
    rt = make_runtime(lambda: Settings(resident=False, idle_timeout_sec=30))
    try:
        await rt.acquire_detector()  # inflight=1
        n = await rt.unload("detector")
        assert n == 0
        assert rt.snapshot()["kinds"]["detector"]["loaded"] is True
        assert rt.snapshot()["kinds"]["detector"]["inflight"] == 1
        await rt.release_detector()
        n = await rt.unload("detector")
        assert n == 1
        assert rt.snapshot()["kinds"]["detector"]["loaded"] is False
    finally:
        await rt.terminate()


@pytest.mark.asyncio
async def test_unload_then_reacquire_rebuilds(make_runtime):
    rt = make_runtime(lambda: Settings(resident=False, idle_timeout_sec=30))
    try:
        await rt.acquire_detector()  # 1
        await rt.release_detector()
        await rt.unload("detector")
        await rt.acquire_detector()  # 2（重建）
        assert FakeModel._counter == 2
    finally:
        await rt.terminate()


@pytest.mark.asyncio
async def test_settings_fingerprint_change_rebuilds(make_runtime):
    holder = [Settings(resident=False, idle_timeout_sec=30, threads=2)]
    rt = make_runtime(lambda: holder[0])
    try:
        await rt.acquire_detector()
        await rt.release_detector()
        holder[0] = Settings(resident=False, idle_timeout_sec=30, threads=3)
        await rt.acquire_detector()  # 指纹变化 → 重建
        assert FakeModel._counter == 2
    finally:
        await rt.terminate()


@pytest.mark.asyncio
async def test_terminate_blocks_acquire(make_runtime):
    rt = make_runtime(lambda: Settings(resident=False, idle_timeout_sec=30))
    await rt.terminate()
    with pytest.raises(RuntimeError):
        await rt.acquire_detector()


@pytest.mark.asyncio
async def test_snapshot_fields(make_runtime):
    rt = make_runtime(lambda: Settings(resident=False, idle_timeout_sec=30, threads=2))
    try:
        await rt.acquire_classifier()
        snap = rt.snapshot()
        assert snap["threads"] == 2
        assert snap["max_concurrency"] == 1
        assert snap["resident"] is False
        assert snap["kinds"]["classifier"]["loaded"] is True
        assert snap["kinds"]["classifier"]["npz_version"] == 1
        assert snap["kinds"]["classifier"]["key"] == "wuwa_mnv4l_448_int8"
        assert snap["kinds"]["detector"]["loaded"] is False
    finally:
        await rt.terminate()
