"""pipeline 模块测试：stub runtime/downloader + 真 registry + 真 Settings。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from core.config import Settings
from core.detector import Detection
from core.downloader import ModelFileStatus, ModelStatus
from core.image_io import SourceImage
from core.pipeline import Pipeline
from core.registry import load_registry

PLUG_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def registry():
    return load_registry(PLUG_ROOT / "data" / "registry.json")


class FakeDetector:
    """可编程检测 stub：记录调用次数。"""

    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = 0

    def detect(self, image_bgr):
        self.calls += 1
        return list(self.boxes)


class FakeClassifier:
    """可编程分类 stub。"""

    def __init__(self, results):
        self.results = results
        self.calls = 0

    def predict_batch(self, crops):
        self.calls += 1
        return list(self.results)


class FakeRuntime:
    """pipeline 依赖的最小 runtime 替代：acquire/release + 同步 executor。"""

    def __init__(self, detector, classifier):
        self.detector = detector
        self.classifier = classifier

    async def acquire_detector(self):
        return self.detector

    async def acquire_classifier(self):
        return self.classifier

    async def release_detector(self):
        pass

    async def release_classifier(self):
        pass

    async def run_in_executor(self, func, *args):
        return func(*args)

    def snapshot(self):
        return {
            "kinds": {
                "detector": {"loaded": True},
                "classifier": {"loaded": True, "npz_version": 1},
            }
        }


class FakeDownloader:
    """pipeline 依赖的下载器替代：status 固定就绪，ensure 返回无 display_names。"""

    def __init__(self):
        self.ensures = 0

    async def ensure(self, spec):
        self.ensures += 1
        return {"model": Path("x"), "prototypes": Path("x")}

    def status(self, spec):
        return ModelStatus(
            spec_key=spec.key,
            ready=True,
            files=[ModelFileStatus("prototypes", None, None)],
            revision="rev1",
            total_bytes=0,
        )


def _make_pipeline(registry, cache_dir, detector, classifier):
    rt = FakeRuntime(detector, classifier)
    dl = FakeDownloader()
    pipeline = Pipeline(
        get_settings=lambda: Settings(),
        registry=registry,
        runtime=rt,
        downloader=dl,
        cache_dir=cache_dir,
    )
    return pipeline, rt, dl


def _src(index=0, sha1=None, w=320, h=240):
    sha1 = sha1 or "a" * 40
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = 90
    return SourceImage(index=index, sha1=sha1, size_bytes=1, bgr=img, scaled=False)


@pytest.mark.asyncio
async def test_recognized_writes_jpg_and_persons(registry, tmp_path, make_result):
    cache = tmp_path / "cache"
    detector = FakeDetector(
        [
            Detection(xyxy=(10, 10, 160, 120), conf=0.9),
            Detection(xyxy=(165, 10, 310, 120), conf=0.8),
        ]
    )
    classifier = FakeClassifier(
        [
            make_result(pred="jinhsi_(wuthering_waves)", head_conf=0.93),
            make_result(pred=None, unknown=True),
        ]
    )
    pipeline, rt, dl = _make_pipeline(registry, cache, detector, classifier)
    image = _src(w=320, h=240)
    outcomes = await pipeline.run([image])
    assert len(outcomes) == 1
    o = outcomes[0]
    assert o.annotated_path is not None
    assert o.annotated_path.exists()
    assert o.annotated_path.stat().st_size > 0
    assert o.annotated_path.name == f"{image.sha1[:16]}.jpg"
    # persons 编号 1..N 连续
    assert [p.index for p in o.persons] == [1, 2]
    assert o.persons[0].display_name == "jinhsi"
    assert o.persons[0].color_name == "橙色"
    assert o.persons[0].head_conf == 0.93
    assert o.persons[1].display_name is None
    assert o.persons[1].color_name == "灰色"
    assert o.scope_note == "《鸣潮》可操控角色"


@pytest.mark.asyncio
async def test_all_unknown_no_annotate_and_cached(registry, tmp_path, make_result):
    cache = tmp_path / "cache"
    detector = FakeDetector([Detection(xyxy=(10, 10, 160, 120), conf=0.9)])
    classifier = FakeClassifier([make_result(pred=None, unknown=True)])
    pipeline, rt, dl = _make_pipeline(registry, cache, detector, classifier)
    image = _src()
    out1 = await pipeline.run([image])
    assert out1[0].annotated_path is None
    assert out1[0].persons == []
    # 第二次 run 命中 LRU：stub detect 不再被调用
    out2 = await pipeline.run([image])
    assert out2[0].annotated_path is None
    assert detector.calls == 1


@pytest.mark.asyncio
async def test_no_boxes_no_annotate_and_cached(registry, tmp_path):
    cache = tmp_path / "cache"
    detector = FakeDetector([])
    classifier = FakeClassifier([])
    pipeline, rt, dl = _make_pipeline(registry, cache, detector, classifier)
    image = _src()
    await pipeline.run([image])
    assert detector.calls == 1
    await pipeline.run([image])
    assert detector.calls == 1  # 缓存命中


@pytest.mark.asyncio
async def test_lru_capacity_64_evicts(registry, tmp_path):
    """塞 65 张不同 sha 的 1x1 jpg 小图 → 第一张被淘汰后重新推理。"""
    cache = tmp_path / "cache"
    detector = FakeDetector([])  # 无框 → 空结果也入缓存
    classifier = FakeClassifier([])
    pipeline, rt, dl = _make_pipeline(registry, cache, detector, classifier)
    imgs = []
    for i in range(65):
        pixel = np.zeros((1, 1, 3), dtype=np.uint8)
        pixel[:] = (i * 7 % 256, i * 13 % 256, i * 29 % 256)
        ok, buf = cv2.imencode(".jpg", pixel)
        assert ok
        data = buf.tobytes()
        imgs.append(
            SourceImage(
                index=i,
                sha1=hashlib.sha1(data).hexdigest(),
                size_bytes=len(data),
                bgr=pixel,
                scaled=False,
            )
        )
    await pipeline.run(imgs)
    assert detector.calls == 65
    # 第一张已被 LRU 淘汰 → 重跑触发重新推理
    await pipeline.run([imgs[0]])
    assert detector.calls == 66


def test_crop_full_frame_stays_in_bounds(synthetic_image):
    """全图框外扩 5% 贴边不越界 → 裁剪 == 全图。"""
    img = synthetic_image(100, 80)
    h, w = img.shape[:2]
    crops = Pipeline._crop_boxes(
        img, [Detection(xyxy=(0.0, 0.0, float(w - 1), float(h - 1)), conf=1.0)]
    )
    assert len(crops) == 1
    assert np.array_equal(crops[0], img)


def test_crop_pad_expands_interior(synthetic_image):
    """内部框外扩 5% 后裁剪大于原框。"""
    img = synthetic_image(400, 400)
    box = Detection(xyxy=(100.0, 100.0, 200.0, 200.0), conf=1.0)
    crops = Pipeline._crop_boxes(img, [box])
    # 100px 框，pw=ph=5 → left=95, right=205, top=95, bottom=205
    assert crops[0].shape == (110, 110, 3)
