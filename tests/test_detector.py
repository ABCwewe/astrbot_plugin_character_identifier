"""detector 模块测试：假会话 + 合成图，几何反推还原坐标、过滤、NMS、分支判别。"""

from __future__ import annotations

import numpy as np
import pytest
from core.detector import DetectorError, PersonDetector

from tests.conftest import FakeDetectorSession

IMGSZ = 640


def _regular_out(rows, n_anchors=100):
    """rows: [(cx, cy, w, h, conf), ...] → [1, 5, N] 常规分支输出。"""
    arr = np.zeros((1, 5, n_anchors), dtype=np.float32)
    for i, (cx, cy, w, h, conf) in enumerate(rows):
        arr[0, 0, i] = cx
        arr[0, 1, i] = cy
        arr[0, 2, i] = w
        arr[0, 3, i] = h
        arr[0, 4, i] = conf
    return arr


def _make(
    image_bgr,
    rows,
    conf=0.327,
    nms_iou=0.5,
    max_persons=8,
    min_box_px=24,
    out_shape=None,
    session=None,
    branch_out=None,
):
    h, w = image_bgr.shape[:2]
    scale = IMGSZ / max(h, w)
    nw = max(1, round(w * scale))
    nh = max(1, round(h * scale))
    dx = (IMGSZ - nw) / 2.0
    dy = (IMGSZ - nh) / 2.0
    if branch_out is not None:
        oshape, out = branch_out
    else:
        oshape, out = [1, 5, 100], _regular_out(rows)
    if session is None:
        session = FakeDetectorSession(oshape, out)
    det = PersonDetector(
        session,
        imgsz=IMGSZ,
        conf=conf,
        nms_iou=nms_iou,
        max_persons=max_persons,
        min_box_px=min_box_px,
    )
    return det, dx, dy, scale, session


def test_regular_branch_geometry_recovery(synthetic_image):
    """800x600 图：scale=0.8, dy=80，高置信框反推回原图坐标，误差 <1.5px。"""
    img = synthetic_image(800, 600, seed=1)
    # 原图框 (100,50)-(400,350) → cx=250, cy=200, w=h=300
    # letterbox: cx'=250*0.8+0=200, cy'=200*0.8+80=240, w'=h'=240
    det, dx, dy, scale, session = _make(img, [(200, 240, 240, 240, 0.95)], conf=0.5)
    results = det.detect(img)
    assert len(results) == 1
    r = results[0]
    x1, y1, x2, y2 = r.xyxy
    assert abs(x1 - 100) < 1.5
    assert abs(y1 - 50) < 1.5
    assert abs(x2 - 400) < 1.5
    assert abs(y2 - 350) < 1.5
    assert r.conf == pytest.approx(0.95)
    # 送入会话的 blob 形状与值域
    blob = session.record["blob"]
    assert blob.shape == (1, 3, IMGSZ, IMGSZ)
    assert blob.dtype == np.float32
    assert blob.min() >= 0.0 and blob.max() <= 1.0


def test_low_conf_filtered(synthetic_image):
    """低于 det_conf 的框被过滤。"""
    img = synthetic_image(640, 480, seed=2)  # scale=1, dy=80
    det, _, _, _, _ = _make(
        img,
        [(175, 255, 150, 150, 0.9), (475, 255, 150, 150, 0.2)],
        conf=0.5,
    )
    results = det.detect(img)
    assert len(results) == 1
    assert results[0].conf == pytest.approx(0.9)


def test_nms_deduplicates_overlap(synthetic_image):
    """两个高重叠框只保留高分者。"""
    img = synthetic_image(640, 480, seed=3)
    det, _, _, _, _ = _make(
        img,
        [(175, 255, 150, 150, 0.9), (195, 275, 150, 150, 0.8)],
        conf=0.5,
    )
    results = det.detect(img)
    assert len(results) == 1
    assert results[0].conf == pytest.approx(0.9)


def test_min_box_px_filtered(synthetic_image):
    """短边 < min_box_px 的框被过滤，正常框保留。"""
    img = synthetic_image(640, 480, seed=4)
    det, _, _, _, _ = _make(
        img,
        [(110, 190, 20, 20, 0.9), (100, 230, 100, 100, 0.9)],  # 20px 框 + 正常框
        conf=0.5,
    )
    results = det.detect(img)
    assert len(results) == 1
    # 保留的是大框：反推坐标 (50,100)-(150,200)
    x1, y1, x2, y2 = results[0].xyxy
    assert abs(x1 - 50) < 1.5 and abs(y1 - 100) < 1.5
    assert abs(x2 - 150) < 1.5 and abs(y2 - 200) < 1.5


def test_max_persons_truncation(synthetic_image):
    """置信度降序取前 max_persons。"""
    img = synthetic_image(640, 480, seed=5)
    det, _, _, _, _ = _make(
        img,
        [
            (125, 230, 150, 100, 0.95),
            (375, 230, 150, 100, 0.85),
            (625, 230, 150, 100, 0.75),
        ],
        conf=0.5,
        max_persons=2,
    )
    results = det.detect(img)
    assert [r.conf for r in results] == [pytest.approx(0.95), pytest.approx(0.85)]


def test_auto_branch_dynamic_output(synthetic_image):
    """输出形状全动态（['batch','anchors','x']）→ auto 分支按实际输出判别为 regular。"""
    img = synthetic_image(800, 600, seed=6)
    out = _regular_out([(200, 240, 240, 240, 0.95)], n_anchors=100)
    det, _, _, _, session = _make(
        img, [], conf=0.5, branch_out=(["batch", "anchors", "x"], out)
    )
    assert det._branch == "auto"
    results = det.detect(img)
    assert det._branch == "regular"
    assert len(results) == 1
    x1, y1, x2, y2 = results[0].xyxy
    assert abs(x1 - 100) < 1.5 and abs(y1 - 50) < 1.5
    assert abs(x2 - 400) < 1.5 and abs(y2 - 350) < 1.5


def test_end2end_branch(synthetic_image):
    """end2end [1,N,6]：行 = x1,y1,x2,y2,conf,cls，直接还原。"""
    img = synthetic_image(800, 600, seed=7)
    out = np.zeros((1, 100, 6), dtype=np.float32)
    # letterbox 坐标：x1=80, y1=120, x2=240, y2=280（对应原图 (100,50)-(300,250)）
    out[0, 0] = [80, 120, 240, 280, 0.9, 0]
    det, _, _, _, _ = _make(img, [], conf=0.5, branch_out=([1, 100, 6], out))
    results = det.detect(img)
    assert len(results) == 1
    x1, y1, x2, y2 = results[0].xyxy
    assert abs(x1 - 100) < 1.5 and abs(y1 - 50) < 1.5
    assert abs(x2 - 300) < 1.5 and abs(y2 - 250) < 1.5
    assert results[0].conf == pytest.approx(0.9)


def test_auto_branch_end2end(synthetic_image):
    """auto 分支按实际输出判别为 end2end。"""
    img = synthetic_image(800, 600, seed=8)
    out = np.zeros((1, 100, 6), dtype=np.float32)
    out[0, 0] = [80, 120, 240, 280, 0.9, 0]
    det, _, _, _, _ = _make(
        img, [], conf=0.5, branch_out=(["batch", "anchors", "x"], out)
    )
    assert det._branch == "auto"
    results = det.detect(img)
    assert det._branch == "end2end"
    assert len(results) == 1


def test_unrecognizable_output_raises(synthetic_image):
    """实际输出 [1,7,7] 无法判别 → DetectorError。"""
    img = synthetic_image(640, 480, seed=9)
    out = np.zeros((1, 7, 7), dtype=np.float32)
    det, _, _, _, _ = _make(
        img, [], conf=0.5, branch_out=(["batch", "anchors", "x"], out)
    )
    with pytest.raises(DetectorError):
        det.detect(img)


def test_static_input_size_overrides_imgsz():
    """模型静态输入 H/W 与配置不一致时以模型为准。"""
    session = FakeDetectorSession([1, 5, 100], np.zeros((1, 5, 100), np.float32))
    session.get_inputs = lambda: [
        type(
            "N",
            (),
            {"name": "images", "shape": [1, 3, 512, 512], "type": "tensor(float)"},
        )()
    ]
    det = PersonDetector(
        session, imgsz=IMGSZ, conf=0.5, nms_iou=0.5, max_persons=8, min_box_px=24
    )
    assert det.input_size == 512


def test_non_square_static_input_raises():
    session = FakeDetectorSession([1, 5, 100], np.zeros((1, 5, 100), np.float32))
    session.get_inputs = lambda: [
        type(
            "N",
            (),
            {"name": "images", "shape": [1, 3, 640, 480], "type": "tensor(float)"},
        )()
    ]
    with pytest.raises(DetectorError):
        PersonDetector(
            session, imgsz=IMGSZ, conf=0.5, nms_iou=0.5, max_persons=8, min_box_px=24
        )


def test_empty_image_returns_empty(synthetic_image):
    """空图返回 []（不抛）。"""
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    det, _, _, _, _ = _make(img, [], conf=0.5)
    assert det.detect(img) == []
