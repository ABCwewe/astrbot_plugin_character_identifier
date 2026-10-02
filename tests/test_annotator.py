"""annotator 模块测试：渲染不修改输入、调色板/灰框像素、缩放、JPEG、清理、字体回落。"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np
from core.annotator import GREY_BGR, PALETTE_BGR, Annotator, purge_old_files


def test_render_returns_new_array_input_untouched(synthetic_image, make_result):
    img = synthetic_image(400, 300, seed=2)
    original = img.copy()
    boxes = [(50, 60, 250, 260), (270, 60, 380, 200)]
    results = [make_result(), make_result(pred=None, unknown=True)]
    out = Annotator().render(img, boxes, results)
    assert out is not img
    assert out.shape == img.shape
    assert np.array_equal(img, original)  # 输入未被修改
    # 已识别 → 橙色框（PALETTE_BGR[0]）实色标签像素存在
    assert np.any(np.all(out == PALETTE_BGR[0], axis=2))
    # unknown → 灰框（GREY_BGR）像素存在
    assert np.any(np.all(out == GREY_BGR, axis=2))


def test_long_side_resized_to_out_max_side(synthetic_image):
    img = synthetic_image(2560, 1440, seed=3)
    out = Annotator(out_max_side=1280).render(img, [], [])
    assert max(out.shape[0], out.shape[1]) == 1280
    # 小于上限不缩放
    small = synthetic_image(400, 300, seed=7)
    out2 = Annotator(out_max_side=1280).render(small, [], [])
    assert out2.shape == small.shape


def test_encode_jpeg_roundtrip(synthetic_image):
    img = synthetic_image(320, 200, seed=4)
    data = Annotator().encode_jpeg(img, quality=90)
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == img.shape


def test_purge_old_files(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    old = d / "a.jpg"
    old.write_bytes(b"x")
    recent = d / "b.jpg"
    recent.write_bytes(b"y")
    past = time.time() - 10 * 86400
    os.utime(old, (past, past))
    # keep_days=0（钳制为 1 天）：过期文件全删
    assert purge_old_files(d, 0) == 1
    assert not old.exists()
    assert recent.exists()
    # keep_days=1：12 小时前的文件保留
    half_day = time.time() - 12 * 3600
    os.utime(recent, (half_day, half_day))
    assert purge_old_files(d, 1) == 0
    assert recent.exists()
    # 目录不存在返回 0
    assert purge_old_files(tmp_path / "nope", 7) == 0


def test_name_mode_without_font_falls_back_to_index(synthetic_image, make_result):
    """label_mode=name 但无 font_path：构造不抛，行为与 index 模式完全一致（ASCII 标签）。"""
    img = synthetic_image(300, 200, seed=5)
    boxes = [(30, 30, 150, 150)]
    results = [make_result()]
    idx = Annotator(label_mode="index").render(img, boxes, results)
    nm = Annotator(label_mode="name", font_path="").render(img, boxes, results)
    assert np.array_equal(idx, nm)
    # index 路径画了实色标签（#1 底色）
    assert np.any(np.all(nm == PALETTE_BGR[0], axis=2))


def test_invalid_font_path_falls_back_to_index(synthetic_image, make_result):
    """font_path 指向不存在的文件 → 回落 index 模式。"""
    img = synthetic_image(300, 200, seed=8)
    boxes = [(30, 30, 150, 150)]
    results = [make_result()]
    nm = Annotator(label_mode="name", font_path="C:/no/such/font.ttf").render(
        img, boxes, results
    )
    assert np.array_equal(nm, Annotator(label_mode="index").render(img, boxes, results))
