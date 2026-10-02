"""标注渲染（子任务 D）：调色板、画框、标签（编号/中文名）、缩放、JPEG 编码。

cv2 惰性导入（模块顶层只依赖标准库与 numpy）；PIL（ImageFont/Image/ImageDraw）
仅在 label_mode=name 的渲染路径惰性导入。所有绘制不修改输入图。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

if TYPE_CHECKING:
    from .classifier import CharResult

logger = logging.getLogger(__name__)

# BGR 10 色高对比调色板：避开红绿难分组合（色觉障碍友好），按编号循环
PALETTE_BGR: Final[tuple[tuple[int, int, int], ...]] = (
    (0, 140, 255),  # 橙色
    (255, 200, 0),  # 青色
    (255, 100, 0),  # 蓝色
    (0, 230, 255),  # 黄色
    (240, 32, 160),  # 紫色
    (190, 210, 0),  # 绿松石
    (190, 120, 255),  # 粉色
    (255, 255, 255),  # 白色
    (160, 255, 0),  # 亮绿蓝
    (43, 90, 139),  # 棕色
)
GREY_BGR: Final[tuple[int, int, int]] = (128, 128, 128)
GREY_NAME: Final[str] = "灰色"  # 未识别角色的说明文字色名
# 与 PALETTE_BGR 一一对应的中文色名（说明文字用）
COLOR_NAMES_ZH: Final[tuple[str, ...]] = (
    "橙色",
    "青色",
    "蓝色",
    "黄色",
    "紫色",
    "绿松石",
    "粉色",
    "白色",
    "亮绿蓝",
    "棕色",
)

_SUFFIX = "_(wuthering_waves)"  # 类名后缀，展示时去除
_LABEL_PAD = 3  # 标签内边距（像素）


def _text_color_for(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """按背景亮度选对比色：亮底黑字、暗底白字。"""
    b, g, r = bgr
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 150 else (255, 255, 255)


class Annotator:
    """在图上绘制检测框与标签，并编码 JPEG。"""

    def __init__(
        self,
        *,
        label_mode: str = "index",
        font_path: str = "",
        out_max_side: int = 1280,
    ) -> None:
        self._label_mode = label_mode
        self._font_path = font_path
        self._out_max_side = max(512, int(out_max_side))
        self._font_cache: dict[int, Any] = {}
        # name 模式：构造时预加载一次字体验证可用性，失败则整体回落 index 模式
        self._font_ok = False
        if label_mode == "name":
            if not font_path:
                logger.debug("label_mode=name 但未配置 font_path，回落 index 模式")
            else:
                try:
                    self._get_font(16)
                    self._font_ok = True
                except Exception:
                    logger.debug(
                        "字体加载失败（%s），回落 index 模式", font_path, exc_info=True
                    )

    def render(
        self,
        image_bgr: np.ndarray,
        boxes: Sequence[tuple[float, float, float, float]],
        results: Sequence[CharResult | None],
        display_names: Mapping[str, str] | None = None,
    ) -> np.ndarray:
        """绘制全部框与标签，返回新 BGR 图（不改输入）；长边超 out_max_side 整体缩放。"""
        import cv2

        h, w = image_bgr.shape[:2]
        line_w = max(2, round(min(h, w) / 300))
        if self._label_mode == "name" and self._font_ok:
            from PIL import Image, ImageDraw

            pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            for i, box in enumerate(boxes):
                color = self._color_for(i, results)
                text = self._label_text(i, results, display_names)
                self._draw_name_label(pil_img, draw, box, text, color, line_w)
            annotated = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        else:
            annotated = image_bgr.copy()
            for i, box in enumerate(boxes):
                color = self._color_for(i, results)
                self._draw_index_label(annotated, box, i, color, line_w)
        max_side = max(annotated.shape[0], annotated.shape[1])
        if max_side > self._out_max_side:
            scale = self._out_max_side / max_side
            nw = max(1, round(annotated.shape[1] * scale))
            nh = max(1, round(annotated.shape[0] * scale))
            annotated = cv2.resize(annotated, (nw, nh), interpolation=cv2.INTER_AREA)
        return annotated

    @staticmethod
    def _color_for(
        i: int, results: Sequence[CharResult | None]
    ) -> tuple[int, int, int]:
        """未知/缺失结果为灰色；已识别按编号循环取调色板色。"""
        result = results[i] if i < len(results) else None
        if result is None or result.unknown:
            return GREY_BGR
        return PALETTE_BGR[i % len(PALETTE_BGR)]

    def _label_text(
        self,
        i: int,
        results: Sequence[CharResult | None],
        display_names: Mapping[str, str] | None,
    ) -> str:
        """name 模式标签文本：映射命中→中文名；未命中→类名清洗；unknown→未识别。"""
        result = results[i] if i < len(results) else None
        if result is None or result.unknown or not result.pred:
            return "未识别"
        if display_names:
            name = display_names.get(result.pred)
            if name:
                return name
        return self._fallback_name(result.pred)

    @staticmethod
    def _fallback_name(class_name: str) -> str:
        """类名未命中映射时的展示名：去 _(wuthering_waves) 后缀、下划线换空格。"""
        name = class_name
        if name.endswith(_SUFFIX):
            name = name[: -len(_SUFFIX)]
        return name.replace("_", " ")

    def _get_font(self, size: int) -> Any:
        """按字号惰性加载中文字体并缓存（PIL 仅此处导入）。"""
        if size not in self._font_cache:
            from PIL import ImageFont

            self._font_cache[size] = ImageFont.truetype(self._font_path, size=size)
        return self._font_cache[size]

    @staticmethod
    def _draw_index_label(
        img: np.ndarray,
        box: tuple[float, float, float, float],
        index: int,
        color: tuple[int, int, int],
        line_w: int,
    ) -> None:
        """cv2 绘制：框 + "#N" 标签（实色底+对比字），框上沿内侧、越界移入框内。"""
        import cv2

        h_img, w_img = img.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_w)
        text = f"#{index + 1}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.6, line_w * 0.5)
        thickness = max(1, line_w - 1)
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        pad = _LABEL_PAD
        rect_w = tw + 2 * pad
        rect_h = th + baseline + 2 * pad
        lx = min(max(x1, 0), max(0, w_img - rect_w))
        ry = y1 - rect_h  # 底边贴框上沿（框内）
        if ry < 0:
            ry = y1  # 越界移入框内
        ry = min(max(ry, 0), max(0, h_img - rect_h))
        cv2.rectangle(img, (lx, ry), (lx + rect_w, ry + rect_h), color, -1)
        cv2.putText(
            img,
            text,
            (lx + pad, ry + pad + th),
            font,
            scale,
            _text_color_for(color),
            thickness,
            cv2.LINE_AA,
        )

    def _draw_name_label(
        self,
        pil_img: Any,
        draw: Any,
        box: tuple[float, float, float, float],
        text: str,
        color: tuple[int, int, int],
        line_w: int,
    ) -> None:
        """PIL 绘制：框 + 中文名标签（实色底+对比字），框上沿内侧、越界移入框内。"""
        w_img, h_img = pil_img.size
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        color_rgb = (color[2], color[1], color[0])
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=line_w)
        font = self._get_font(max(14, round(line_w * 4)))
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pad = _LABEL_PAD
        rect_w = tw + 2 * pad
        rect_h = th + 2 * pad
        lx = min(max(x1, 0), max(0, w_img - rect_w))
        ry = y1 - rect_h  # 底边贴框上沿（框内）
        if ry < 0:
            ry = y1  # 越界移入框内
        ry = min(max(ry, 0), max(0, h_img - rect_h))
        draw.rectangle([lx, ry, lx + rect_w, ry + rect_h], fill=color_rgb)
        draw.text(
            (lx + pad - bbox[0], ry + pad - bbox[1]),
            text,
            font=font,
            fill=_text_color_for(color),
        )

    def encode_jpeg(self, image_bgr: np.ndarray, quality: int = 85) -> bytes:
        """JPEG 编码；失败抛 RuntimeError。"""
        import cv2

        ok, buf = cv2.imencode(
            ".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
        )
        if not ok:
            raise RuntimeError("JPEG 编码失败")
        return buf.tobytes()


def purge_old_files(cache_dir: Path, keep_days: int) -> int:
    """删除 mtime 早于 keep_days 天的缓存文件，返回删除数；目录不存在返回 0。"""
    if not cache_dir.is_dir():
        return 0
    cutoff = time.time() - max(1, int(keep_days)) * 86400
    removed = 0
    try:
        entries = list(cache_dir.iterdir())
    except OSError:
        return 0
    for entry in entries:
        try:
            if entry.is_file() and entry.stat().st_mtime < cutoff:
                entry.unlink()
                removed += 1
        except OSError:
            logger.debug("清理缓存文件失败：%s", entry, exc_info=True)
    return removed
