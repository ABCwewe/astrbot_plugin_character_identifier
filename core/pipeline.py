"""处理流水线：解码图 → 检测 → 识别 → 标注 → 缓存，产出 ImageOutcome。"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .annotator import COLOR_NAMES_ZH, GREY_NAME, Annotator, purge_old_files
from .config import Settings
from .downloader import ModelDownloader
from .registry import ModelRegistry, load_display_names
from .runtime import ModelRuntime

logger = logging.getLogger(__name__)

_LRU_CAP = 64
_CROP_PAD_RATIO = 0.05  # 检测框外扩比例（模型针对全身立绘训练）


@dataclass(frozen=True)
class PersonLine:
    """单个检测框的文本行结果（编号与标注图一致）。"""

    index: int
    color_name: str
    display_name: str | None
    head_conf: float | None
    top5: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class ImageOutcome:
    """单张图的处理结果；annotated_path 为 None 表示该图不注入。"""

    source_index: int
    sha1: str
    annotated_path: Path | None
    persons: list[PersonLine]
    scope_note: str


def _fallback_name(class_name: str) -> str:
    """映射表未命中：去 `_(wuthering_waves)` 后缀、下划线换空格，不自行翻译。"""
    name = class_name
    if name.endswith("_(wuthering_waves)"):
        name = name[: -len("_(wuthering_waves)")]
    return name.replace("_", " ")


class Pipeline:
    """串联检测/识别/标注；LRU 结果缓存避免同图重复推理。"""

    def __init__(
        self,
        *,
        get_settings: Callable[[], Settings],
        registry: ModelRegistry,
        runtime: ModelRuntime,
        downloader: ModelDownloader,
        cache_dir: Path,
    ) -> None:
        self._get_settings = get_settings
        self._registry = registry
        self._runtime = runtime
        self._downloader = downloader
        self._cache_dir = cache_dir
        self._lru: OrderedDict[tuple, ImageOutcome] = OrderedDict()
        self._names_cache: tuple[Path, float, dict[str, str] | None] | None = None

    @property
    def cache_dir(self) -> Path:
        """标注图缓存目录。"""
        return self._cache_dir

    # ---------- 主入口 ----------

    async def run(self, images: Sequence[object]) -> list[ImageOutcome]:
        """逐张处理；images 为 image_io.SourceImage 序列（鸭子类型避免环依赖）。"""
        settings = self._get_settings()
        if not images:
            return []
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        outcomes: list[ImageOutcome] = []
        for image in images:
            outcome = await self._run_one(image, settings)
            outcomes.append(outcome)
        return outcomes

    async def purge_cache(self, keep_days: int) -> int:
        """清理过期标注图；返回删除数。"""
        return await self._runtime.run_in_executor(
            purge_old_files, self._cache_dir, keep_days
        )

    # ---------- 单图处理 ----------

    async def _run_one(self, image: object, settings: Settings) -> ImageOutcome:
        key = await self._cache_key(image, settings)
        cached = self._lru.get(key)
        if cached is not None:
            self._lru.move_to_end(key)
            return cached
        outcome = await self._process(image, settings)
        self._lru[key] = outcome
        while len(self._lru) > _LRU_CAP:
            self._lru.popitem(last=False)
        return outcome

    async def _process(self, image: object, settings: Settings) -> ImageOutcome:
        empty = ImageOutcome(
            source_index=image.index,
            sha1=image.sha1,
            annotated_path=None,
            persons=[],
            scope_note=self._scope_note(settings),
        )
        try:
            detector = await self._runtime.acquire_detector()
        except Exception:
            logger.exception("检测模型不可用，跳过该图")
            return empty
        try:
            boxes = await self._runtime.run_in_executor(detector.detect, image.bgr)
            if not boxes:
                return empty
            classifier = await self._runtime.acquire_classifier()
            try:
                crops = self._crop_boxes(image.bgr, boxes)
                results = await self._runtime.run_in_executor(
                    classifier.predict_batch, crops
                )
            finally:
                await self._runtime.release_classifier()
            if all(r is None or r.unknown for r in results):
                return empty
            outcome = await self._render_outcome(image, settings, boxes, results)
            return outcome
        finally:
            await self._runtime.release_detector()

    # ---------- 渲染与缓存 ----------

    async def _render_outcome(self, image: object, settings: Settings, boxes, results):
        annotator = Annotator(
            label_mode=settings.label_mode,
            font_path=settings.font_path,
            out_max_side=settings.out_max_side,
        )
        display_names = await self._display_names(settings)
        rendered = await self._runtime.run_in_executor(
            annotator.render, image.bgr, [d.xyxy for d in boxes], results, display_names
        )
        data = await self._runtime.run_in_executor(annotator.encode_jpeg, rendered)
        path = self._cache_dir / f"{image.sha1[:16]}.jpg"
        await self._runtime.run_in_executor(path.write_bytes, data)

        persons: list[PersonLine] = []
        for i, result in enumerate(results):
            number = i + 1
            if result is None or result.unknown:
                top5: tuple[tuple[str, float], ...] = ()
                if settings.show_top5_for_unknown and result is not None:
                    top5 = tuple(
                        (
                            display_names.get(name, _fallback_name(name))
                            if display_names
                            else _fallback_name(name),
                            round(score, 4),
                        )
                        for name, score in result.top5
                    )
                persons.append(
                    PersonLine(
                        index=number,
                        color_name=GREY_NAME,
                        display_name=None,
                        head_conf=None,
                        top5=top5,
                    )
                )
                continue
            if display_names and result.pred in display_names:
                name = display_names[result.pred]
            else:
                name = _fallback_name(result.pred or "")
            persons.append(
                PersonLine(
                    index=number,
                    color_name=COLOR_NAMES_ZH[i % len(COLOR_NAMES_ZH)],
                    display_name=name,
                    head_conf=round(float(result.head_conf), 2),
                )
            )
        scope_note = self._scope_note(settings)
        return ImageOutcome(
            source_index=image.index,
            sha1=image.sha1,
            annotated_path=path,
            persons=persons,
            scope_note=scope_note,
        )

    async def _display_names(self, settings: Settings) -> dict[str, str] | None:
        """加载展示名映射（记忆化：路径+mtime 未变则复用）。"""
        spec = self._registry.classifier(settings.classifier_key)
        try:
            paths = await self._downloader.ensure(spec)
        except Exception:
            logger.debug("分类器模型未就绪，展示名映射暂缺")
            return None
        path = paths.get("display_names")
        if path is None:
            return None
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        if (
            self._names_cache
            and self._names_cache[0] == path
            and self._names_cache[1] == mtime
        ):
            return self._names_cache[2]
        try:
            names = await self._runtime.run_in_executor(load_display_names, path)
        except Exception:
            logger.debug("展示名映射加载失败", exc_info=True)
            names = None
        self._names_cache = (path, mtime, names)
        return names

    # ---------- 缓存键 ----------

    async def _cache_key(self, image: object, settings: Settings) -> tuple:
        spec = self._registry.classifier(settings.classifier_key)
        status = self._downloader.status(spec)
        revision = status.revision or "unknown"
        version = -1
        names_sha = ""
        for f in status.files:
            if f.role == "display_names":
                names_sha = f.sha256 or ""
        snap = self._runtime.snapshot()["kinds"]["classifier"]
        if snap["loaded"]:
            version = snap.get("npz_version", -1)
        params = (
            settings.det_imgsz,
            settings.det_conf,
            settings.nms_iou,
            settings.max_persons,
            settings.min_box_px,
            settings.preprocess_backend,
            settings.label_mode,
            settings.out_max_side,
            settings.show_top5_for_unknown,
        )
        params_hash = hashlib.sha1(repr(params).encode()).hexdigest()[:8]
        if settings.label_mode == "name":
            return (
                image.sha1,
                spec.key,
                revision,
                version,
                params_hash,
                names_sha[:12],
            )
        return (image.sha1, spec.key, revision, version, params_hash)

    def _scope_note(self, settings: Settings) -> str:
        try:
            spec = self._registry.classifier(settings.classifier_key)
        except Exception:
            return ""
        return spec.scope_note

    # ---------- 裁剪 ----------

    @staticmethod
    def _crop_boxes(image_bgr, boxes) -> list:
        """按检测框外扩 5% 从原图裁剪（截到图内，最小 1px）。"""
        import numpy as np

        h, w = image_bgr.shape[:2]
        crops = []
        for det in boxes:
            x1, y1, x2, y2 = det.xyxy
            pw = (x2 - x1) * _CROP_PAD_RATIO
            ph = (y2 - y1) * _CROP_PAD_RATIO
            left = max(0, int(np.floor(x1 - pw)))
            top = max(0, int(np.floor(y1 - ph)))
            right = min(w, int(np.ceil(x2 + pw)))
            bottom = min(h, int(np.ceil(y2 + ph)))
            if right - left < 1 or bottom - top < 1:
                left, top = max(0, int(x1)), max(0, int(y1))
                right, bottom = (
                    min(w, max(left + 1, int(x2))),
                    min(h, max(top + 1, int(y2))),
                )
            crops.append(np.ascontiguousarray(image_bgr[top:bottom, left:right]))
        return crops
