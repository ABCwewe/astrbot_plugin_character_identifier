"""ORT 会话工厂与生命周期管理（常驻/空闲卸载、引用计数、看门狗）。"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings
from .downloader import ModelDownloader
from .registry import ModelRegistry, ModelSpec

logger = logging.getLogger(__name__)

_KINDS = ("detector", "classifier")


@dataclass
class _KindState:
    """单个模型种类（detector/classifier）的运行时状态。"""

    model: Any = None  # PersonDetector | CharacterClassifier
    fingerprint: tuple[Any, ...] | None = None
    last_used: float = 0.0
    inflight: int = 0
    loaded_at: float = 0.0
    load_seconds: float = 0.0
    session_bytes: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class ModelRuntime:
    """惰性加载检测/分类会话；空闲卸载；引用计数保护在途推理。"""

    def __init__(
        self,
        *,
        get_settings: Callable[[], Settings],
        registry: ModelRegistry,
        downloader: ModelDownloader,
        models_root: Path,
    ) -> None:
        self._get_settings = get_settings
        self._registry = registry
        self._downloader = downloader
        self._models_root = models_root
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, get_settings().max_concurrency),
            thread_name_prefix="charann-inf",
        )
        self._states: dict[str, _KindState] = {kind: _KindState() for kind in _KINDS}
        self._locks: dict[str, asyncio.Lock] = {kind: asyncio.Lock() for kind in _KINDS}
        self._idle_events: dict[str, asyncio.Event] = {
            kind: asyncio.Event() for kind in _KINDS
        }
        for event in self._idle_events.values():
            event.set()  # 初始空闲态
        self._watchdog: asyncio.Task[None] | None = None
        self._closed = False

    # ---------- 对 pipeline 的接口 ----------

    async def acquire_detector(self) -> Any:
        """确保检测模型就绪并返回 PersonDetector；在途计数 +1（配对 release_detector）。"""
        await self._ensure_kind("detector")
        self._states["detector"].inflight += 1
        self._idle_events["detector"].clear()
        self._states["detector"].last_used = time.monotonic()
        return self._states["detector"].model

    async def acquire_classifier(self) -> Any:
        """确保分类模型就绪并返回 CharacterClassifier；在途计数 +1。"""
        await self._ensure_kind("classifier")
        self._states["classifier"].inflight += 1
        self._idle_events["classifier"].clear()
        self._states["classifier"].last_used = time.monotonic()
        return self._states["classifier"].model

    async def release_detector(self) -> None:
        state = self._states["detector"]
        state.inflight = max(0, state.inflight - 1)
        if state.inflight == 0:
            self._idle_events["detector"].set()

    async def release_classifier(self) -> None:
        state = self._states["classifier"]
        state.inflight = max(0, state.inflight - 1)
        if state.inflight == 0:
            self._idle_events["classifier"].set()

    async def run_in_executor(self, func: Callable[..., Any], /, *args: Any) -> Any:
        """把阻塞函数放到推理线程池执行（ORT 推理释放 GIL，不卡事件循环）。"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, func, *args)

    # ---------- 管理接口 ----------

    async def unload(self, kind: str | None = None) -> int:
        """立即卸载（常驻配置下也可手动调用）；返回实际卸载的种类数。"""
        kinds = _KINDS if kind is None else (kind,)
        count = 0
        for k in kinds:
            async with self._locks[k]:
                state = self._states[k]
                if state.model is None:
                    continue
                if state.inflight > 0:
                    logger.warning(
                        "模型 %s 有 %d 个在途推理，跳过卸载", k, state.inflight
                    )
                    continue
                await self._drop_model(k)
                count += 1
        return count

    async def terminate(self) -> None:
        """插件卸载/停用：取消看门狗、等待在途任务、释放全部会话与线程池。"""
        self._closed = True
        if self._watchdog is not None:
            self._watchdog.cancel()
            try:
                await self._watchdog
            except asyncio.CancelledError:
                pass
            self._watchdog = None
        await self._drain_inflight()
        for k in _KINDS:
            async with self._locks[k]:
                await self._drop_model(k)
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def _drain_inflight(self, timeout: float = 10.0) -> None:
        """等待全部在途推理结束（Event 通知，带总超时）。"""
        deadline = asyncio.get_running_loop().time() + timeout
        for k in _KINDS:
            if self._states[k].inflight == 0:
                continue
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return
            try:
                await asyncio.wait_for(self._idle_events[k].wait(), timeout=remaining)
            except asyncio.TimeoutError:
                logger.warning("等待 %s 在途推理结束超时，强制继续卸载", k)

    def snapshot(self) -> dict[str, Any]:
        """/charann status 用：每 kind 的加载状态与配置摘要。"""
        settings = self._get_settings()
        result: dict[str, Any] = {
            "resident": settings.resident,
            "threads": settings.threads,
            "max_concurrency": settings.max_concurrency,
            "low_memory": settings.low_memory,
            "idle_timeout_sec": settings.idle_timeout_sec,
            "models_root": str(self._models_root),
            "kinds": {},
        }
        for k in _KINDS:
            state = self._states[k]
            key = settings.detector_key if k == "detector" else settings.classifier_key
            result["kinds"][k] = {
                "key": key,
                "loaded": state.model is not None,
                "inflight": state.inflight,
                "load_seconds": round(state.load_seconds, 3),
                "session_bytes": state.session_bytes,
                "idle_seconds": (
                    round(time.monotonic() - state.last_used, 1)
                    if state.last_used and state.model is not None
                    else None
                ),
                **state.extra,
            }
        return result

    # ---------- 内部实现 ----------

    def _fingerprint(self, kind: str, settings: Settings) -> tuple[Any, ...]:
        if kind == "detector":
            return (
                settings.detector_key,
                settings.threads,
                settings.low_memory,
                settings.det_imgsz,
                settings.det_conf,
                settings.nms_iou,
                settings.max_persons,
                settings.min_box_px,
            )
        return (
            settings.classifier_key,
            settings.threads,
            settings.low_memory,
            settings.preprocess_backend,
        )

    def _spec(self, kind: str, settings: Settings) -> ModelSpec:
        if kind == "detector":
            return self._registry.detector(settings.detector_key)
        return self._registry.classifier(settings.classifier_key)

    async def _ensure_kind(self, kind: str) -> None:
        """加载/重建会话；同一把锁串行化加载与卸载。"""
        if self._closed:
            raise RuntimeError("runtime 已终止")
        self._ensure_watchdog()
        settings = self._get_settings()
        fingerprint = self._fingerprint(kind, settings)
        async with self._locks[kind]:
            state = self._states[kind]
            if state.model is not None and state.fingerprint == fingerprint:
                return
            if state.model is not None:
                if state.inflight > 0:
                    logger.warning("模型 %s 配置变更但有在途推理，本次沿用旧会话", kind)
                    return
                logger.info("配置变更，重建 %s 会话", kind)
                await self._drop_model(kind)
            spec = self._spec(kind, settings)
            paths = await self._downloader.ensure(spec)
            t0 = time.monotonic()
            model, extra = await self.run_in_executor(
                self._build_model, kind, spec, paths, settings
            )
            state.model = model
            state.fingerprint = fingerprint
            state.loaded_at = time.monotonic()
            state.load_seconds = state.loaded_at - t0
            state.last_used = state.loaded_at
            state.extra = extra
            logger.info(
                "%s 会话加载完成（%.2fs，key=%s）", kind, state.load_seconds, spec.key
            )

    def _build_model(
        self,
        kind: str,
        spec: ModelSpec,
        paths: dict[str, Path],
        settings: Settings,
    ) -> tuple[Any, dict[str, Any]]:
        """在工作线程构建 ORT 会话与模型对象（阻塞）。"""
        import numpy as np

        session = self._build_session(paths["model"], settings)
        if kind == "detector":
            from .detector import PersonDetector

            model = PersonDetector(
                session,
                imgsz=settings.det_imgsz,
                conf=settings.det_conf,
                nms_iou=settings.nms_iou,
                max_persons=settings.max_persons,
                min_box_px=settings.min_box_px,
            )
            return model, {"input_size": model.input_size}
        from .classifier import CharacterClassifier

        with np.load(paths["prototypes"], allow_pickle=False) as z:
            protos = np.asarray(z["protos"], dtype=np.float32)
            class_names = [str(c) for c in z["class_names"]]
            proto_class_idx = np.asarray(z["proto_class_idx"])
            tau = float(z["tau"])
            p_min = float(z["p_min"])
            version = int(z["version"])
        model = CharacterClassifier(
            session,
            protos=protos,
            class_names=class_names,
            proto_class_idx=proto_class_idx,
            tau=tau,
            p_min=p_min,
            version=version,
            backend=settings.preprocess_backend,
        )
        return model, {
            "input_size": model.input_size,
            "dynamic_batch": model.dynamic_batch,
            "npz_version": version,
        }

    def _build_session(self, model_path: Path, settings: Settings) -> Any:
        """统一 SessionOptions 的 CPU 会话工厂（惰性 import onnxruntime）。"""
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = settings.threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if settings.low_memory:
            so.enable_cpu_mem_arena = False
            so.enable_mem_pattern = False
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session = ort.InferenceSession(
            str(model_path), so, providers=["CPUExecutionProvider"]
        )
        return session

    async def _drop_model(self, kind: str) -> None:
        """释放一个 kind 的会话（持有锁时调用）。"""
        state = self._states[kind]
        state.model = None
        state.fingerprint = None
        state.extra = {}
        await self.run_in_executor(gc.collect)

    def _ensure_watchdog(self) -> None:
        """非常驻模式下确保看门狗在跑。"""
        settings = self._get_settings()
        if settings.resident or self._watchdog is not None or self._closed:
            return
        self._watchdog = asyncio.get_running_loop().create_task(self._watchdog_loop())

    async def _watchdog_loop(self) -> None:
        """周期检查空闲超时并卸载；每 kind 独立计时。"""
        try:
            while True:
                settings = self._get_settings()
                if settings.resident:
                    return
                interval = min(settings.idle_timeout_sec / 4, 30.0)
                await asyncio.sleep(max(5.0, interval))
                settings = self._get_settings()
                if settings.resident or self._closed:
                    return
                now = time.monotonic()
                for k in _KINDS:
                    state = self._states[k]
                    if (
                        state.model is not None
                        and state.inflight == 0
                        and now - state.last_used >= settings.idle_timeout_sec
                    ):
                        async with self._locks[k]:
                            if (
                                self._states[k].model is not None
                                and self._states[k].inflight == 0
                            ):
                                logger.info(
                                    "模型 %s 空闲超过 %ds，卸载释放内存",
                                    k,
                                    settings.idle_timeout_sec,
                                )
                                await self._drop_model(k)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("看门狗异常退出")
