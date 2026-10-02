"""astrbot_plugin_character_identifier 插件入口。

on_llm_request 钩子：图片 → 检测 → 鸣潮角色识别 → 标注 → 注入 ProviderRequest。
任何异常/超时/模型未就绪：原样放行 LLM 请求，绝不丢消息。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from .core.config import load_settings
from .core.downloader import DownloadError, ModelDownloader
from .core.image_io import ImageCollector
from .core.injector import apply as apply_injection
from .core.pipeline import Pipeline
from .core.registry import load_registry
from .core.runtime import ModelRuntime

PLUGIN_NAME = "astrbot_plugin_character_identifier"
REGISTRY_PATH = Path(__file__).resolve().parent / "data" / "registry.json"


class CharAnnotatePlugin(Star):
    """鸣潮角色识别标注插件。"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self._config = config
        self._settings = load_settings(config)
        self._registry = load_registry(REGISTRY_PATH)
        data_root = Path(get_astrbot_data_path()) / "plugin_data" / PLUGIN_NAME
        self._downloader = ModelDownloader(
            get_settings=lambda: self._settings,
            models_root=data_root / "models",
            registry=self._registry,
        )
        self._runtime = ModelRuntime(
            get_settings=lambda: self._settings,
            registry=self._registry,
            downloader=self._downloader,
            models_root=data_root / "models",
        )
        self._collector = ImageCollector(get_settings=lambda: self._settings)
        self._pipeline = Pipeline(
            get_settings=lambda: self._settings,
            registry=self._registry,
            runtime=self._runtime,
            downloader=self._downloader,
            cache_dir=data_root / "cache",
        )
        self._infer_semaphore = asyncio.Semaphore(
            max(1, self._settings.max_concurrency)
        )
        self._download_task: asyncio.Task[None] | None = None

    # ---------- LLM 请求钩子 ----------

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """LLM 请求前：识别并标注图片；失败一律原样放行。"""
        try:
            settings = self._settings
            if not settings.enabled:
                return
            umo = event.unified_msg_origin
            if settings.session_whitelist and umo not in settings.session_whitelist:
                return
            if umo in settings.session_blacklist:
                return
            urls = list(req.image_urls or [])
            if not urls:
                return
            images = await self._collector.collect(urls)
            if not images:
                return
            async with self._infer_semaphore:
                outcomes = await asyncio.wait_for(
                    self._pipeline.run(images), timeout=settings.infer_timeout
                )
            if outcomes:
                apply_injection(req, outcomes, lambda: self._settings)
                n = sum(1 for o in outcomes if o.annotated_path is not None)
                logger.info(
                    "角色识别完成：%d/%d 张图注入（耗时受 %ds 超时约束）",
                    n,
                    len(outcomes),
                    settings.infer_timeout,
                )
        except asyncio.TimeoutError:
            logger.warning(
                "角色识别超时（%ss），本次请求原样放行", self._settings.infer_timeout
            )
        except Exception:
            logger.exception("角色识别失败，本次请求原样放行")

    # ---------- 管理指令 ----------

    @filter.command_group("charann")
    def charann(self, event: AstrMessageEvent):
        """角色识别插件管理指令组。"""

    @charann.command("status")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def charann_status(self, event: AstrMessageEvent):
        """查看模型就绪/加载状态与缓存占用。"""
        snap = self._runtime.snapshot()
        settings = self._settings
        cache_dir = self._pipeline.cache_dir
        lines = [
            f"启用: {settings.enabled}；常驻: {settings.resident}"
            f"（空闲卸载 {settings.idle_timeout_sec}s）；线程: {settings.threads}×{settings.max_concurrency}",
            f"检测模型: {snap['kinds']['detector']['key']}（loaded={snap['kinds']['detector']['loaded']}）",
            f"分类模型: {snap['kinds']['classifier']['key']}（loaded={snap['kinds']['classifier']['loaded']}）",
        ]
        for kind in ("detector", "classifier"):
            try:
                spec = (
                    self._registry.detector(settings.detector_key)
                    if kind == "detector"
                    else self._registry.classifier(settings.classifier_key)
                )
                st = self._downloader.status(spec)
                lines.append(
                    f"{kind} 就绪: {st.ready}，占用 {st.total_bytes / 1e6:.1f} MB"
                )
            except Exception as exc:
                lines.append(f"{kind} 状态读取失败: {exc}")
        total = 0
        if cache_dir.exists():
            total = sum(p.stat().st_size for p in cache_dir.rglob("*") if p.is_file())
        lines.append(
            f"标注缓存: {total / 1e6:.1f} MB（保留 {settings.cache_keep_days} 天）"
        )
        yield event.plain_result("\n".join(lines))

    @charann.command("download")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def charann_download(self, event: AstrMessageEvent):
        """后台预下载当前所选检测/分类模型。"""
        if self._download_task and not self._download_task.done():
            yield event.plain_result("已有下载任务在进行中")
            return
        self._download_task = asyncio.create_task(self._download_models())
        yield event.plain_result("已开始后台下载当前所选模型，完成后见日志")

    @charann.command("unload")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def charann_unload(self, event: AstrMessageEvent):
        """立即卸载模型会话释放内存。"""
        n = await self._runtime.unload()
        yield event.plain_result(f"已卸载 {n} 个会话")

    @charann.command("clear_cache")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def charann_clear_cache(self, event: AstrMessageEvent):
        """清理标注图缓存。"""
        removed = await self._pipeline.purge_cache(keep_days=0)
        yield event.plain_result(f"已清理 {removed} 个缓存文件")

    async def _download_models(self) -> None:
        """顺序预下载检测与分类模型（含 display_names）。"""
        settings = self._settings
        try:
            det_spec = self._registry.detector(settings.detector_key)
            await self._downloader.ensure(det_spec)
            logger.info("检测模型 %s 就绪", det_spec.key)
        except DownloadError as exc:
            logger.warning("检测模型下载失败: %s", exc)
            return
        try:
            cls_spec = self._registry.classifier(settings.classifier_key)
            await self._downloader.ensure(cls_spec)
            logger.info("分类模型 %s 就绪", cls_spec.key)
        except DownloadError as exc:
            logger.warning("分类模型下载失败: %s", exc)

    # ---------- 生命周期 ----------

    async def terminate(self):
        """卸载/停用：取消后台任务并释放全部会话。"""
        if self._download_task is not None:
            self._download_task.cancel()
            try:
                await self._download_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("后台下载任务清理异常", exc_info=True)
            self._download_task = None
        await self._runtime.terminate()
