"""图片采集（子任务 D）：从 req.image_urls 的多种形态取图、解码、体积/尺寸限制。

只负责「取图 → 解码 → 整体缩放」，不做并发控制（信号量由 pipeline 统一管理）。
单张失败（网络/超时/超限/解码失败）一律跳过并记 debug 日志，绝不影响其他图片。
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote

import aiohttp
import numpy as np

if TYPE_CHECKING:
    from .config import Settings

logger = logging.getLogger(__name__)

# 单张网络图片请求总超时（秒）
_HTTP_TIMEOUT_SEC: float = 10.0
# 单张网络图片体积上限（字节，10MB）
_MAX_BYTES: int = 10 * 1024 * 1024
# 单张图片面积上限（像素²，4096×4096），超过直接跳过
_MAX_PIXELS: int = 4096 * 4096
# 网络分块读取大小
_CHUNK_SIZE: int = 64 * 1024


class ImageSourceError(RuntimeError):
    """单张图片不可用（网络失败/非图片/超限/解码失败等）；调用方跳过该张。"""


@dataclass
class SourceImage:
    """一张成功解码的图片（BGR）。"""

    index: int  # 在 req.image_urls 中的原始下标
    sha1: str  # 原始字节 sha1 hex
    size_bytes: int  # 原始字节数
    bgr: np.ndarray  # 解码后 BGR 图（可能已整体缩放）
    scaled: bool  # 是否被 det_pre_max_side 整体缩小过


class ImageCollector:
    """从 req.image_urls 各形态取图并解码。"""

    def __init__(self, get_settings: Callable[[], Settings]) -> None:
        self._get_settings = get_settings

    async def collect(self, image_urls: Sequence[str]) -> list[SourceImage]:
        """取前 max_images 张图解码；失败/超限的图跳过，返回可用列表（可为空）。"""
        if not image_urls:
            return []
        settings = self._get_settings()
        limit = max(1, settings.max_images)
        urls = list(image_urls)[:limit]
        images: list[SourceImage] = []
        timeout = aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_SEC)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for index, url in enumerate(urls):
                try:
                    source = await self._collect_one(session, index, url, settings)
                except Exception:
                    logger.debug("跳过图片 %d（%r）", index, url, exc_info=True)
                    continue
                if source is not None:
                    images.append(source)
        return images

    async def _collect_one(
        self,
        session: aiohttp.ClientSession,
        index: int,
        url: str,
        settings: Settings,
    ) -> SourceImage | None:
        """单张图：取字节 → 解码 → 面积限制 → 最长边缩放；不可用返回 None。"""
        raw = await self._fetch_bytes(session, url)
        if raw is None:
            return None
        bgr = self._decode(raw)
        if bgr is None:
            logger.debug("图片 %d 解码失败（%r）", index, url)
            return None
        h, w = bgr.shape[:2]
        if h * w > _MAX_PIXELS:
            logger.debug("图片 %d 面积 %dx%d 超过上限，跳过", index, w, h)
            return None
        scaled = False
        max_side = max(h, w)
        det_pre = settings.det_pre_max_side
        if max_side > det_pre:
            import cv2

            scale = det_pre / max_side
            nw = max(1, round(w * scale))
            nh = max(1, round(h * scale))
            bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
            scaled = True
        return SourceImage(
            index=index,
            sha1=hashlib.sha1(raw).hexdigest(),
            size_bytes=len(raw),
            bgr=bgr,
            scaled=scaled,
        )

    async def _fetch_bytes(
        self, session: aiohttp.ClientSession, url: str
    ) -> bytes | None:
        """按 URL 形态取原始字节；不可用的形态返回 None（debug 日志）。"""
        stripped = url.strip()
        if not stripped:
            return None
        low = stripped.lower()
        if low.startswith(("http://", "https://")):
            return await self._fetch_http(session, stripped)
        if low.startswith("base64://"):
            return self._decode_base64(stripped[len("base64://") :])
        if low.startswith("data:image/"):
            return self._decode_data_uri(stripped)
        return self._read_local(stripped)

    async def _fetch_http(self, session: aiohttp.ClientSession, url: str) -> bytes:
        """GET 下载网络图片；非 image/*、SVG、超 10MB 或网络错误抛 ImageSourceError。"""
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise ImageSourceError(f"HTTP {resp.status}")
                content_type = resp.headers.get("Content-Type", "").lower()
                if not content_type.startswith("image/"):
                    raise ImageSourceError(f"非图片 Content-Type: {content_type!r}")
                if content_type.startswith("image/svg"):
                    raise ImageSourceError("SVG 图片不支持")
                chunks: list[bytes] = []
                total = 0
                async for chunk in resp.content.iter_chunked(_CHUNK_SIZE):
                    total += len(chunk)
                    if total > _MAX_BYTES:
                        raise ImageSourceError(f"超过 10MB 上限（{total} 字节）")
                    chunks.append(chunk)
                return b"".join(chunks)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise ImageSourceError(f"下载失败: {exc}") from exc

    @staticmethod
    def _decode_data_uri(uri: str) -> bytes:
        """解析 data:image/...;base64, 数据 URI；非 base64 形态抛 ImageSourceError。"""
        header, sep, payload = uri.partition(",")
        if not sep or ";base64" not in header:
            raise ImageSourceError("不支持的数据 URI（非 base64）")
        return ImageCollector._decode_base64(payload)

    @staticmethod
    def _decode_base64(text: str) -> bytes:
        """解码 base64 文本（兼容 URL-safe 字母表）；失败抛 ImageSourceError。"""
        try:
            return base64.b64decode(text)
        except (ValueError, binascii.Error):
            try:
                return base64.b64decode(text, altchars=b"-_")
            except (ValueError, binascii.Error) as exc:
                raise ImageSourceError(f"base64 解码失败: {exc}") from exc

    @staticmethod
    def _read_local(url: str) -> bytes | None:
        """读取本地路径（file:// 剥前缀）；不存在/读取失败返回 None。"""
        path_str = url
        if path_str.startswith("file://"):
            path_str = path_str[len("file://") :]
            # file:///C:/... 在 Windows 上剥掉多余的斜杠
            if len(path_str) >= 3 and path_str[0] == "/" and path_str[2] == ":":
                path_str = path_str[1:]
        path_str = unquote(path_str)
        path = Path(path_str)
        if not path.is_file():
            logger.debug("本地图片不存在：%s", path_str)
            return None
        try:
            return path.read_bytes()
        except OSError:
            logger.debug("本地图片读取失败：%s", path_str, exc_info=True)
            return None

    @staticmethod
    def _decode(raw: bytes) -> np.ndarray | None:
        """cv2 解码为 BGR（动图取首帧，EXIF 方向已应用）；失败返回 None。"""
        import cv2

        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        return None if img is None else img
