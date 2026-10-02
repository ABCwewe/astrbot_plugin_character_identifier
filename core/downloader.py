"""模型下载：HuggingFace 异步下载、main sha 解析、.part 原子落盘与本地校验。

行为要点（AGENTS.md §6.2 / §6.4.1 与契约 core/downloader.py）：
- ref=="main" 时先 GET `{endpoint}/api/models/{repo_id}/revision/main` 解析 commit sha，
  失败（如镜像不支持 api/models 接口）回落直接用 ref；解析结果按 (spec.key, endpoint)
  缓存到实例 dict，避免每次请求重复解析。
- 每次网络请求新建 aiohttp.ClientSession（简单、低频）；大文件分块写 `<name>.part`，
  非空与 sha256（spec 提供时）校验通过后 os.replace 原子改名；不实现断点续传。
- 每个 key 一把 asyncio.Lock；单文件失败指数退避重试 3 次（1s/2s/4s）。
- 成对语义：spec.files 全部角色就绪才算就绪；本轮任一失败 → 删除本轮新建文件
  （已存在的旧校验文件不动）并抛 DownloadError。
- display_names 随分类器以同一 revision 下载（伪 role "display_names"），不参与成对
  校验：下载失败仅 logger.warning 并从返回 dict 中省略，不影响模型就绪。
- offline_mode=True 时仅检查本地，缺失即抛 DownloadError。
- hf_token 只进 Authorization 请求头，绝不写日志。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import aiohttp

from .config import Settings
from .registry import ModelRegistry, ModelSpec

logger = logging.getLogger(__name__)

# 下载分块大小（字节）
_CHUNK_SIZE = 1 << 18
# 单文件总超时（秒）
_SINGLE_FILE_TIMEOUT = 60
# main sha 解析接口超时（秒），镜像不可用时避免长时间阻塞热路径
_SHA_API_TIMEOUT = 10
# 失败重试次数：指数退避 1s/2s/4s，共 3 次重试
_MAX_RETRIES = 3
_RETRY_BACKOFF = (1, 2, 4)
# 校验缓存条目上限，超出即整体清空，防止内存膨胀
_VERIFY_CACHE_MAX = 512


class DownloadError(RuntimeError):
    """模型下载或本地校验失败。"""


@dataclass(frozen=True)
class ModelFileStatus:
    """单个文件（角色或 display_names 伪角色）的本地就绪状态。"""

    role: str
    path: Path | None  # 本地已就绪路径；未就绪 None
    sha256: str | None  # 已校验值（spec 提供期望值时）


@dataclass(frozen=True)
class ModelStatus:
    """一个模型的本地就绪状态（纯本地、不触网）。"""

    spec_key: str
    ready: bool
    files: list[ModelFileStatus]
    revision: str | None  # 落盘目录使用的 sha 或 tag
    total_bytes: int


class ModelDownloader:
    """HF 模型下载器：sha 解析、原子落盘、并发锁与重试。

    :param get_settings: 返回最新 Settings 的可调用对象（每次网络操作前读取）。
    :param models_root: 模型根目录（plugin_data/{name}/models）。
    :param registry: 模型注册表（本实现仅保存引用，语义由调用方保持一致）。
    """

    def __init__(
        self,
        get_settings: Callable[[], Settings],
        models_root: Path,
        registry: ModelRegistry,
    ) -> None:
        self._get_settings = get_settings
        self._models_root = Path(models_root)
        self._registry = registry
        # 每 key 一把下载锁，避免同一 key 并发重复下载
        self._locks: dict[str, asyncio.Lock] = {}
        # (spec.key, endpoint) -> 已解析的 revision（sha 或回落 ref），避免重复解析
        self._revision_cache: dict[tuple[str, str], str] = {}
        # (path, size, mtime_ns) -> 是否通过校验，避免重复哈希
        self._verify_cache: dict[tuple[Path, int, int], bool] = {}

    # ---- 路径与本地校验 ----

    def _repo_dir(self, spec: ModelSpec) -> Path:
        """该模型落盘根目录：models_root/{repo_id 的 / 换为 --}。"""
        return self._models_root / spec.repo_id.replace("/", "--")

    def _file_path(self, spec: ModelSpec, revision: str, repo_path: str) -> Path:
        """按落盘布局拼出本地文件路径（仓库内路径只取文件名）。"""
        return self._repo_dir(spec) / revision / Path(repo_path).name

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """分块计算文件 sha256（hex）。"""
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _file_ready(self, path: Path, expected_sha256: str | None) -> bool:
        """单文件就绪判定：非空 +（spec 提供时）sha256 一致；带结果缓存。"""
        try:
            if not path.is_file():
                return False
            st = path.stat()
            if st.st_size == 0:
                return False
            if expected_sha256 is None:
                return True
            key = (path, st.st_size, st.st_mtime_ns)
            ok = self._verify_cache.get(key)
            if ok is None:
                ok = self._sha256_file(path).lower() == expected_sha256.lower()
                if len(self._verify_cache) >= _VERIFY_CACHE_MAX:
                    self._verify_cache.clear()
                self._verify_cache[key] = ok
            return ok
        except OSError:
            return False

    def _dir_ready(self, spec: ModelSpec, rev_dir: Path) -> bool:
        """目录就绪判定：spec.files 全部角色文件都就绪（成对语义，display_names 除外）。"""
        for role, repo_path in spec.files.items():
            if not self._file_ready(
                rev_dir / Path(repo_path).name, spec.sha256.get(role)
            ):
                return False
        return True

    def _find_ready_dir(self, spec: ModelSpec) -> tuple[Path | None, str | None]:
        """ref==main 时在 repo 目录下找最新一个全部角色就绪的 sha 子目录。"""
        repo_dir = self._repo_dir(spec)
        if not repo_dir.is_dir():
            return None, None
        candidates: list[tuple[int, Path]] = []
        try:
            for child in repo_dir.iterdir():
                if child.is_dir():
                    try:
                        candidates.append((child.stat().st_mtime_ns, child))
                    except OSError:
                        continue
        except OSError:
            return None, None
        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, child in candidates:
            if self._dir_ready(spec, child):
                return child, child.name
        return None, None

    # ---- 公开只读接口 ----

    def status(self, spec: ModelSpec) -> ModelStatus:
        """纯本地就绪状态：sha256 校验（spec 提供时）+ 非空检查全过才算就绪。

        列表含 spec.files 各角色及可选的 "display_names" 伪角色项；
        display_names 未就绪不影响整体 ready。
        """
        if not spec.files:
            raise ValueError(f"模型 {spec.key} 的 files 为空")
        ready_dir: Path | None = None
        revision: str | None = None
        if spec.ref == "main":
            ready_dir, revision = self._find_ready_dir(spec)
        else:
            rev_dir = self._repo_dir(spec) / spec.ref
            if self._dir_ready(spec, rev_dir):
                ready_dir, revision = rev_dir, spec.ref
        files: list[ModelFileStatus] = []
        total_bytes = 0
        for role, repo_path in spec.files.items():
            if ready_dir is not None:
                path = ready_dir / Path(repo_path).name
                files.append(
                    ModelFileStatus(role=role, path=path, sha256=spec.sha256.get(role))
                )
                try:
                    total_bytes += path.stat().st_size
                except OSError:
                    pass
            else:
                files.append(ModelFileStatus(role=role, path=None, sha256=None))
        if spec.display_names is not None:
            if ready_dir is not None:
                path = ready_dir / Path(spec.display_names).name
                ok = self._file_ready(path, None)
                files.append(
                    ModelFileStatus(
                        role="display_names", path=path if ok else None, sha256=None
                    )
                )
                if ok:
                    try:
                        total_bytes += path.stat().st_size
                    except OSError:
                        pass
            else:
                files.append(
                    ModelFileStatus(role="display_names", path=None, sha256=None)
                )
        return ModelStatus(
            spec_key=spec.key,
            ready=ready_dir is not None,
            files=files,
            revision=revision,
            total_bytes=total_bytes,
        )

    def disk_usage(self) -> int:
        """models_root 递归字节数（含 .part 残留）。"""
        if not self._models_root.is_dir():
            return 0
        total = 0
        try:
            for path in self._models_root.rglob("*"):
                if path.is_file():
                    try:
                        total += path.stat().st_size
                    except OSError:
                        continue
        except OSError:
            pass
        return total

    async def remove(self, spec: ModelSpec) -> None:
        """删除该 key 的本地落盘目录并清空相关缓存。

        注意：落盘布局按 repo_id 分目录，多个 key 共用同一 repo_id 时
        （如两个 wuwa 分类器条目）会一并删除该仓库目录。
        """
        async with self._lock_for(spec.key):
            repo_dir = self._repo_dir(spec)
            shutil.rmtree(repo_dir, ignore_errors=True)
            self._verify_cache = {
                key: ok
                for key, ok in self._verify_cache.items()
                if not key[0].is_relative_to(repo_dir)
            }
            self._revision_cache = {
                key: rev
                for key, rev in self._revision_cache.items()
                if key[0] != spec.key
            }

    # ---- 网络与下载 ----

    @staticmethod
    def _endpoint(settings: Settings) -> str:
        """规范化 HF endpoint（去尾斜杠；空值回落默认地址）。"""
        return (settings.hf_endpoint or "https://huggingface.co").rstrip("/")

    @staticmethod
    def _headers(settings: Settings) -> dict[str, str]:
        """构造请求头；hf_token 只进 Authorization 头，绝不写日志。"""
        headers = {"User-Agent": "astrbot_plugin_character_identifier/1.0"}
        if settings.hf_token:
            headers["Authorization"] = f"Bearer {settings.hf_token}"
        return headers

    async def _fetch_sha(
        self, endpoint: str, repo_id: str, settings: Settings
    ) -> str | None:
        """GET {endpoint}/api/models/{repo_id}/revision/main 解析 commit sha；失败返回 None。"""
        url = f"{endpoint}/api/models/{repo_id}/revision/main"
        timeout = aiohttp.ClientTimeout(total=_SHA_API_TIMEOUT)
        try:
            async with aiohttp.ClientSession(
                timeout=timeout, headers=self._headers(settings)
            ) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.debug("main sha 解析失败 HTTP %s: %s", resp.status, url)
                        return None
                    payload = await resp.json(content_type=None)
            if isinstance(payload, dict):
                sha = payload.get("sha")
                if isinstance(sha, str) and sha.strip():
                    return sha.strip()
            logger.debug("main sha 解析响应缺少 sha 字段: %s", url)
        except Exception as exc:  # 网络异常/超时/非 JSON → 回落直接用 ref
            logger.debug("main sha 解析异常（回落 ref）: %s", exc)
        return None

    async def _resolve_revision(self, spec: ModelSpec, settings: Settings) -> str:
        """解析落盘 revision：main 先问 API 取 sha（失败回落 ref），结果按 (key, endpoint) 缓存。"""
        if spec.ref != "main":
            return spec.ref
        endpoint = self._endpoint(settings)
        cache_key = (spec.key, endpoint)
        cached = self._revision_cache.get(cache_key)
        if cached is not None:
            return cached
        sha = await self._fetch_sha(endpoint, spec.repo_id, settings)
        revision = sha if sha is not None else spec.ref
        self._revision_cache[cache_key] = revision
        if sha is None:
            logger.warning(
                "无法解析 %s 的 main commit sha，回落到 ref=%s", spec.repo_id, spec.ref
            )
        return revision

    async def _download_one(
        self,
        dest: Path,
        url: str,
        expected_sha256: str | None,
        settings: Settings,
    ) -> None:
        """单次下载：分块写 .part，非空 + sha256 校验通过后 os.replace 原子改名。

        失败时清理本次产生的 .part 后抛 DownloadError。
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(f"{dest.name}.part")
        timeout = aiohttp.ClientTimeout(total=_SINGLE_FILE_TIMEOUT)
        try:
            async with aiohttp.ClientSession(
                timeout=timeout, headers=self._headers(settings)
            ) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise DownloadError(
                            f"下载 {dest.name} 失败: HTTP {resp.status}"
                        )
                    content_length = resp.content_length
                    if content_length == 0:
                        raise DownloadError(
                            f"下载 {dest.name} 失败: 内容为空 (Content-Length=0)"
                        )
                    hasher = hashlib.sha256()
                    size = 0
                    with open(part, "wb") as fh:
                        async for chunk in resp.content.iter_chunked(_CHUNK_SIZE):
                            fh.write(chunk)
                            hasher.update(chunk)
                            size += len(chunk)
            if content_length is not None and size != content_length:
                raise DownloadError(
                    f"下载 {dest.name} 不完整: 期望 {content_length} 字节, 实际 {size} 字节"
                )
            if size == 0:
                raise DownloadError(f"下载 {dest.name} 失败: 内容为空")
            if expected_sha256 is not None:
                actual = hasher.hexdigest()
                if actual.lower() != expected_sha256.lower():
                    raise DownloadError(
                        f"sha256 校验失败: {dest.name} "
                        f"(期望 {expected_sha256[:12]}..., 实际 {actual[:12]}...)"
                    )
            os.replace(part, dest)
        except DownloadError:
            part.unlink(missing_ok=True)
            raise
        except Exception as exc:
            part.unlink(missing_ok=True)
            raise DownloadError(f"下载 {dest.name} 失败: {exc}") from exc

    async def _download_file(
        self,
        spec: ModelSpec,
        repo_path: str,
        expected_sha256: str | None,
        revision: str,
        settings: Settings,
    ) -> Path:
        """下载仓库内单个文件到落盘目录；失败指数退避重试 _MAX_RETRIES 次，返回最终路径。"""
        dest = self._file_path(spec, revision, repo_path)
        url = (
            f"{self._endpoint(settings)}/{spec.repo_id}/resolve/{revision}/{repo_path}"
        )
        last_error: DownloadError | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                await self._download_one(dest, url, expected_sha256, settings)
                return dest
            except DownloadError as exc:
                last_error = exc
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_BACKOFF[attempt]
                    logger.warning(
                        "下载 %s 第 %d 次失败，%ds 后重试: %s",
                        dest.name,
                        attempt + 1,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
        raise DownloadError(
            f"模型 {spec.key} 文件 {dest.name} 重试 {_MAX_RETRIES} 次后仍失败"
        ) from last_error

    @staticmethod
    def _cleanup_round(created: list[Path], base: Path) -> None:
        """删除本轮新建的落盘文件；目录因此变空时才移除目录本身（旧校验文件不动）。"""
        for path in created:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            base.rmdir()
        except OSError:
            pass

    async def _ensure_locked(self, spec: ModelSpec) -> dict[str, Path]:
        """持锁状态下的 ensure 主体：就绪检查 → sha 解析 → 逐角色下载 → 整对提交。"""
        settings = self._get_settings()
        st = self.status(spec)
        if st.ready:
            # 已整体就绪直接返回（files 含 display_names 项，就绪才带路径）
            return {fs.role: fs.path for fs in st.files if fs.path is not None}
        if settings.offline_mode:
            raise DownloadError(f"离线模式（offline_mode）下模型 {spec.key} 未就绪")
        revision = await self._resolve_revision(spec, settings)
        base = self._repo_dir(spec) / revision
        created: list[Path] = []
        try:
            for role, repo_path in spec.files.items():
                dest = self._file_path(spec, revision, repo_path)
                if self._file_ready(dest, spec.sha256.get(role)):
                    continue  # 该角色已就绪（如同 revision 目录的历史残留），跳过下载
                await self._download_file(
                    spec, repo_path, spec.sha256.get(role), revision, settings
                )
                created.append(dest)
            display_dest: Path | None = None
            if spec.display_names is not None:
                display_dest = self._file_path(spec, revision, spec.display_names)
                if not self._file_ready(display_dest, None):
                    try:
                        await self._download_file(
                            spec, spec.display_names, None, revision, settings
                        )
                        created.append(display_dest)
                    except DownloadError as exc:
                        # display_names 不参与成对校验：失败只影响展示名，不影响模型就绪
                        logger.warning(
                            "展示名映射文件 %s 下载失败（不影响模型就绪）: %s",
                            spec.display_names,
                            exc,
                        )
                        display_dest = None
        except Exception as exc:
            self._cleanup_round(created, base)
            raise DownloadError(f"模型 {spec.key} 下载失败") from exc
        logger.info("模型 %s 下载完成并校验通过: %s", spec.key, base)
        result = {
            role: self._file_path(spec, revision, repo_path)
            for role, repo_path in spec.files.items()
        }
        if display_dest is not None:
            result["display_names"] = display_dest
        return result

    def _lock_for(self, key: str) -> asyncio.Lock:
        """按 key 取（必要时创建）下载锁。"""
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    async def ensure(self, spec: ModelSpec) -> dict[str, Path]:
        """确保模型全部角色就绪并返回 role->本地路径。

        已就绪直接返回；缺失则下载（每 key 一把 asyncio.Lock 串行，失败重试 3 次
        指数退避 1s/2s/4s）。本轮任一失败 → 删除本轮新建文件并抛 DownloadError；
        offline_mode 下缺失即抛。分类器的 display_names 映射文件随同一 revision
        尽力下载（伪 role "display_names"，失败仅 warning 并从返回 dict 中省略）。
        """
        async with self._lock_for(spec.key):
            return await self._ensure_locked(spec)
