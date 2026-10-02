"""模型注册表：读取 data/registry.json，模型条目 → ModelSpec；展示名映射加载。"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_KINDS = frozenset({"detector", "classifier"})


class RegistryError(RuntimeError):
    """注册表文件缺失、格式非法或条目不合法。"""


@dataclass(frozen=True)
class ModelSpec:
    """单个可下载模型的完整描述。"""

    key: str
    kind: str  # "detector" | "classifier"
    repo_id: str
    ref: str  # "main" / tag / commit
    files: Mapping[str, str]  # role -> 仓库内相对路径
    preprocess: str  # 预处理方案 id
    sha256: Mapping[str, str] = field(default_factory=dict)  # role -> 期望 sha256
    display_names: str | None = None  # 仓库内中文名映射文件路径
    scope_note: str = ""


@dataclass(frozen=True)
class ModelRegistry:
    """key -> ModelSpec 的只读集合。"""

    specs: Mapping[str, ModelSpec]

    def get(self, key: str) -> ModelSpec:
        spec = self.specs.get(key)
        if spec is None:
            raise RegistryError(f"注册表中不存在模型 key: {key}")
        return spec

    def detector(self, key: str) -> ModelSpec:
        spec = self.get(key)
        if spec.kind != "detector":
            raise RegistryError(f"模型 {key} 类型为 {spec.kind}，不是 detector")
        return spec

    def classifier(self, key: str) -> ModelSpec:
        spec = self.get(key)
        if spec.kind != "classifier":
            raise RegistryError(f"模型 {key} 类型为 {spec.kind}，不是 classifier")
        return spec

    def __bool__(self) -> bool:
        return bool(self.specs)


def _parse_entry(key: str, entry: Mapping[str, Any]) -> ModelSpec:
    kind = entry.get("kind")
    if kind not in _KINDS:
        raise RegistryError(f"条目 {key}: kind 非法 {kind!r}")
    repo_id = entry.get("repo_id")
    if not isinstance(repo_id, str) or "/" not in repo_id:
        raise RegistryError(f"条目 {key}: repo_id 非法 {repo_id!r}")
    ref = entry.get("ref") or "main"
    files = entry.get("files")
    if not isinstance(files, Mapping) or not files:
        raise RegistryError(f"条目 {key}: files 必须为非空对象")
    if "model" not in files:
        raise RegistryError(f"条目 {key}: files 必含 model")
    if kind == "classifier" and "prototypes" not in files:
        raise RegistryError(f"条目 {key}: classifier 必须成对提供 prototypes")
    files_map = {str(r): str(p) for r, p in files.items()}
    sha_raw = entry.get("sha256") or {}
    if not isinstance(sha_raw, Mapping):
        raise RegistryError(f"条目 {key}: sha256 必须为对象")
    sha_map = {str(r): str(v) for r, v in sha_raw.items()}
    display = entry.get("display_names")
    display = str(display) if isinstance(display, str) and display else None
    return ModelSpec(
        key=str(key),
        kind=str(kind),
        repo_id=repo_id,
        ref=str(ref),
        files=files_map,
        preprocess=str(entry.get("preprocess") or ""),
        sha256=sha_map,
        display_names=display,
        scope_note=str(entry.get("scope_note") or ""),
    )


def load_registry(path: Path) -> ModelRegistry:
    """从 JSON 文件加载注册表；格式错误抛 RegistryError。"""
    import json

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RegistryError(f"无法读取注册表 {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RegistryError(f"注册表 JSON 非法: {exc}") from exc
    if not isinstance(data, Mapping) or not data:
        raise RegistryError("注册表必须为非空 JSON 对象")
    specs: dict[str, ModelSpec] = {}
    for key, entry in data.items():
        if not isinstance(entry, Mapping):
            raise RegistryError(f"条目 {key} 必须为对象")
        specs[str(key)] = _parse_entry(key, entry)
    return ModelRegistry(specs=specs)


def _parse_display_names_lines(text: str) -> dict[str, str]:
    names: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, sep, value = stripped.partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip("'\"").strip()
        if key and value:
            names[key] = value
    return names


def load_display_names(path: Path) -> dict[str, str]:
    """加载类名 → 中文名映射（class_names_zh.yaml）。

    优先 yaml.safe_load；PyYAML 不可用时按行解析（格式仅 `key: "value"` 与注释）。
    解析失败抛 RegistryError；`_negative` 键原样保留，由展示层过滤。
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RegistryError(f"无法读取展示名映射 {path}: {exc}") from exc
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        names = _parse_display_names_lines(text)
    else:
        try:
            loaded = yaml.safe_load(text)
        except Exception as exc:  # yaml.YAMLError 及其他解析异常
            raise RegistryError(f"展示名映射 YAML 解析失败: {exc}") from exc
        if isinstance(loaded, Mapping):
            names = {str(k): str(v) for k, v in loaded.items() if v is not None}
        else:
            names = _parse_display_names_lines(text)
    if names:
        return names
    logger.warning("展示名映射 %s 为空", path)
    return {}
