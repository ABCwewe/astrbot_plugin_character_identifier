"""配置转换：AstrBotConfig（dict 子类）→ 强类型 Settings。

只做一次性的取值、校验与钳制；其余模块只读 Settings，不直接碰 dict。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

logger = logging.getLogger(__name__)

_VALID_DETSZ = frozenset({320, 416, 480, 512, 640})
_VALID_BACKEND = frozenset({"opencv", "pillow"})
_VALID_LABEL_MODE = frozenset({"index", "name"})


@dataclass(frozen=True)
class Settings:
    """插件的强类型配置视图。字段与 `_conf_schema.json` 分组一一对应（扁平化）。"""

    enabled: bool = True
    # models.*
    detector_key: str = "person_detect_v1.1_n"
    classifier_key: str = "wuwa_mnv4l_448_int8"
    preprocess_backend: str = "opencv"
    check_update: bool = False
    hf_endpoint: str = "https://huggingface.co"
    hf_token: str = ""
    offline_mode: bool = False
    # runtime.*
    resident: bool = False
    idle_timeout_sec: int = 300
    threads: int = 2
    max_concurrency: int = 1
    low_memory: bool = True
    infer_timeout: int = 15
    # detect.*
    det_imgsz: int = 640
    det_conf: float = 0.327
    nms_iou: float = 0.5
    max_persons: int = 8
    min_box_px: int = 24
    det_pre_max_side: int = 2048
    # annotate.*
    label_mode: str = "index"
    font_path: str = ""
    out_max_side: int = 1280
    max_images: int = 3
    temp_text: bool = False
    show_top5_for_unknown: bool = False
    cache_keep_days: int = 7
    # scope.*
    session_whitelist: tuple[str, ...] = ()
    session_blacklist: tuple[str, ...] = ()


def _get(cfg: Mapping[str, Any], group: str, key: str) -> Any:
    """读取嵌套配置组下的键；组缺失返回 None。"""
    sub = cfg.get(group)
    if isinstance(sub, Mapping):
        return sub.get(key)
    return None


def _coerce_bool(value: Any, default: bool, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    logger.warning("配置 %s=%r 非 bool，回落默认 %s", name, value, default)
    return default


def _coerce_str(value: Any, default: str, name: str) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return default
    logger.warning("配置 %s=%r 非 string，回落默认 %r", name, value, default)
    return default


def _coerce_int(
    value: Any, default: int, name: str, lo: int, hi: int | None = None
) -> int:
    try:
        iv = int(value)
    except (TypeError, ValueError):
        if value is not None:
            logger.warning("配置 %s=%r 非 int，回落默认 %s", name, value, default)
        return default
    if iv < lo:
        logger.warning("配置 %s=%d 低于下限，钳制为 %d", name, iv, lo)
        return lo
    if hi is not None and iv > hi:
        logger.warning("配置 %s=%d 超过上限，钳制为 %d", name, iv, hi)
        return hi
    return iv


def _coerce_float(value: Any, default: float, name: str, lo: float, hi: float) -> float:
    try:
        fv = float(value)
    except (TypeError, ValueError):
        if value is not None:
            logger.warning("配置 %s=%r 非 float，回落默认 %s", name, value, default)
        return default
    if not lo <= fv <= hi:
        clamped = min(max(fv, lo), hi)
        logger.warning(
            "配置 %s=%s 不在 [%s, %s] 内，钳制为 %s", name, fv, lo, hi, clamped
        )
        return clamped
    return fv


def _coerce_choice(value: Any, default: str, name: str, choices: frozenset[str]) -> str:
    if isinstance(value, str) and value in choices:
        return value
    if value is not None:
        logger.warning(
            "配置 %s=%r 非法（可选 %s），回落默认 %r",
            name,
            value,
            sorted(choices),
            default,
        )
    return default


def _coerce_str_list(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if isinstance(item, (str, int)))
    logger.warning("配置 %s=%r 非 list，忽略", name, value)
    return ()


def load_settings(
    cfg: Mapping[str, Any], *, logger: logging.Logger | None = None
) -> Settings:
    """从 AstrBotConfig 扁平化构造 Settings；非法值回落默认并告警。"""
    log = logger or logging.getLogger(__name__)

    def g(group: str, key: str, fallback: Any) -> Any:
        val = _get(cfg, group, key)
        return fallback if val is None else val

    backend = _coerce_choice(
        g("models", "preprocess_backend", "opencv"),
        "opencv",
        "models.preprocess_backend",
        _VALID_BACKEND,
    )
    label_mode = _coerce_choice(
        g("annotate", "label_mode", "index"),
        "index",
        "annotate.label_mode",
        _VALID_LABEL_MODE,
    )
    det_imgsz = _coerce_int(
        g("detect", "det_imgsz", 640), 640, "detect.det_imgsz", 32, 4096
    )
    if det_imgsz not in _VALID_DETSZ:
        log.warning("配置 detect.det_imgsz=%d 不受支持，回落 640", det_imgsz)
        det_imgsz = 640

    kwargs: dict[str, Any] = {
        "enabled": _coerce_bool(cfg.get("enabled"), True, "enabled"),
        "detector_key": _coerce_str(
            g("models", "detector", "person_detect_v1.1_n"),
            "person_detect_v1.1_n",
            "models.detector",
        ),
        "classifier_key": _coerce_str(
            g("models", "classifier", "wuwa_mnv4l_448_int8"),
            "wuwa_mnv4l_448_int8",
            "models.classifier",
        ),
        "preprocess_backend": backend,
        "check_update": _coerce_bool(
            g("models", "check_update", False), False, "models.check_update"
        ),
        "hf_endpoint": _coerce_str(
            g("models", "hf_endpoint", "https://huggingface.co"),
            "https://huggingface.co",
            "models.hf_endpoint",
        ).rstrip("/"),
        "hf_token": _coerce_str(g("models", "hf_token", ""), "", "models.hf_token"),
        "offline_mode": _coerce_bool(
            g("models", "offline_mode", False), False, "models.offline_mode"
        ),
        "resident": _coerce_bool(
            g("runtime", "resident", False), False, "runtime.resident"
        ),
        "idle_timeout_sec": _coerce_int(
            g("runtime", "idle_timeout_sec", 300), 300, "runtime.idle_timeout_sec", 30
        ),
        "threads": _coerce_int(g("runtime", "threads", 2), 2, "runtime.threads", 1, 8),
        "max_concurrency": _coerce_int(
            g("runtime", "max_concurrency", 1), 1, "runtime.max_concurrency", 1, 4
        ),
        "low_memory": _coerce_bool(
            g("runtime", "low_memory", True), True, "runtime.low_memory"
        ),
        "infer_timeout": _coerce_int(
            g("runtime", "infer_timeout", 15), 15, "runtime.infer_timeout", 5, 3600
        ),
        "det_imgsz": det_imgsz,
        "det_conf": _coerce_float(
            g("detect", "det_conf", 0.327), 0.327, "detect.det_conf", 0.0, 1.0
        ),
        "nms_iou": _coerce_float(
            g("detect", "nms_iou", 0.5), 0.5, "detect.nms_iou", 0.1, 1.0
        ),
        "max_persons": _coerce_int(
            g("detect", "max_persons", 8), 8, "detect.max_persons", 1, 16
        ),
        "min_box_px": _coerce_int(
            g("detect", "min_box_px", 24), 24, "detect.min_box_px", 8
        ),
        "label_mode": label_mode,
        "font_path": _coerce_str(
            g("annotate", "font_path", ""), "", "annotate.font_path"
        ),
        "out_max_side": _coerce_int(
            g("annotate", "out_max_side", 1280), 1280, "annotate.out_max_side", 512
        ),
        "max_images": _coerce_int(
            g("annotate", "max_images", 3), 3, "annotate.max_images", 1, 10
        ),
        "temp_text": _coerce_bool(
            g("annotate", "temp_text", False), False, "annotate.temp_text"
        ),
        "show_top5_for_unknown": _coerce_bool(
            g("annotate", "show_top5_for_unknown", False),
            False,
            "annotate.show_top5_for_unknown",
        ),
        "cache_keep_days": _coerce_int(
            g("annotate", "cache_keep_days", 7), 7, "annotate.cache_keep_days", 1
        ),
        "session_whitelist": _coerce_str_list(
            g("scope", "session_whitelist", []), "scope.session_whitelist"
        ),
        "session_blacklist": _coerce_str_list(
            g("scope", "session_blacklist", []), "scope.session_blacklist"
        ),
    }

    known = {f.name for f in fields(Settings)}
    return Settings(**{k: v for k, v in kwargs.items() if k in known})
