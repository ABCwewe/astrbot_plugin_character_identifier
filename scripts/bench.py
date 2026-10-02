"""基准测试：延迟、峰值 RSS、加载/卸载耗时（AGENTS.md §10）。

默认下载并加载当前配置的检测器 + 分类器（large int8），
在合成图上测量：会话加载耗时、检测延迟、分类单裁剪/8裁剪延迟、
端到端（检测 + 8 框分类）、卸载释放耗时与 RSS 变化。

用法：
  python scripts/bench.py [--runs 10] [--threads 2] [--cls-key wuwa_mnv4l_448_int8]
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.classifier import CharacterClassifier  # noqa: E402
from core.config import Settings  # noqa: E402
from core.detector import PersonDetector  # noqa: E402
from core.downloader import ModelDownloader  # noqa: E402
from core.registry import ModelRegistry, load_registry  # noqa: E402

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
# 标准开发布局下插件位于 <AstrBot>/data/plugins/<name>/，其上两级即 AstrBot data 目录
DEFAULT_DATA_ROOT = PLUGIN_ROOT.parents[1]


def plugin_data_dir(data_root: Path) -> Path:
    """插件数据目录：data_root/plugin_data/astrbot_plugin_character_identifier。"""
    return data_root / "plugin_data" / "astrbot_plugin_character_identifier"


def make_session(model_path: Path, threads: int, low_memory: bool = True):
    """与 core/runtime 一致的 CPU 会话工厂（脚本内联副本）。"""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if low_memory:
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])


def rss_mb() -> float:
    """当前进程 RSS（MB）；psutil 不可用返回 NaN。"""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        return float("nan")


def synth_image(w: int = 1280, h: int = 960) -> np.ndarray:
    """确定性合成 BGR 测试图（渐变 + 色块，近似插画复杂度）。"""
    rng = np.random.default_rng(42)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    xw = np.linspace(0, 255, w, dtype=np.uint8)
    xh = np.linspace(0, 255, h, dtype=np.uint8)
    img[:, :, 0] = xw[None, :]
    img[:, :, 1] = xh[:, None]
    img[:, :, 2] = 255 - xw[None, :]
    for _ in range(24):
        x0, y0 = rng.integers(0, w - 100), rng.integers(0, h - 200)
        x1, y1 = x0 + rng.integers(60, 120), y0 + rng.integers(120, 220)
        img[y0:y1, x0:x1] = rng.integers(0, 255, 3, dtype=np.uint8)
    return img


def p95(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * len(xs) + 0.5)) - 1)]


async def main_async(args, data_root: Path) -> None:
    plugin_root = Path(__file__).resolve().parents[1]
    registry: ModelRegistry = load_registry(plugin_root / "data" / "registry.json")
    settings = Settings(threads=args.threads)
    downloader = ModelDownloader(
        get_settings=lambda: settings,
        models_root=plugin_data_dir(data_root) / "models",
        registry=registry,
    )

    det_spec = registry.detector(args.det_key)
    cls_spec = registry.classifier(args.cls_key)
    print(f"确保模型就绪：{det_spec.key} + {cls_spec.key}（缺失则自动下载）")
    t0 = time.perf_counter()
    det_paths = await downloader.ensure(det_spec)
    cls_paths = await downloader.ensure(cls_spec)
    print(f"模型就绪（含缺失下载）：{time.perf_counter() - t0:.1f}s\n")

    rss_base = rss_mb()

    # ---- 检测器 ----
    t0 = time.perf_counter()
    det_sess = make_session(det_paths["model"], args.threads, settings.low_memory)
    det_load = time.perf_counter() - t0
    det = PersonDetector(
        det_sess,
        imgsz=settings.det_imgsz,
        conf=settings.det_conf,
        nms_iou=settings.nms_iou,
        max_persons=settings.max_persons,
        min_box_px=settings.min_box_px,
    )
    print(
        f"检测器加载: {det_load * 1000:.0f} ms（RSS {rss_base:.0f} → {rss_mb():.0f} MB）"
    )

    img = synth_image()
    det_ts = []
    for _ in range(args.runs):
        t = time.perf_counter()
        boxes = det.detect(img)
        det_ts.append((time.perf_counter() - t) * 1000)
    print(
        f"检测延迟（{args.runs} 次, imgsz={settings.det_imgsz}）: "
        f"均值 {sum(det_ts) / len(det_ts):.1f} ms, p95 {p95(det_ts):.1f} ms, "
        f"检出 {len(boxes)} 框（合成图仅供参考）"
    )

    # ---- 分类器 ----
    import onnxruntime as ort  # noqa: F401  # 确保会话类型可被裁剪引用

    t0 = time.perf_counter()
    cls_sess = make_session(cls_paths["model"], args.threads, settings.low_memory)
    cls_load = time.perf_counter() - t0
    with np.load(cls_paths["prototypes"], allow_pickle=False) as z:
        cls = CharacterClassifier(
            cls_sess,
            protos=np.asarray(z["protos"], dtype=np.float32),
            class_names=[str(c) for c in z["class_names"]],
            proto_class_idx=np.asarray(z["proto_class_idx"]),
            tau=float(z["tau"]),
            p_min=float(z["p_min"]),
            version=int(z["version"]),
            backend=settings.preprocess_backend,
        )
    rss_cls = rss_mb()
    print(f"分类器加载: {cls_load * 1000:.0f} ms（RSS {rss_cls:.0f} MB）")

    crop = img[100:800, 200:900]
    single_ts, batch_ts = [], []
    for _ in range(args.runs):
        t = time.perf_counter()
        cls.predict(crop)
        single_ts.append((time.perf_counter() - t) * 1000)
    crops = [
        img[r0 : r0 + 600, c0 : c0 + 400]
        for r0 in (0, 200, 360)
        for c0 in (0, 400, 800)
    ][:8]
    for _ in range(args.runs):
        t = time.perf_counter()
        cls.predict_batch(crops)
        batch_ts.append((time.perf_counter() - t) * 1000)
    print(
        f"分类单裁剪（S={cls.input_size}, batch={cls.dynamic_batch and '动态' or '固定1'}）: "
        f"均值 {sum(single_ts) / len(single_ts):.1f} ms, p95 {p95(single_ts):.1f} ms"
    )
    print(
        f"分类 8 裁剪: 均值 {sum(batch_ts) / len(batch_ts):.1f} ms "
        f"（单裁剪均值 {sum(batch_ts) / len(batch_ts) / 8:.1f} ms）"
    )

    # ---- 端到端 ----
    e2e_ts = []
    for _ in range(args.runs):
        t = time.perf_counter()
        boxes = det.detect(img)
        # 合成图无人脸/人物，无论检出与否都按 8 框分类，保证统计口径稳定
        cls.predict_batch(crops)
        e2e_ts.append((time.perf_counter() - t) * 1000)
    print(
        f"端到端（检测+8分类）: 均值 {sum(e2e_ts) / len(e2e_ts):.0f} ms, "
        f"p95 {p95(e2e_ts):.0f} ms"
    )

    # ---- 卸载 ----
    rss_before = rss_mb()
    t0 = time.perf_counter()
    del det_sess, cls_sess, det, cls
    gc.collect()
    unload = (time.perf_counter() - t0) * 1000
    print(f"卸载: {unload:.0f} ms（RSS {rss_before:.0f} → {rss_mb():.0f} MB）")

    if np.isnan(rss_base):
        print("\n提示：安装 psutil 可显示 RSS（AstrBot 环境已自带）。")


def main() -> int:
    ap = argparse.ArgumentParser(description="角色识别插件基准测试")
    ap.add_argument("--det-key", default="person_detect_v1.1_n")
    ap.add_argument("--cls-key", default="wuwa_mnv4l_448_int8")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument(
        "--data-root",
        default=None,
        help="AstrBot data 目录（默认按标准布局自动推导）",
    )
    args = ap.parse_args()
    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT
    asyncio.run(main_async(args, data_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
