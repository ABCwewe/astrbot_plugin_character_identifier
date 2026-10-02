"""预处理一致性验证：OpenCV 实现 vs 参考实现（Pillow，examples/predict.py）。

对一批样图比较：
  ① 输入张量差异（OpenCV 后端 vs 参考 Pillow 实现）
  ② embedding 余弦（两张量分别过真实 ONNX 会话）
  ③ unknown/pred 一致率（三道闸判定）
  ④ 各模型变体（mnv4l_448_int8 / mnv4s_384_fp32）各自结果
附带：插件 pillow 后端 vs 参考实现必须逐位一致（自检）。

参考实现逐位复制自 HF `ABCwewe/wuwa_playable_character_identifier`
`examples/predict.py`（同仓库许可 CC BY-NC 4.0）。

验收线（AGENTS.md §11 默认，可调）：embedding 余弦均值 >= 0.995；
pred/unknown 一致率 >= 99%。样图集由作者提供（--images 目录），不内置。

用法：
  python scripts/parity_check.py --images <样图目录>
  python scripts/parity_check.py --images <样图目录> --runs-model 0   # 只比张量，不跑会话
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps  # 脚本路径：参考实现与样图解码需要（AstrBot 自带）

# 允许从插件根导入 core 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.classifier import CharacterClassifier  # noqa: E402
from core.config import Settings  # noqa: E402
from core.downloader import ModelDownloader  # noqa: E402
from core.registry import ModelRegistry, load_registry  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
# 标准开发布局下插件位于 <AstrBot>/data/plugins/<name>/，其上两级即 AstrBot data 目录
DEFAULT_DATA_ROOT = PLUGIN_ROOT.parents[1]

# ---------------- 参考实现（predict.py 逐位复制） ----------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LETTERBOX_SIZE = 512


def letterbox_square(img, size: int = LETTERBOX_SIZE):
    """参考实现：长边缩放到 size（双三次），居中，边缘均值色填充。"""
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    small = img.resize((nw, nh), Image.BICUBIC)
    if (nw, nh) == (size, size):
        return small
    arr = np.asarray(small, dtype=np.uint8)
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]], axis=0)
    fill = np.clip(border.astype(np.float32).mean(axis=0).round(), 0, 255).astype(
        np.uint8
    )
    out = np.empty((size, size, 3), dtype=np.uint8)
    out[:] = fill
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy : oy + nh, ox : ox + nw] = arr
    return Image.fromarray(out, "RGB")


def reference_preprocess(img, size: int) -> np.ndarray:
    """参考实现：EXIF/透明处理 → letterbox 512 → 二次缩放 → 归一化 NCHW。"""
    img = ImageOps.exif_transpose(img)
    if img.mode == "P":
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.convert("RGBA").split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    sq = letterbox_square(img, LETTERBOX_SIZE)
    if sq.size != (size, size):
        sq = sq.resize((size, size), Image.BICUBIC)
    x = np.asarray(sq, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def judge(protos, class_names, proto_class_idx, tau, p_min, emb, logits):
    """参考判定语义（predict.py Predictor.predict 的三道闸）。"""
    fused = emb[0] / (np.linalg.norm(emb[0]) + 1e-9)
    sims = protos @ fused
    top1_cos = float(sims.max())
    p = np.exp(logits[0] - logits[0].max())
    p = p / p.sum()
    head_conf = float(np.max(p))
    head_top = class_names[int(np.argmax(logits[0]))]
    unknown = bool(top1_cos < tau or head_conf < p_min or head_top == "_negative")
    pred = None if unknown else head_top
    return pred, unknown, top1_cos, head_conf


# ---------------- 模型加载 ----------------


def make_session(model_path: Path, threads: int):
    """与 core/runtime.SessionOptions 一致的 CPU 会话（脚本内联副本）。"""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])


def fake_session(input_size: int):
    """仅用于构造 CharacterClassifier 的假会话（预处理路径不调用 run）。"""

    class _Out:
        def __init__(self, name, shape):
            self.name, self.shape = name, shape

    class _Sess:
        def get_inputs(self):
            return [_Out("x", ["N", 3, input_size, input_size])]

        def get_outputs(self):
            return [_Out("embedding", ["N", 256]), _Out("logits", ["N", 58])]

        def run(self, *a, **k):  # pragma: no cover - 不应被调用
            raise RuntimeError("parity 预处理对比不应触发推理")

    return _Sess()


async def ensure_models(registry: ModelRegistry, settings: Settings, data_root: Path):
    """经插件 downloader 确保模型就绪（缺则下载）。"""
    downloader = ModelDownloader(
        get_settings=lambda: settings,
        models_root=data_root
        / "plugin_data"
        / "astrbot_plugin_character_identifier"
        / "models",
        registry=registry,
    )
    paths: dict[str, dict] = {}
    for spec in registry.specs.values():
        if spec.kind == "classifier":
            paths[spec.key] = await downloader.ensure(spec)
    return paths


def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return (
            np.asarray(z["protos"], dtype=np.float32),
            [str(c) for c in z["class_names"]],
            np.asarray(z["proto_class_idx"]),
            float(z["tau"]),
            float(z["p_min"]),
            int(z["version"]),
        )


# ---------------- 主流程 ----------------


def check_variant(spec, paths, images, args) -> bool:
    """对单个分类器变体执行全部一致性检查，返回是否达标。"""
    protos, class_names, proto_class_idx, tau, p_min, version = load_npz(
        paths["prototypes"]
    )
    fake = fake_session(spec_input_size(paths))
    oc = CharacterClassifier(
        fake,
        protos=protos,
        class_names=class_names,
        proto_class_idx=proto_class_idx,
        tau=tau,
        p_min=p_min,
        version=version,
        backend="opencv",
    )
    pil = CharacterClassifier(
        fake,
        protos=protos,
        class_names=class_names,
        proto_class_idx=proto_class_idx,
        tau=tau,
        p_min=p_min,
        version=version,
        backend="pillow",
    )
    size = oc.input_size

    session = None
    if args.runs_model:
        session = make_session(paths["model"], args.threads)

    diffs, cosines = [], []
    bitwise_ok = True
    agree_pred = agree_unknown = n = 0
    for path in images:
        try:
            img = Image.open(path)
            img.load()
        except Exception as exc:
            print(f"  [跳过] {path.name}: 无法解码（{exc}）")
            continue
        rgb = np.asarray(ImageOps.exif_transpose(img).convert("RGB"), dtype=np.uint8)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        x_ref = reference_preprocess(img, size)
        x_oc = oc._preprocess_opencv(bgr)
        x_pil = pil._preprocess_pillow(bgr)

        d = float(np.max(np.abs(x_ref - x_oc)))
        diffs.append(d)
        if not np.array_equal(x_ref, x_pil):
            bitwise_ok = False
            print(f"  [警告] {path.name}: 插件 pillow 后端与参考实现不逐位一致")
        if session is not None:
            out_ref = session.run(None, {session.get_inputs()[0].name: x_ref})
            out_oc = session.run(None, {session.get_inputs()[0].name: x_oc})
            e_ref, e_oc = out_ref[0][0], out_oc[0][0]
            cos = float(
                np.dot(e_ref, e_oc)
                / (np.linalg.norm(e_ref) * np.linalg.norm(e_oc) + 1e-9)
            )
            cosines.append(cos)
            pred_r, unk_r, _, _ = judge(
                protos, class_names, proto_class_idx, tau, p_min, out_ref[0], out_ref[1]
            )
            pred_o, unk_o, _, _ = judge(
                protos, class_names, proto_class_idx, tau, p_min, out_oc[0], out_oc[1]
            )
            n += 1
            agree_pred += int(pred_r == pred_o)
            agree_unknown += int(unk_r == unk_o)
        print(
            f"  {path.name}: 张量max|Δ|={d:.5f}"
            + (f", embedding余弦={cosines[-1]:.5f}" if cosines else "")
        )

    if not diffs:
        print(f"[{spec.key}] 无有效样图")
        return False
    mean_diff = sum(diffs) / len(diffs)
    ok = True
    print(f"[{spec.key}] 张量 max|Δ| 均值={mean_diff:.5f}，最大={max(diffs):.5f}")
    if bitwise_ok:
        print(f"[{spec.key}] pillow 后端 vs 参考：逐位一致 ✓")
    else:
        ok = False
    if cosines:
        mean_cos = sum(cosines) / len(cosines)
        print(
            f"[{spec.key}] embedding 余弦均值={mean_cos:.5f}（下限 {args.cos_min}），"
            f"最小={min(cosines):.5f}"
        )
        print(
            f"[{spec.key}] pred 一致率={agree_pred}/{n}，unknown 一致率={agree_unknown}/{n}"
            f"（下限 {args.agree_min:.0%}）"
        )
        if (
            mean_cos < args.cos_min
            or agree_pred / n < args.agree_min
            or agree_unknown / n < args.agree_min
        ):
            ok = False
    print(f"[{spec.key}] 结论: {'PASS' if ok else 'FAIL'}")
    if session is not None:
        del session
    return ok


def spec_input_size(paths) -> int:
    """从 npz/模型读输入边长；简化：448/384 由文件名判断，未知回落 448。"""
    name = paths["model"].name
    if "384" in name:
        return 384
    return 448


def main() -> int:
    ap = argparse.ArgumentParser(
        description="预处理一致性验证（OpenCV vs 参考 Pillow）"
    )
    ap.add_argument("--images", required=True, help="样图目录（作者提供，不内置）")
    ap.add_argument(
        "--registry", default=None, help="registry.json 路径（默认插件内置）"
    )
    ap.add_argument("--variants", default="all", choices=["all", "int8", "small"])
    ap.add_argument(
        "--runs-model", type=int, default=1, help="是否跑真实会话（0=仅张量对比）"
    )
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cos-min", type=float, default=0.995)
    ap.add_argument("--agree-min", type=float, default=0.99)
    ap.add_argument(
        "--data-root",
        default=None,
        help="AstrBot data 目录（默认按标准布局自动推导）",
    )
    args = ap.parse_args()

    img_dir = Path(args.images)
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"样图目录无图片: {img_dir}")
        return 2
    print(f"样图 {len(images)} 张，来自 {img_dir}")

    plugin_root = Path(__file__).resolve().parents[1]
    registry_path = (
        Path(args.registry) if args.registry else plugin_root / "data" / "registry.json"
    )
    registry: ModelRegistry = load_registry(registry_path)
    settings = Settings()
    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT
    paths = asyncio.run(ensure_models(registry, settings, data_root))

    wanted = {
        "all": ("wuwa_mnv4l_448_int8", "wuwa_mnv4s_384_fp32"),
        "int8": ("wuwa_mnv4l_448_int8",),
        "small": ("wuwa_mnv4s_384_fp32",),
    }[args.variants]

    all_ok = True
    for key in wanted:
        spec = registry.classifier(key)
        ok = check_variant(spec, paths[key], images, args)
        all_ok = all_ok and ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
