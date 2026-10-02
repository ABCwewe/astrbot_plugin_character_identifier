"""角色分类器：BGR 裁剪图 → letterbox 预处理 → embedding+logits 推理 → 原型余弦 + 三道闸拒识。

判定语义与模型仓库 `ABCwewe/wuwa_playable_character_identifier` 的
`examples/predict.py`（权威参考实现）逐位一致，本模块仅把预处理从 Pillow
移植到 OpenCV，并提供 pillow 兜底后端（惰性导入，逐位复刻参考实现）。

输入约定：图像缓冲为 BGR uint8（全局约定），本模块内部先转 RGB。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: 训练口径的 letterbox 边长：先缩到 512，再二次 resize 到模型输入尺寸。
_LETTERBOX_SIZE = 512
#: 模型输入边长允许值（384/448 为现有变体；512 表示 letterbox 后无需二次缩放）。
_SUPPORTED_SIZES = frozenset({384, 448, 512})
#: 分类头负类：top-1 落在此类 → 不是任何已知角色。
_NEGATIVE_CLASS = "_negative"

#: ImageNet 归一化参数，预展开成 (3,1,1) 广播形状的 float32 常量（全程 float32）。
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


@dataclass(frozen=True)
class CharResult:
    """单个裁剪图的识别结果。

    - ``pred``：精确类名；三道闸任一命中（unknown）时为 None。
    - ``head_conf``：分类头 softmax 最大值，展示用置信度（不是原型余弦）。
    - ``top1_cos``：与最佳角色原型的余弦，仅内部/调试用。
    - ``top5``：按类去重后的原型余弦 top5（同类取最佳），降序，精确类名。
    """

    pred: str | None
    unknown: bool
    head_conf: float
    top1_cos: float
    top5: tuple[tuple[str, float], ...]


class ClassifierError(RuntimeError):
    """分类器输入/输出与契约不符等可恢复错误。"""


def _softmax(x: np.ndarray) -> np.ndarray:
    """数值稳定的 softmax（``exp(x-max)``，逐位复刻参考实现）。"""
    p = np.exp(x - x.max())
    return p / p.sum()


class CharacterClassifier:
    """鸣潮角色分类器（MobileNetV4 双输出 + 原型余弦 + 三道闸拒识）。"""

    def __init__(
        self,
        session: Any,
        *,
        protos: np.ndarray,
        class_names: Sequence[str],
        proto_class_idx: np.ndarray,
        tau: float,
        p_min: float,
        version: int,
        backend: str = "opencv",
    ) -> None:
        """构造并校验模型输入/输出契约。

        ``session`` 为 onnxruntime.InferenceSession（runtime 保证 CPUExecutionProvider）；
        ``protos/proto_class_idx/class_names/tau/p_min/version`` 均来自配套原型库 npz
        （由 runtime 读取后传入，本模块不做 ``np.load``）。``backend`` 为 "opencv"（默认）
        或 "pillow"（惰性导入 Pillow，逐位复刻参考实现）。
        """
        if backend not in ("opencv", "pillow"):
            raise ValueError(f"不支持的预处理后端: {backend!r}（可选 opencv/pillow）")
        self._backend = backend
        self._session = session

        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ClassifierError(f"分类器应恰好有 1 个输入，实际 {len(inputs)} 个")
        in0 = inputs[0]
        shape = tuple(in0.shape)
        if len(shape) != 4:
            raise ClassifierError(f"分类器输入应为 NCHW 4 维，实际 shape={shape}")
        # 空间尺寸必须为静态、相等且 ∈ {384,448,512}；动态尺寸按契约抛错
        s = shape[2]
        if not isinstance(s, int) or s not in _SUPPORTED_SIZES:
            raise ClassifierError(
                f"分类器输入边长必须为静态 384/448/512，实际 shape[2]={s!r}"
            )
        if shape[3] != s:
            raise ClassifierError(f"分类器输入高宽不一致: shape={shape}")
        self._input_size = int(s)
        # batch 维：固定整数 1 → 逐个推理；动态（str/None）→ 合批
        batch = shape[0]
        self._dynamic_batch = not (isinstance(batch, int) and batch == 1)
        self._input_name = in0.name

        emb_idx, logits_idx = self._locate_outputs(session.get_outputs())
        self._emb_idx = emb_idx
        self._logits_idx = logits_idx

        protos = np.ascontiguousarray(protos, dtype=np.float32)
        if protos.ndim != 2 or protos.shape[1] != 256:
            raise ClassifierError(
                f"原型库 protos 应为 (N,256)，实际 shape={protos.shape}"
            )
        idx = np.asarray(proto_class_idx)
        if idx.ndim != 1 or idx.shape[0] != protos.shape[0]:
            raise ClassifierError(
                f"proto_class_idx 长度 {idx.shape[0] if idx.ndim else 0} 与 protos 行数 {protos.shape[0]} 不符"
            )
        names = tuple(str(c) for c in class_names)
        if len(idx) and int(idx.max()) >= len(names):
            raise ClassifierError("proto_class_idx 越界 class_names")
        self._protos = protos
        self._proto_class_idx = idx
        self._class_names = names
        self._tau = float(tau)
        self._p_min = float(p_min)
        self.version = int(version)

    @staticmethod
    def _locate_outputs(outputs: Sequence[Any]) -> tuple[int, int]:
        """按形状/输出名定位 embedding 与 logits 输出下标，返回 (emb_idx, logits_idx)。

        优先按维度判别：256 维 → embedding，58 维 → logits；两者维度同型/歧义时
        按输出名（含 "embed"/"logit"）判别；仍无法判别则抛 ClassifierError。
        """
        if len(outputs) != 2:
            raise ClassifierError(f"分类器应恰好有 2 个输出，实际 {len(outputs)} 个")
        by_shape: dict[str, list[int]] = {"embedding": [], "logits": []}
        for i, out in enumerate(outputs):
            dims = {int(d) for d in out.shape if isinstance(d, int)}
            if 256 in dims and 58 not in dims:
                by_shape["embedding"].append(i)
            elif 58 in dims and 256 not in dims:
                by_shape["logits"].append(i)
        if len(by_shape["embedding"]) == 1 and len(by_shape["logits"]) == 1:
            return by_shape["embedding"][0], by_shape["logits"][0]
        by_name: dict[str, int] = {}
        for i, out in enumerate(outputs):
            name = (out.name or "").lower()
            if "embed" in name:
                by_name.setdefault("embedding", i)
            elif "logit" in name:
                by_name.setdefault("logits", i)
        if "embedding" in by_name and "logits" in by_name:
            return by_name["embedding"], by_name["logits"]
        detail = "; ".join(f"{o.name}(shape={o.shape})" for o in outputs)
        raise ClassifierError(f"无法按形状或输出名区分 embedding/logits 输出: {detail}")

    @property
    def input_size(self) -> int:
        """模型输入边长 S（384/448/512）。"""
        return self._input_size

    @property
    def dynamic_batch(self) -> bool:
        """batch 维是否动态（True → ``predict_batch`` 合批一次推理）。"""
        return self._dynamic_batch

    def predict(self, crop_bgr: np.ndarray) -> CharResult:
        """同步识别单张 BGR 裁剪图。"""
        x = self._preprocess(crop_bgr)
        outs = self._session.run(None, {self._input_name: x})
        return self._judge(outs, 0)

    def predict_batch(self, crops: Sequence[np.ndarray]) -> list[CharResult]:
        """批量识别；``dynamic_batch`` 时合批一次推理（按实际 N，不做 padding），
        否则逐个 ``predict``。空列表返回 ``[]``。"""
        n = len(crops)
        if n == 0:
            return []
        if not self._dynamic_batch:
            return [self.predict(c) for c in crops]
        xs = [self._preprocess(c) for c in crops]
        x = np.concatenate(xs, axis=0)  # (N,3,S,S) float32 连续
        outs = self._session.run(None, {self._input_name: x})
        return [self._judge(outs, i) for i in range(n)]

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """BGR 裁剪图 → (1,3,S,S) float32 NCHW 连续张量（与参考实现逐位对齐）。"""
        if (
            not isinstance(crop_bgr, np.ndarray)
            or crop_bgr.ndim != 3
            or crop_bgr.shape[2] != 3
            or crop_bgr.dtype != np.uint8
        ):
            raise ClassifierError(
                f"裁剪图应为 HWC BGR uint8，实际 shape={getattr(crop_bgr, 'shape', None)} "
                f"dtype={getattr(crop_bgr, 'dtype', None)}"
            )
        if crop_bgr.shape[0] == 0 or crop_bgr.shape[1] == 0:
            raise ClassifierError("空裁剪图无法识别")
        if self._backend == "opencv":
            return self._preprocess_opencv(crop_bgr)
        return self._preprocess_pillow(crop_bgr)

    def _preprocess_opencv(self, crop_bgr: np.ndarray) -> np.ndarray:
        """OpenCV 后端：BGR→RGB → letterbox 512（纯缩小 INTER_AREA，其余 INTER_CUBIC）
        → 二次 resize → 归一化。填充色 = 缩放后四边像素均值（角点重复计入）。"""
        import cv2

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = _LETTERBOX_SIZE / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        if nw != w or nh != h:
            interp = cv2.INTER_AREA if (nw <= w and nh <= h) else cv2.INTER_CUBIC
            small = cv2.resize(rgb, (nw, nh), interpolation=interp)
        else:
            small = rgb
        if (nw, nh) != (_LETTERBOX_SIZE, _LETTERBOX_SIZE):
            arr = small
            border = np.concatenate(
                [arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]], axis=0
            )
            fill = np.clip(
                border.astype(np.float32).mean(axis=0).round(), 0, 255
            ).astype(np.uint8)
            out = np.empty((_LETTERBOX_SIZE, _LETTERBOX_SIZE, 3), dtype=np.uint8)
            out[:] = fill
            oy, ox = (_LETTERBOX_SIZE - nh) // 2, (_LETTERBOX_SIZE - nw) // 2
            out[oy : oy + nh, ox : ox + nw] = arr
            sq = out
        else:
            sq = small
        if self._input_size != _LETTERBOX_SIZE:
            interp = (
                cv2.INTER_AREA
                if self._input_size < _LETTERBOX_SIZE
                else cv2.INTER_CUBIC
            )
            sq = cv2.resize(
                sq, (self._input_size, self._input_size), interpolation=interp
            )
        return self._normalize(sq)

    def _preprocess_pillow(self, crop_bgr: np.ndarray) -> np.ndarray:
        """pillow 后端：惰性导入 PIL，逐位复刻参考实现（BICUBIC + 边缘均值填充色）。"""
        from PIL import Image

        rgb = np.ascontiguousarray(crop_bgr[..., ::-1])  # BGR → RGB，不依赖 cv2
        img = Image.fromarray(rgb)
        w, h = img.size
        scale = _LETTERBOX_SIZE / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        sq = img.resize((nw, nh), Image.BICUBIC)
        if (nw, nh) != (_LETTERBOX_SIZE, _LETTERBOX_SIZE):
            arr = np.asarray(sq, dtype=np.uint8)
            border = np.concatenate(
                [arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]], axis=0
            )
            fill = np.clip(
                border.astype(np.float32).mean(axis=0).round(), 0, 255
            ).astype(np.uint8)
            out = np.empty((_LETTERBOX_SIZE, _LETTERBOX_SIZE, 3), dtype=np.uint8)
            out[:] = fill
            oy, ox = (_LETTERBOX_SIZE - nh) // 2, (_LETTERBOX_SIZE - nw) // 2
            out[oy : oy + nh, ox : ox + nw] = arr
            sq = Image.fromarray(out, "RGB")
        if sq.size != (self._input_size, self._input_size):
            sq = sq.resize((self._input_size, self._input_size), Image.BICUBIC)
        return self._normalize(np.asarray(sq, dtype=np.uint8))

    @staticmethod
    def _normalize(sq: np.ndarray) -> np.ndarray:
        """HWC uint8 → (1,3,S,S) float32 NCHW 连续（/255 + ImageNet mean/std，全程 float32）。"""
        x = sq.astype(np.float32) / np.float32(255.0)
        x = x.transpose(2, 0, 1)
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
        return np.ascontiguousarray(x[None])

    def _judge(self, outs: Sequence[np.ndarray], index: int) -> CharResult:
        """按参考判定语义对第 ``index`` 个样本输出做三道闸拒识。"""
        emb = outs[self._emb_idx]
        logits = outs[self._logits_idx]
        e0 = emb[index] if emb.ndim == 2 else emb
        l0 = logits[index] if logits.ndim == 2 else logits
        fused = e0 / (np.linalg.norm(e0) + 1e-9)
        sims = self._protos @ fused
        top1_cos = float(sims.max())
        best: dict[str, float] = {}  # top5 按类去重（同类取最佳），降序遍历
        for i in np.argsort(-sims):
            name = self._class_names[int(self._proto_class_idx[int(i)])]
            if name not in best:
                best[name] = float(sims[i])
                if len(best) >= 5:
                    break
        top5 = tuple(sorted(best.items(), key=lambda kv: -kv[1]))
        probs = _softmax(l0)
        head_conf = float(probs.max())
        head_top = self._class_names[int(np.argmax(l0))]
        unknown = bool(
            top1_cos < self._tau
            or head_conf < self._p_min
            or head_top == _NEGATIVE_CLASS
        )
        pred = None if unknown else head_top
        return CharResult(
            pred=pred,
            unknown=unknown,
            head_conf=head_conf,
            top1_cos=top1_cos,
            top5=top5,
        )
