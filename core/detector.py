"""人物检测：YOLOv8 单类 person 的 ONNX 前/后处理（纯同步实现）。

流程：letterbox（灰边 114、记录 scale/pad）→ blobFromImage → 推理 → 置信度过滤 →
NMS（cv2.dnn.NMSBoxes）→ 短边/长宽比过滤 → 置信度降序取前 max_persons。
cv2 在 PersonDetector 构造时惰性导入（仅加载模型才引入 OpenCV，见 AGENTS.md §6.3），
模块顶部只依赖 numpy；本模块不 import astrbot。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: 长宽比过滤阈值：min(w,h)/max(w,h) 低于该值（即 >8:1 或 <1:8）视为极端长宽比丢弃。
_ASPECT_MIN = 0.125


@dataclass(frozen=True)
class Detection:
    """单个检测框，坐标已还原到原图（未 letterbox）并裁剪到图内。"""

    xyxy: tuple[float, float, float, float]
    conf: float


class DetectorError(RuntimeError):
    """模型会话或输出形态不符合 YOLOv8 约定，无法执行检测。"""


def _import_cv2() -> Any:
    """惰性导入 OpenCV：仅在 PersonDetector 构造（加载模型）时执行。"""
    import cv2

    return cv2


def _resolve_imgsz(imgsz: int, h_dim: Any, w_dim: Any) -> int:
    """按输入静态 H/W 决定实际 letterbox 尺寸；静态尺寸优先覆盖配置。"""
    static_h = isinstance(h_dim, int) and h_dim > 0
    static_w = isinstance(w_dim, int) and w_dim > 0
    if static_h and static_w:
        if h_dim != w_dim:
            raise DetectorError(f"检测模型静态输入必须为方形，实际 {h_dim}x{w_dim}")
        if h_dim != imgsz:
            logger.warning(
                "检测模型输入尺寸 %d 与配置 det_imgsz=%d 不一致，以模型为准",
                h_dim,
                imgsz,
            )
        return h_dim
    if static_h:
        if h_dim != imgsz:
            logger.warning(
                "检测模型输入高度固定为 %d，与配置 det_imgsz=%d 不一致，以模型为准",
                h_dim,
                imgsz,
            )
        return h_dim
    if static_w:
        if w_dim != imgsz:
            logger.warning(
                "检测模型输入宽度固定为 %d，与配置 det_imgsz=%d 不一致，以模型为准",
                w_dim,
                imgsz,
            )
        return w_dim
    return imgsz


def _classify_output(oshape: list[Any]) -> str:
    """判别输出分支：常规 [1,4+nc,N]、end2end [1,N,5|6]、动态维度返回 "auto"。

    全动态（导出时未固定形状）时无法从元数据判别，由首次推理按实际输出决定。
    """
    dim1, dim2 = oshape[1], oshape[2]
    if not isinstance(dim1, int) and not isinstance(dim2, int):
        return "auto"
    # end2end：末维为 5/6 列（xyxy+conf(,cls)），且首维不是与之相等的行数（避免与常规 [1,5,5] 歧义）。
    is_end2end = (
        isinstance(dim2, int)
        and dim2 in (5, 6)
        and not (isinstance(dim1, int) and dim1 == dim2)
    )
    if is_end2end:
        return "end2end"
    if isinstance(dim1, int) and dim1 >= 5:
        return "regular"
    raise DetectorError(f"无法判别的检测输出形态: {oshape}")


def _branch_from_array(out: np.ndarray) -> str:
    """按实际推理输出判别分支（"auto" 时一次性调用并缓存）。"""
    if out.ndim != 3:
        raise DetectorError(f"检测模型输出必须为 3 维，实际 {out.ndim} 维")
    dim1, dim2 = int(out.shape[1]), int(out.shape[2])
    if dim1 <= 32 < dim2:  # [1, 4+nc, N]，nc 很小、anchor 数很大
        return "regular"
    if dim2 in (5, 6) and dim1 > 32:  # [1, N, 5|6]
        return "end2end"
    raise DetectorError(f"无法判别的检测输出形态: [1,{dim1},{dim2}]")


class PersonDetector:
    """YOLOv8 单类 person 检测器（onnxruntime 会话 + OpenCV 前/后处理）。"""

    def __init__(
        self,
        session: Any,
        *,
        imgsz: int,
        conf: float,
        nms_iou: float,
        max_persons: int,
        min_box_px: int,
    ) -> None:
        """构造：读取并断言输入/输出形态，确定常规或 end2end 分支。

        Args:
            session: onnxruntime.InferenceSession（runtime 保证 CPUExecutionProvider）。
            imgsz: letterbox 目标边长；若模型输入静态 H/W 且不一致，以模型值为准覆盖。
            conf: 置信度阈值。
            nms_iou: NMS IoU 阈值。
            max_persons: 每图最多保留的检测数。
            min_box_px: 最小框短边（像素），低于该值丢弃。
        """
        self._cv2 = _import_cv2()
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if not inputs or not outputs:
            raise DetectorError("检测模型缺少输入或输出")
        inp = inputs[0]
        ishape = list(inp.shape)
        if len(ishape) != 4:
            raise DetectorError(
                f"检测模型输入必须为 4 维 NCHW，实际 {len(ishape)} 维: {ishape}"
            )
        if "float" not in (inp.type or "").lower():
            raise DetectorError(f"检测模型输入类型必须为 float32，实际 {inp.type}")
        if isinstance(ishape[1], int) and ishape[1] != 3:
            raise DetectorError(f"检测模型输入通道数必须为 3（NCHW），实际 {ishape[1]}")
        oshape = list(outputs[0].shape)
        if len(oshape) != 3:
            raise DetectorError(
                f"检测模型输出必须为 3 维，实际 {len(oshape)} 维: {oshape}"
            )
        if len(outputs) > 1:
            logger.debug("检测模型有 %d 个输出，仅使用第一个", len(outputs))
        self._session = session
        self._input_name = inp.name
        self._imgsz = _resolve_imgsz(imgsz, ishape[2], ishape[3])
        self._conf = conf
        self._nms_iou = nms_iou
        self._max_persons = max_persons
        self._min_box_px = min_box_px
        self._branch = _classify_output(oshape)
        logger.debug(
            "检测器就绪：branch=%s input=%s imgsz=%d conf=%.3f iou=%.2f",
            self._branch,
            self._input_name,
            self._imgsz,
            self._conf,
            self._nms_iou,
        )

    @property
    def input_size(self) -> int:
        """letterbox 目标边长（像素）。"""
        return self._imgsz

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        """同步检测：letterbox → blob → 推理 → 过滤/NMS/尺寸过滤 → 降序取前 max_persons。

        纯函数式：无网络与线程操作。输入为 BGR 图像，输出框坐标已还原到原图并裁剪到图内。
        """
        cv2 = self._cv2
        if image_bgr.ndim != 3:
            raise DetectorError(f"输入图像必须为 3 通道 BGR，实际 {image_bgr.ndim} 维")
        h, w = image_bgr.shape[:2]
        if h < 1 or w < 1:
            return []
        # letterbox：等比缩放 + 灰边 114，记录 scale 与 pad(dx,dy)。
        scale = self._imgsz / max(h, w)
        nw = max(1, round(w * scale))
        nh = max(1, round(h * scale))
        dx = (self._imgsz - nw) / 2.0
        dy = (self._imgsz - nh) / 2.0
        if scale != 1.0:
            interp = cv2.INTER_AREA if nw <= w and nh <= h else cv2.INTER_LINEAR
            img = cv2.resize(image_bgr, (nw, nh), interpolation=interp)
        else:
            img = image_bgr
        left, top = int(dx), int(dy)
        right = self._imgsz - nw - left
        bottom = self._imgsz - nh - top
        if left or top or right or bottom:
            img = cv2.copyMakeBorder(
                img,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )
        blob = cv2.dnn.blobFromImage(
            img,
            scalefactor=1.0 / 255.0,
            size=(self._imgsz, self._imgsz),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        raw = self._session.run(None, {self._input_name: blob})[0]
        out = np.asarray(raw)
        if self._branch == "auto":
            self._branch = _branch_from_array(out)
            logger.info("检测输出分支按实际输出判定为 %s", self._branch)
        if self._branch == "regular":
            # [1, 4+nc, N] 转置为 [N, 4+nc]；列 = cx,cy,w,h,cls...（YOLOv8 无 obj，
            # 类分数直接作 conf；nc=1 时即第 5 列）。
            arr = np.transpose(out, (0, 2, 1))[0]
            x1 = arr[:, 0] - arr[:, 2] / 2.0
            y1 = arr[:, 1] - arr[:, 3] / 2.0
            x2 = arr[:, 0] + arr[:, 2] / 2.0
            y2 = arr[:, 1] + arr[:, 3] / 2.0
            confs = arr[:, 4:].max(axis=1)
        else:
            # end2end：列 = x1,y1,x2,y2,conf(,cls)，取第 5 列作 conf。
            arr = out[0]
            x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
            confs = arr[:, 4]
        idx = np.nonzero(confs >= self._conf)[0]
        if idx.size == 0:
            return []
        # 反变换还原到原图坐标并裁剪到图内。
        x1 = np.clip((x1[idx] - dx) / scale, 0.0, float(w - 1))
        y1 = np.clip((y1[idx] - dy) / scale, 0.0, float(h - 1))
        x2 = np.clip((x2[idx] - dx) / scale, 0.0, float(w - 1))
        y2 = np.clip((y2[idx] - dy) / scale, 0.0, float(h - 1))
        confs = confs[idx]
        boxes: list[list[float]] = []
        scores: list[float] = []
        for i in range(x1.shape[0]):
            bw = float(x2[i] - x1[i])
            bh = float(y2[i] - y1[i])
            boxes.append([float(x1[i]), float(y1[i]), bw, bh])
            scores.append(float(confs[i]))
        keep = cv2.dnn.NMSBoxes(boxes, scores, self._conf, self._nms_iou)
        if isinstance(keep, tuple):  # 旧版 OpenCV 返回 (indices,)
            keep = keep[0] if len(keep) else []
        dets: list[Detection] = []
        for i in np.asarray(keep, dtype=np.int64).reshape(-1):
            i = int(i)
            x1_, y1_, x2_, y2_ = float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])
            bw = x2_ - x1_
            bh = y2_ - y1_
            if bw <= 0.0 or bh <= 0.0:
                continue
            mn, mx = min(bw, bh), max(bw, bh)
            if mn < self._min_box_px or mn / mx < _ASPECT_MIN:
                continue
            dets.append(Detection(xyxy=(x1_, y1_, x2_, y2_), conf=float(confs[i])))
        dets.sort(key=lambda d: d.conf, reverse=True)
        return dets[: self._max_persons]
