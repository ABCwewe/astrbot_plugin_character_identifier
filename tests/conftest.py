"""公共夹具与路径配置。

统一把插件根（PLUG）与 AstrBot 根插入 sys.path，使 `core.*`、`scripts.*`、
`astrbot.*` 均可直接导入。公共 fixture（合成图、假会话、aiohttp 测试服务器）放这里。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from aiohttp import web

PLUG_ROOT = Path(__file__).resolve().parents[1]
ASTRBOT_ROOT = Path(r"D:\CodeProject\AstrBot\AstrBot")

for _p in (PLUG_ROOT, ASTRBOT_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


# ---------------- 基础图像 ----------------


@pytest.fixture
def synthetic_image():
    """生成平滑渐变 BGR 图；插值类断言对平滑内容差异极小。"""

    def _make(w: int, h: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        grad = np.linspace(0.0, 255.0, w, dtype=np.float32)
        base = np.tile(grad, (h, 1))
        noise = rng.uniform(-4.0, 4.0, (h, w)).astype(np.float32) * 0.2
        arr = np.clip(base + noise, 0, 255).astype(np.uint8)
        bgr = np.repeat(arr[..., None], 3, axis=2)
        return np.ascontiguousarray(bgr)

    return _make


@pytest.fixture
def make_result():
    """构造真实 CharResult 的工厂。"""
    from core.classifier import CharResult

    def _make(
        pred: str | None = "jinhsi_(wuthering_waves)",
        unknown: bool = False,
        head_conf: float = 0.93,
        top1_cos: float = 0.99,
        top5: tuple[tuple[str, float], ...] = (),
    ) -> CharResult:
        return CharResult(
            pred=pred,
            unknown=unknown,
            head_conf=head_conf,
            top1_cos=top1_cos,
            top5=top5,
        )

    return _make


# ---------------- 假会话 ----------------


class _FakeNode:
    """模仿 onnxruntime 输入/输出描述对象。"""

    def __init__(self, name: str, shape, type_: str = "tensor(float)"):
        self.name = name
        self.shape = list(shape)
        self.type = type_


class FakeDetectorSession:
    """可编程的检测会话：按预设输出数组返回，记录送入的 blob。"""

    def __init__(self, out_shape, out_data=None, *, record=None, input_name="images"):
        self._out_shape = list(out_shape)
        self._out_data = out_data
        self.record = record if record is not None else {}
        self._input_name = input_name

    def get_inputs(self):
        return [_FakeNode(self._input_name, [1, 3, 640, 640])]

    def get_outputs(self):
        return [_FakeNode("output0", self._out_shape)]

    def run(self, output_names, input_feed):
        self.record["blob"] = input_feed[self._input_name]
        if self._out_data is None:
            raise AssertionError("FakeDetectorSession 未预设输出")
        return [np.asarray(self._out_data, dtype=np.float32)]


class FakeClassifierSession:
    """可编程的分类会话：动态返回 embedding/logits，记录每次 run 的输入。"""

    def __init__(
        self,
        input_size: int = 448,
        dynamic_batch: bool = True,
        *,
        outputs=None,
        record=None,
    ):
        self._input_size = input_size
        self._dynamic_batch = dynamic_batch
        self._outputs = outputs  # callable(x) -> (emb, logits)；None → 零输出
        self.record = record if record is not None else {"runs": []}

    def get_inputs(self):
        batch = "N" if self._dynamic_batch else 1
        return [_FakeNode("x", [batch, 3, self._input_size, self._input_size])]

    def get_outputs(self):
        return [
            _FakeNode("embedding", ["N", 256]),
            _FakeNode("logits", ["N", 58]),
        ]

    def run(self, output_names, input_feed):
        x = input_feed["x"]
        self.record["runs"].append(x)
        n = x.shape[0]
        if self._outputs is not None:
            emb, logits = self._outputs(x)
        else:
            emb = np.zeros((n, 256), dtype=np.float32)
            logits = np.zeros((n, 58), dtype=np.float32)
        return [np.asarray(emb, dtype=np.float32), np.asarray(logits, dtype=np.float32)]


# ---------------- aiohttp 测试服务器 ----------------


@pytest_asyncio.fixture
async def http_server():
    """启动 127.0.0.1 随机端口的 aiohttp 服务器；调用 start(app) -> 端口号。"""
    runners: list[web.AppRunner] = []

    async def _start(app: web.Application) -> int:
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        runners.append(runner)
        return runner.addresses[0][1]

    yield _start
    for runner in runners:
        await runner.cleanup()
