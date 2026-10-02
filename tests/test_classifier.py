"""classifier 模块测试：三道闸拒识、top5 去重、softmax 稳定、批量推理、预处理一致性。"""

from __future__ import annotations

import numpy as np
import pytest
from core.classifier import CharacterClassifier, ClassifierError

from tests.conftest import FakeClassifierSession, _FakeNode

CLASS_NAMES = ["_negative", "a", "b", "c", "d", "e", "f", "g"]
TAU = 0.5
P_MIN = 0.5
INPUT_SIZE = 448


def _protos(seed: int = 7, n: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.standard_normal((n, 256), dtype=np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return np.ascontiguousarray(p)


def _make_clf(session, *, protos=None, idx=None, backend="opencv"):
    protos = _protos() if protos is None else protos
    idx = np.arange(1, 8, dtype=np.int64) if idx is None else idx
    return CharacterClassifier(
        session,
        protos=protos,
        class_names=CLASS_NAMES,
        proto_class_idx=idx,
        tau=TAU,
        p_min=P_MIN,
        version=1,
        backend=backend,
    )


def _fixed_outputs(emb: np.ndarray, logits: np.ndarray):
    def _f(x):
        n = x.shape[0]
        return (
            np.tile(np.asarray(emb, dtype=np.float32), (n, 1)),
            np.tile(np.asarray(logits, dtype=np.float32), (n, 1)),
        )

    return _f


def _softmax_ref(v: np.ndarray) -> np.ndarray:
    p = np.exp(v - v.max())
    return p / p.sum()


def _crop(synthetic_image, w=100, h=80):
    return np.ascontiguousarray(synthetic_image(w, h, seed=42))


# ---------------- 三道闸与通过 ----------------


def test_pass_all_gates(synthetic_image):
    protos = _protos()
    idx = np.arange(1, 8, dtype=np.int64)
    emb = np.asarray(protos[3], dtype=np.float32)
    logits = np.array([0.1, 0.2, 0.3, 5.0, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    clf = _make_clf(
        FakeClassifierSession(INPUT_SIZE, outputs=_fixed_outputs(emb, logits)), idx=idx
    )
    r = clf.predict(_crop(synthetic_image))

    fused = emb / (np.linalg.norm(emb) + 1e-9)
    sims = protos @ fused
    exp_top1 = float(sims.max())
    exp_probs = _softmax_ref(logits)
    exp_head = float(exp_probs.max())
    exp_top = CLASS_NAMES[int(np.argmax(logits))]

    assert r.unknown is False
    assert r.pred == exp_top == "c"
    assert r.head_conf == pytest.approx(exp_head, rel=1e-5, abs=1e-7)
    assert r.top1_cos == pytest.approx(exp_top1, abs=1e-6)
    assert len(r.top5) <= 5
    best_idx = int(np.argmax(sims))
    assert r.top5[0][0] == CLASS_NAMES[int(idx[best_idx])]
    assert r.top5[0][1] == pytest.approx(exp_top1, abs=1e-6)
    # top5 降序
    scores = [s for _, s in r.top5]
    assert scores == sorted(scores, reverse=True)


def test_gate1_top1_cos_below_tau(synthetic_image):
    protos = _protos()
    # 与所有原型都不对齐的随机单位向量 → top1_cos << tau
    rng = np.random.default_rng(123)
    emb = rng.standard_normal(256).astype(np.float32)
    emb /= np.linalg.norm(emb)
    assert (protos @ emb).max() < TAU  # 自检：确保触发的是第一道闸
    logits = np.array([0.1, 0.2, 5.0, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    clf = _make_clf(
        FakeClassifierSession(INPUT_SIZE, outputs=_fixed_outputs(emb, logits))
    )
    r = clf.predict(_crop(synthetic_image))
    assert r.unknown is True
    assert r.pred is None
    # 第二道闸未触发（head_conf 高），确认是第一道闸
    assert r.head_conf >= P_MIN


def test_gate2_head_conf_below_pmin(synthetic_image):
    protos = _protos()
    emb = np.asarray(protos[3], dtype=np.float32)  # top1_cos ~1.0
    logits = np.array([0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], dtype=np.float32)
    clf = _make_clf(
        FakeClassifierSession(INPUT_SIZE, outputs=_fixed_outputs(emb, logits))
    )
    r = clf.predict(_crop(synthetic_image))
    assert r.unknown is True
    assert r.pred is None
    assert r.top1_cos >= TAU  # 第一道闸未触发
    assert r.head_conf < P_MIN


def test_gate3_head_top_negative(synthetic_image):
    protos = _protos()
    emb = np.asarray(protos[3], dtype=np.float32)
    logits = np.array([5.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    clf = _make_clf(
        FakeClassifierSession(INPUT_SIZE, outputs=_fixed_outputs(emb, logits))
    )
    r = clf.predict(_crop(synthetic_image))
    assert r.unknown is True
    assert r.pred is None
    assert r.head_conf >= P_MIN
    assert r.top1_cos >= TAU


def test_top5_dedup_same_class(synthetic_image):
    """两个原型同类只留最佳分数。"""
    protos = _protos(n=8)
    idx = np.array([1, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)  # 行0、1 同类 a
    emb = np.asarray(protos[0], dtype=np.float32)  # 与原型0完全对齐
    logits = np.array([0.1, 0.2, 0.3, 5.0, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    clf = _make_clf(
        FakeClassifierSession(INPUT_SIZE, outputs=_fixed_outputs(emb, logits)),
        protos=protos,
        idx=idx,
    )
    r = clf.predict(_crop(synthetic_image))
    names = [n for n, _ in r.top5]
    assert names.count("a") == 1
    assert r.top5[0] == ("a", pytest.approx(1.0, abs=1e-4))
    assert len(r.top5) == 5


def test_softmax_large_logits_stable(synthetic_image):
    """logits + 1e4 不溢出，head_conf 与数值稳定 softmax 一致。"""
    protos = _protos()
    emb = np.asarray(protos[3], dtype=np.float32)
    logits = np.array(
        [1e4, 1e4 - 1, 1e4 - 2, 1e4 - 3, 1e4 - 4, 1e4 - 5, 1e4 - 6, 1e4 - 7],
        dtype=np.float32,
    )
    clf = _make_clf(
        FakeClassifierSession(INPUT_SIZE, outputs=_fixed_outputs(emb, logits))
    )
    r = clf.predict(_crop(synthetic_image))
    assert np.isfinite(r.head_conf)
    expected = float(_softmax_ref(logits.astype(np.float64)).max())
    assert r.head_conf == pytest.approx(expected, rel=1e-5, abs=1e-4)


# ---------------- 批量推理 ----------------


def test_predict_batch_dynamic_single_run(synthetic_image):
    record = {"runs": []}
    clf = _make_clf(
        FakeClassifierSession(INPUT_SIZE, dynamic_batch=True, record=record)
    )
    assert clf.dynamic_batch is True
    crops = [_crop(synthetic_image, 60, 60) for _ in range(3)]
    results = clf.predict_batch(crops)
    assert len(results) == 3
    assert len(record["runs"]) == 1  # 只调 1 次 run
    x = record["runs"][0]
    assert x.shape[0] == 3
    assert x.shape == (3, 3, INPUT_SIZE, INPUT_SIZE)


def test_predict_batch_static_per_crop(synthetic_image):
    record = {"runs": []}
    clf = _make_clf(
        FakeClassifierSession(INPUT_SIZE, dynamic_batch=False, record=record)
    )
    assert clf.dynamic_batch is False
    crops = [_crop(synthetic_image, 60, 60) for _ in range(3)]
    results = clf.predict_batch(crops)
    assert len(results) == 3
    assert len(record["runs"]) == 3  # 逐个推理
    for x in record["runs"]:
        assert x.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)


def test_predict_batch_empty(synthetic_image):
    record = {"runs": []}
    clf = _make_clf(FakeClassifierSession(INPUT_SIZE, record=record))
    assert clf.predict_batch([]) == []
    assert record["runs"] == []


# ---------------- 构造断言 ----------------


def test_double_embedding_output_raises():
    class _Sess:
        def get_inputs(self):
            return [_FakeNode("x", [1, 3, INPUT_SIZE, INPUT_SIZE])]

        def get_outputs(self):
            return [_FakeNode("e1", ["N", 256]), _FakeNode("e2", ["N", 256])]

        def run(self, *a, **k):
            raise AssertionError("构造不应触发推理")

    with pytest.raises(ClassifierError):
        _make_clf(_Sess())


def test_dynamic_spatial_dim_raises():
    class _Sess:
        def get_inputs(self):
            return [_FakeNode("x", [1, 3, "H", "H"])]

        def get_outputs(self):
            return [_FakeNode("embedding", ["N", 256]), _FakeNode("logits", ["N", 58])]

        def run(self, *a, **k):
            raise AssertionError("构造不应触发推理")

    with pytest.raises(ClassifierError):
        _make_clf(_Sess())


def test_invalid_backend_raises():
    with pytest.raises(ValueError):
        CharacterClassifier(
            FakeClassifierSession(INPUT_SIZE),
            protos=_protos(),
            class_names=CLASS_NAMES,
            proto_class_idx=np.arange(1, 8, dtype=np.int64),
            tau=TAU,
            p_min=P_MIN,
            version=1,
            backend="tensorrt",
        )


# ---------------- 预处理一致性 ----------------


def test_pillow_backend_bitwise_matches_reference(synthetic_image):
    """pillow 后端与参考实现（scripts/parity_check.reference_preprocess）逐位一致。"""
    import scripts.parity_check as pc  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    clf = _make_clf(FakeClassifierSession(INPUT_SIZE), backend="pillow")
    for w, h in [(800, 300), (100, 100), (2000, 1200), (512, 512)]:
        bgr = _crop(synthetic_image, w, h)
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        x_ref = pc.reference_preprocess(Image.fromarray(rgb), INPUT_SIZE)
        x_pil = clf._preprocess_pillow(bgr)
        assert x_ref.shape == x_pil.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
        assert np.array_equal(x_ref, x_pil), f"size {w}x{h} 逐位不一致"


def test_opencv_backend_shape_and_bounded_diff(synthetic_image):
    """opencv 后端输出 shape/float32/有限值，中心区域与 pillow 差异有界。"""
    clf_oc = _make_clf(FakeClassifierSession(INPUT_SIZE), backend="opencv")
    clf_pil = _make_clf(FakeClassifierSession(INPUT_SIZE), backend="pillow")
    for w, h in [(300, 200), (1000, 700), (2000, 1200), (512, 512)]:
        bgr = _crop(synthetic_image, w, h)
        x_oc = clf_oc._preprocess_opencv(bgr)
        assert x_oc.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
        assert x_oc.dtype == np.float32
        assert np.isfinite(x_oc).all()
        x_pil = clf_pil._preprocess_pillow(bgr)
        s = INPUT_SIZE
        c_oc = x_oc[:, :, s // 4 : 3 * s // 4, s // 4 : 3 * s // 4]
        c_pil = x_pil[:, :, s // 4 : 3 * s // 4, s // 4 : 3 * s // 4]
        assert float(np.max(np.abs(c_oc - c_pil))) < 0.6, f"size {w}x{h} 差异过大"
