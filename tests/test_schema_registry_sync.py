"""schema 与 registry 一致性测试：models.*.options、det_conf 默认值、文件配对规则。"""

from __future__ import annotations

import json
from pathlib import Path, PurePath

from core.registry import load_registry

PLUG_ROOT = Path(__file__).resolve().parents[1]


def _schema():
    return json.loads((PLUG_ROOT / "_conf_schema.json").read_text(encoding="utf-8"))


def _registry():
    return load_registry(PLUG_ROOT / "data" / "registry.json")


def test_schema_detector_options_match_registry():
    schema = _schema()
    registry = _registry()
    schema_opts = schema["models"]["items"]["detector"]["options"]
    reg_keys = [k for k, spec in registry.specs.items() if spec.kind == "detector"]
    assert list(schema_opts) == reg_keys


def test_schema_classifier_options_match_registry():
    schema = _schema()
    registry = _registry()
    schema_opts = schema["models"]["items"]["classifier"]["options"]
    reg_keys = [k for k, spec in registry.specs.items() if spec.kind == "classifier"]
    assert list(schema_opts) == reg_keys


def test_schema_det_conf_default_is_0_327():
    schema = _schema()
    assert schema["detect"]["items"]["det_conf"]["default"] == 0.327


def test_classifier_files_pairing_rule():
    """model 与 prototypes 同目录，且按 `*_int8.onnx → prototypes_int8.npz` 后缀配对。"""
    registry = _registry()
    for key, spec in registry.specs.items():
        if spec.kind != "classifier":
            continue
        model = PurePath(spec.files["model"])
        proto = PurePath(spec.files["prototypes"])
        assert model.parent == proto.parent, f"{key}: 目录不一致"
        assert model.name.endswith(".onnx")
        if model.name.endswith("_int8.onnx"):
            assert proto.name == "prototypes_int8.npz"
        else:
            assert proto.name == "prototypes.npz"
