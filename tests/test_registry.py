"""registry 模块测试：真实 registry 加载、非法条目、展示名映射（YAML / 按行回落）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from core.registry import RegistryError, load_display_names, load_registry

PLUG_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = PLUG_ROOT / "data" / "registry.json"


@pytest.fixture
def registry():
    return load_registry(REGISTRY)


def test_load_real_registry(registry):
    assert len(registry.specs) == 3
    det = registry.detector("person_detect_v1.1_n")
    assert det.kind == "detector"
    assert det.ref == "main"
    assert "model" in det.files
    assert det.display_names is None
    assert det.scope_note == ""
    for key in ("wuwa_mnv4l_448_int8", "wuwa_mnv4s_384_fp32"):
        spec = registry.classifier(key)
        assert spec.kind == "classifier"
        assert "model" in spec.files and "prototypes" in spec.files
        assert spec.display_names == "class_names_zh.yaml"
        assert spec.scope_note == "《鸣潮》可操控角色"


def test_get_missing_raises(registry):
    with pytest.raises(RegistryError):
        registry.get("no_such_key")
    # kind 校验：classifier 条目不接受 detector() 查询
    with pytest.raises(RegistryError):
        registry.detector("wuwa_mnv4l_448_int8")
    with pytest.raises(RegistryError):
        registry.classifier("person_detect_v1.1_n")


def test_bad_json(tmp_path):
    p = tmp_path / "r.json"
    p.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(RegistryError):
        load_registry(p)


def _write_registry(tmp_path, data: dict) -> Path:
    p = tmp_path / "r.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_missing_model_file(tmp_path):
    p = _write_registry(
        tmp_path,
        {"k": {"kind": "detector", "repo_id": "a/b", "files": {"prototypes": "p.npz"}}},
    )
    with pytest.raises(RegistryError):
        load_registry(p)


def test_classifier_missing_prototypes(tmp_path):
    p = _write_registry(
        tmp_path,
        {"k": {"kind": "classifier", "repo_id": "a/b", "files": {"model": "m.onnx"}}},
    )
    with pytest.raises(RegistryError):
        load_registry(p)


def test_illegal_kind(tmp_path):
    p = _write_registry(
        tmp_path,
        {"k": {"kind": "banana", "repo_id": "a/b", "files": {"model": "m.onnx"}}},
    )
    with pytest.raises(RegistryError):
        load_registry(p)


def test_repo_id_must_contain_slash(tmp_path):
    p = _write_registry(
        tmp_path,
        {"k": {"kind": "detector", "repo_id": "norepo", "files": {"model": "m.onnx"}}},
    )
    with pytest.raises(RegistryError):
        load_registry(p)


def test_display_names_normal_yaml(tmp_path):
    p = tmp_path / "names.yaml"
    p.write_text(
        'jinhsi_(wuthering_waves): "今汐"\n_negative: "未知角色"\n# 注释行\n',
        encoding="utf-8",
    )
    names = load_display_names(p)
    assert names["jinhsi_(wuthering_waves)"] == "今汐"
    # _negative 键原样保留
    assert "_negative" in names
    assert names["_negative"] == "未知角色"


def test_display_names_line_parse_fallback(tmp_path, monkeypatch):
    """使 import yaml 失败，验证按行解析回落路径。"""
    p = tmp_path / "names.yaml"
    p.write_text(
        'jinhsi_(wuthering_waves): "今汐"\n_negative: 未知角色\n# 注释\n   padded:  值 \n',
        encoding="utf-8",
    )
    monkeypatch.setitem(sys.modules, "yaml", None)
    names = load_display_names(p)
    assert names["jinhsi_(wuthering_waves)"] == "今汐"
    assert names["_negative"] == "未知角色"
    assert names["padded"] == "值"


def test_display_names_missing_file(tmp_path):
    with pytest.raises(RegistryError):
        load_display_names(tmp_path / "no_such.yaml")
