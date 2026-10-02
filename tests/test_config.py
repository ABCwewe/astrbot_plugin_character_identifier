"""config 模块测试：默认值、类型非法回落、数值钳制、字符串整理、列表 tuple 化。"""

from __future__ import annotations

from core.config import Settings, load_settings


def test_defaults():
    s = load_settings({})
    assert isinstance(s, Settings)
    assert s.enabled is True
    assert s.detector_key == "person_detect_v1.1_n"
    assert s.classifier_key == "wuwa_mnv4l_448_int8"
    assert s.preprocess_backend == "opencv"
    assert s.hf_endpoint == "https://huggingface.co"
    assert s.offline_mode is False
    assert s.resident is False
    assert s.idle_timeout_sec == 300
    assert s.threads == 2
    assert s.max_concurrency == 1
    assert s.low_memory is True
    assert s.infer_timeout == 15
    assert s.det_imgsz == 640
    assert s.det_conf == 0.327  # 契约锚点
    assert s.nms_iou == 0.5
    assert s.max_persons == 8
    assert s.min_box_px == 24
    assert s.det_pre_max_side == 2048
    assert s.label_mode == "index"
    assert s.font_path == ""
    assert s.out_max_side == 1280
    assert s.max_images == 3
    assert s.temp_text is False
    assert s.show_top5_for_unknown is False
    assert s.cache_keep_days == 7
    assert s.session_whitelist == ()
    assert s.session_blacklist == ()


def test_type_illegal_falls_back_to_default():
    s = load_settings(
        {
            "enabled": "yes",  # 非 bool → 回落 True
            "models": {
                "detector": 123,
                "hf_endpoint": 42,
                "preprocess_backend": "webp",
            },
            "runtime": {"threads": "many", "resident": "on"},
            "detect": {"det_conf": "high", "det_imgsz": "big", "nms_iou": None},
            "annotate": {"label_mode": "emoji", "out_max_side": "huge"},
        }
    )
    assert s.enabled is True
    assert s.detector_key == "person_detect_v1.1_n"
    assert s.hf_endpoint == "https://huggingface.co"
    assert s.preprocess_backend == "opencv"
    assert s.threads == 2
    assert s.resident is False
    assert s.det_conf == 0.327
    assert s.det_imgsz == 640
    assert s.nms_iou == 0.5
    assert s.label_mode == "index"
    assert s.out_max_side == 1280


def test_numeric_out_of_range_clamped():
    s = load_settings(
        {
            "runtime": {"threads": 99, "max_concurrency": 9, "idle_timeout_sec": 1},
            "detect": {
                "det_conf": 5,
                "nms_iou": -1,
                "max_persons": 99,
                "min_box_px": 1,
            },
        }
    )
    assert s.threads == 8
    assert s.max_concurrency == 4
    assert s.idle_timeout_sec == 30
    assert s.det_conf == 1.0
    assert s.nms_iou == 0.1
    assert s.max_persons == 16
    assert s.min_box_px == 8

    s = load_settings(
        {
            "runtime": {"threads": 0, "infer_timeout": 1},
            "detect": {"det_conf": -1, "max_persons": 0},
        }
    )
    assert s.threads == 1
    assert s.infer_timeout == 5
    assert s.det_conf == 0.0
    assert s.max_persons == 1


def test_det_imgsz_unsupported_falls_back_640():
    for bad in (333, 641, 319):
        s = load_settings({"detect": {"det_imgsz": bad}})
        assert s.det_imgsz == 640
    s = load_settings({"detect": {"det_imgsz": 512}})
    assert s.det_imgsz == 512


def test_hf_endpoint_trailing_slash_removed():
    s = load_settings({"models": {"hf_endpoint": "https://hf-mirror.com/"}})
    assert s.hf_endpoint == "https://hf-mirror.com"
    s = load_settings({"models": {"hf_endpoint": "https://hf-mirror.com///"}})
    assert s.hf_endpoint == "https://hf-mirror.com"
    # 空值回落默认
    s = load_settings({"models": {"hf_endpoint": None}})
    assert s.hf_endpoint == "https://huggingface.co"


def test_session_lists_tuple_ized():
    s = load_settings(
        {
            "scope": {
                "session_whitelist": ["a", "b", 3],
                "session_blacklist": ["c"],
            }
        }
    )
    assert s.session_whitelist == ("a", "b", "3")
    assert s.session_blacklist == ("c",)
    # 缺省为空 tuple
    s = load_settings({})
    assert s.session_whitelist == ()
    assert s.session_blacklist == ()
