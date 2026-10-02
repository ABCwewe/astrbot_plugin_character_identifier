"""injector 模块测试：build_text 模板、apply 替换语义、temp_text 标记。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from core.config import Settings
from core.injector import apply, build_text
from core.pipeline import ImageOutcome, PersonLine


def _outcome(
    source_index=0, annotated=True, persons=None, scope_note="《鸣潮》可操控角色"
):
    if persons is None:
        persons = [
            PersonLine(index=1, color_name="橙色", display_name="今汐", head_conf=0.93)
        ]
    return ImageOutcome(
        source_index=source_index,
        sha1="a" * 40,
        annotated_path=Path("x.jpg") if annotated else None,
        persons=persons,
        scope_note=scope_note,
    )


def test_build_text_line_format():
    outcomes = [
        _outcome(
            source_index=0,
            persons=[
                PersonLine(
                    index=1, color_name="橙色", display_name="今汐", head_conf=0.93
                ),
                PersonLine(
                    index=2, color_name="灰色", display_name=None, head_conf=None
                ),
            ],
        ),
        _outcome(source_index=1, annotated=False),  # 未标注 → 不进说明
    ]
    text = build_text(outcomes, show_top5=False)
    lines = text.splitlines()
    assert lines[0] == "第1张图检测到 2 个人物，已用彩色框与编号标注："
    assert "#1 橙色框：今汐（置信度 0.93）" in lines
    assert "#2 灰色框：未识别（不在识别库内或把握不足）" in lines
    assert "说明：识别库仅覆盖《鸣潮》可操控角色" in text
    # 只有被替换的图（source_index 0）进入说明
    assert "第2张图" not in text


def test_build_text_scope_note_default_when_empty():
    outcome = _outcome(scope_note="")
    text = build_text([outcome], show_top5=False)
    assert "说明：识别库仅覆盖《鸣潮》可操控角色" in text


def test_build_text_top5_for_unknown():
    outcome = _outcome(
        persons=[
            PersonLine(
                index=1,
                color_name="灰色",
                display_name=None,
                head_conf=None,
                top5=(("今汐", 0.35), ("忌炎", 0.28)),
            )
        ]
    )
    text = build_text([outcome], show_top5=True)
    assert "近邻候选（仅供参考）：今汐(0.35)、忌炎(0.28)" in text
    # show_top5=False 时不出现近邻候选
    text2 = build_text([outcome], show_top5=False)
    assert "近邻候选" not in text2


def test_build_text_no_content_returns_empty():
    assert build_text([], show_top5=False) == ""
    assert build_text([_outcome(annotated=False)], show_top5=False) == ""
    assert build_text([_outcome(persons=[])], show_top5=False) == ""


def test_apply_replaces_only_annotated():
    outcomes = [
        _outcome(source_index=0, annotated=True),
        _outcome(source_index=1, annotated=False),
        _outcome(source_index=2, annotated=True),
    ]
    req = SimpleNamespace(image_urls=["u0", "u1", "u2"], extra_user_content_parts=[])
    apply(req, outcomes, lambda: Settings(temp_text=False))
    assert req.image_urls[0] == "x.jpg"
    assert req.image_urls[1] == "u1"
    assert req.image_urls[2] == "x.jpg"
    assert len(req.extra_user_content_parts) == 1
    assert req.extra_user_content_parts[0].text.startswith("第1张图")


def test_apply_all_none_no_growth():
    outcomes = [_outcome(annotated=False), _outcome(annotated=False)]
    req = SimpleNamespace(image_urls=["u0", "u1"], extra_user_content_parts=[])
    apply(req, outcomes, lambda: Settings(temp_text=False))
    assert req.image_urls == ["u0", "u1"]
    assert req.extra_user_content_parts == []


def test_apply_temp_text_marks_no_save():
    outcomes = [_outcome(source_index=0, annotated=True)]
    req = SimpleNamespace(image_urls=["u0"], extra_user_content_parts=[])
    apply(req, outcomes, lambda: Settings(temp_text=True))
    assert len(req.extra_user_content_parts) == 1
    part = req.extra_user_content_parts[0]
    assert part._no_save is True


def test_apply_non_temp_does_not_mark():
    outcomes = [_outcome(source_index=0, annotated=True)]
    req = SimpleNamespace(image_urls=["u0"], extra_user_content_parts=[])
    apply(req, outcomes, lambda: Settings(temp_text=False))
    assert req.extra_user_content_parts[0]._no_save is False
