"""改写 ProviderRequest：用标注图替换原图、追加文字说明。"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from .config import Settings
from .pipeline import ImageOutcome

logger = logging.getLogger(__name__)

_HEADER = "[角色识别结果]"


def _color_word(color_name: str) -> str:
    """ "橙色" → "橙色框"。"""
    if color_name.endswith("色"):
        return f"{color_name}框"
    return f"{color_name} 色框"


def build_text(outcomes: Sequence[ImageOutcome], *, show_top5: bool) -> str:
    """按 §6.7 模板生成说明文本；无可注入内容返回空串。"""
    sections: list[str] = []
    for outcome in outcomes:
        if outcome.annotated_path is None or not outcome.persons:
            continue
        lines = [
            f"第{outcome.source_index + 1}张图检测到 {len(outcome.persons)} 个人物，"
            "已用彩色框与编号标注："
        ]
        for person in outcome.persons:
            if person.display_name is None:
                line = f"#{person.index} {_color_word(person.color_name)}：未识别（不在识别库内或把握不足）"
                if show_top5 and person.top5:
                    near = "、".join(
                        f"{name}({score:.2f})" for name, score in person.top5
                    )
                    line += f"；近邻候选（仅供参考）：{near}"
            else:
                line = (
                    f"#{person.index} {_color_word(person.color_name)}："
                    f"{person.display_name}（置信度 {person.head_conf:.2f}）"
                )
            lines.append(line)
        note = outcome.scope_note or "《鸣潮》可操控角色"
        lines.append(
            f"说明：识别库仅覆盖{note}，其他作品的人物会显示“未识别”；结果来自自动识别，可能有误。"
        )
        sections.append("\n".join(lines))
    return "\n".join(sections)


def apply(
    req: Any,
    outcomes: Sequence[ImageOutcome],
    get_settings: Callable[[], Settings],
) -> None:
    """替换已标注的原图并追加文字；全部未标注时不动 req。

    req 为 astrbot ProviderRequest（鸭子类型，保持 core 独立性由 main 传入）。
    """
    settings = get_settings()
    replaced = False
    for outcome in outcomes:
        if outcome.annotated_path is None:
            continue
        idx = outcome.source_index
        if 0 <= idx < len(req.image_urls):
            req.image_urls[idx] = str(outcome.annotated_path)
            replaced = True
    if not replaced:
        return
    text = build_text(outcomes, show_top5=settings.show_top5_for_unknown)
    if not text:
        return
    try:
        from astrbot.core.agent.message import TextPart
    except ImportError:  # 兜底：极老版本无 TextPart
        logger.warning("TextPart 不可用，跳过文字注入")
        return
    part = TextPart(text=text)
    if settings.temp_text:
        try:
            part.mark_as_temp()
        except AttributeError:  # 低版本无 mark_as_temp，忽略该选项
            logger.debug("当前 AstrBot 版本不支持 mark_as_temp，文字将写入历史")
    req.extra_user_content_parts.append(part)
