"""LangGraph 状态图构建与编译（1:1 对齐 Cicada Notes.swift 调度机理）。"""

import json
import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from schemas.minutes import MeetingNotes
from core.llm import get_chat_model
from core.config import get_settings
from core.compactor import NotesCompactor
from .state import MeetingGraphState
from .prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


def evaluate_trigger(state: MeetingGraphState) -> Dict[str, Any]:
    """判断是否需要触发 LLM 纪要提炼。"""
    settings = get_settings()
    total_lines = len(state.get("transcript", []))
    covered = state.get("last_covered_line", 0)
    uncovered = total_lines - covered
    force = state.get("force_update", False)
    final = state.get("final", False)

    # 触发条件：终盘定稿，强制更新，首次有至少 2 行发言，或未处理行数达到批量阈值
    should = (
        final
        or force
        or (covered == 0 and total_lines >= 2)
        or (uncovered >= settings.TRIGGER_LINE_BATCH)
    )

    return {"should_update": should}


def route_decision(state: MeetingGraphState) -> str:
    """路由决策。"""
    if state.get("should_update", False):
        return "generate_notes"
    return END


async def generate_notes(state: MeetingGraphState) -> Dict[str, Any]:
    """调用 LLM 提取会议纪要并融合入库（支持会中滑动窗口与终盘定稿）。"""
    transcript = state.get("transcript", [])
    old_notes = state.get("notes") or MeetingNotes()
    final = state.get("final", False)
    settings = get_settings()

    if not transcript:
        return {"notes": old_notes, "should_update": False, "error": None}

    # 构建行号 -> 时间戳索引表
    evidence_times = {line.id: line.t for line in transcript}

    # 1. 确定滑动窗口边界：会中只编辑尾部 editableTopics (2) 个议题，更早的冻结
    from_line = 0 if final else min(old_notes.covered, len(transcript))
    frozen = 0 if final else max(0, len(old_notes.topics) - settings.EDITABLE_TOPICS)

    # 2. 构建提示词（注入刹车或终盘全局指令）
    user_prompt = build_user_prompt(
        current_notes=old_notes,
        transcript_lines=transcript,
        from_line=from_line,
        frozen=frozen,
        final=final,
    )

    try:
        llm = get_chat_model()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        response = await llm.ainvoke(messages)
        content = response.content.strip()

        # 剔除可能存在的 markdown 代码块包裹
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        # 校验并解析为 Pydantic 对象
        new_data = json.loads(content)
        try:
            parsed_notes = MeetingNotes.model_validate(new_data)
        except Exception as val_err:
            logger.warning(f"首次模型校验告警，执行容错降级清洗: {val_err}")
            if isinstance(new_data, dict):
                # 如果是 fishbone 引起的校验失败，置空 fishbone 并重试，确保核心纪要内容不丢失
                for topic in new_data.get("topics", []):
                    if isinstance(topic, dict) and "fishbone" in topic:
                        topic["fishbone"] = None
            parsed_notes = MeetingNotes.model_validate(new_data)

        # 3. 状态脱水、时间戳保护与议题容量缝合（1:1 对齐 Notes.swift）
        compacted_notes = NotesCompactor.compact(
            old_notes=old_notes,
            next_notes=parsed_notes,
            evidence_times=evidence_times,
            covered_line_num=len(transcript),
            frozen=frozen,
            final=final,
            live_max_topics=settings.LIVE_MAX_TOPICS,
            max_topics=settings.MAX_TOPICS,
        )

        return {
            "notes": compacted_notes or old_notes,
            "last_covered_line": len(transcript),
            "should_update": False,
            "final": final,
            "error": None,
        }
    except Exception as e:
        logger.error(f"生成纪要失败: {str(e)}", exc_info=True)
        return {
            "notes": old_notes,
            "should_update": False,
            "final": final,
            "error": f"LLM 提炼异常: {str(e)}",
        }


def create_meeting_graph():
    """编译 LangGraph 工作流图。"""
    workflow = StateGraph(MeetingGraphState)

    workflow.add_node("evaluate_trigger", evaluate_trigger)
    workflow.add_node("generate_notes", generate_notes)

    workflow.set_entry_point("evaluate_trigger")

    workflow.add_conditional_edges(
        "evaluate_trigger",
        route_decision,
        {
            "generate_notes": "generate_notes",
            END: END,
        },
    )

    workflow.add_edge("generate_notes", END)

    return workflow.compile()
