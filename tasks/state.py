"""LangGraph 状态机 State 契约定义。"""

from typing import List, Optional, TypedDict
from schemas.asr import TranscriptLine
from schemas.minutes import MeetingNotes


class MeetingGraphState(TypedDict, total=False):
    """会议纪要工作流图状态。"""

    transcript: List[TranscriptLine]
    notes: MeetingNotes
    last_covered_line: int
    force_update: bool
    should_update: bool
    final: bool
    error: Optional[str]
