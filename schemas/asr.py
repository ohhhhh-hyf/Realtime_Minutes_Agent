"""ASR 输入数据契约模型与格式化工具。"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ASRLineInput(BaseModel):
    """上游 ASR 推送的单行转写数据模型（严格 4 字段输入校验）。"""

    line_id: Optional[int] = Field(
        default=None,
        description="发言行号，未传时由系统自增分配",
    )
    t: float = Field(
        ...,
        ge=0.0,
        description="音频起始相对时间（绝对秒数），必须大于等于 0.0",
    )
    speaker: str = Field(
        ...,
        min_length=1,
        description="发言人姓名或标识，不能为空",
    )
    text: str = Field(
        ...,
        min_length=1,
        description="发言原话转写文本，不能为空",
    )

    @field_validator("speaker", "text")
    @classmethod
    def strip_and_check_non_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("字段值不能为空白字符")
        return s


class TranscriptLine(BaseModel):
    """系统内部规范化流水行。"""

    id: str = Field(..., description="证据链标识，形如 'L1', 'L2'")
    line_id: int = Field(..., description="行序号，从 1 开始递增")
    t: float = Field(..., description="起始秒数")
    speaker: str = Field(..., description="发言人标识")
    text: str = Field(..., description="发言原话内容")

    @classmethod
    def from_input(cls, item: ASRLineInput, assigned_id: int) -> "TranscriptLine":
        line_num = item.line_id if item.line_id is not None else assigned_id
        return cls(
            id=f"L{line_num}",
            line_id=line_num,
            t=round(item.t, 2),
            speaker=item.speaker,
            text=item.text,
        )

    def format_timestamp(self) -> str:
        """格式化为 MM:SS 或 HH:MM:SS"""
        total_seconds = int(self.t)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def to_display_string(self) -> str:
        """用于前端展示的纯文本格式，如：[L1 00:10 张三] 原话内容"""
        return f"[{self.id} {self.format_timestamp()} {self.speaker}] {self.text}"

    def to_prompt_line(self) -> str:
        """注入 LLM Prompt 的发言行表示，如：L1 (00:10, 张三): 原话内容"""
        return f"{self.id} ({self.format_timestamp()}, {self.speaker}): {self.text}"
