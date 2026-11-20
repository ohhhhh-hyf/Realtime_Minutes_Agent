"""结构化会议纪要强类型数据契约模型（具备强容错与智能字段映射能力）。"""

from typing import List, Literal, Optional, Any, Dict
from pydantic import BaseModel, Field, model_validator, field_validator


class PointItem(BaseModel):
    """议题下的单个论点项（论据一体化，彻底拔除 details，1:1 对齐 Cicada 原生架构）。"""

    text: str = Field(..., description="论据一体化核心要点（完整断言，融入关键量化事实）")
    evidence: List[str] = Field(default_factory=list, description="支撑原话行号")

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"text": data, "evidence": []}
        if isinstance(data, dict):
            # 兼容 text / point / content / title
            text_val = data.get("text") or data.get("point") or data.get("content") or data.get("title") or "论点阐述"
            ev = data.get("evidence") or []
            if isinstance(ev, str):
                ev = [ev] if ev else []
            return {"text": str(text_val).strip(), "evidence": ev}
        return data

    @property
    def point(self) -> str:
        """兼容已有代码与测试的 .point 属性访问。"""
        return self.text

    def __str__(self) -> str:
        return self.text


class FishboneBranch(BaseModel):
    """鱼骨图分析单个分支。"""

    label: str = Field(default="原因分类", description="分支类别标签，如人员、工具、流程等")
    detail: str = Field(default="", description="分支事实明细")
    support: int = Field(default=0, description="支撑度")
    t: Optional[float] = Field(default=None, description="对应时间戳秒数")
    evidence: List[str] = Field(default_factory=list, description="证据原话行号")

    @model_validator(mode="before")
    @classmethod
    def normalize_branch(cls, data: Any) -> Any:
        # 支持纯字符串分支，如 "人员管理：沟通不及时"
        if isinstance(data, str):
            s = data.strip()
            if "：" in s:
                parts = s.split("：", 1)
                return {"label": parts[0].strip(), "detail": parts[1].strip()}
            elif ":" in s:
                parts = s.split(":", 1)
                return {"label": parts[0].strip(), "detail": parts[1].strip()}
            return {"label": "原因分析", "detail": s}

        if isinstance(data, dict):
            # 智能映射：兼容 LLM 常见的 category, tag, dimension 等作为 label
            if "label" not in data or not data["label"]:
                for key in ("category", "tag", "name", "dimension", "title", "branch"):
                    if key in data and data[key]:
                        data["label"] = str(data[key]).strip()
                        break
                if "label" not in data or not data["label"]:
                    data["label"] = "原因分类"

            # 智能映射：兼容 LLM 常见的 cause, reason, content 等作为 detail
            if "detail" not in data or not data["detail"]:
                for key in ("cause", "reason", "content", "text", "description", "point"):
                    if key in data and data[key]:
                        data["detail"] = str(data[key]).strip()
                        break
                if "detail" not in data:
                    data["detail"] = ""
        return data


class Fishbone(BaseModel):
    """鱼骨图根问题分析。"""

    question: str = Field(default="核心问题原因分析", description="探讨的核心问题")
    branches: List[FishboneBranch] = Field(default_factory=list, description="分支原因分析")

    @model_validator(mode="before")
    @classmethod
    def normalize_fishbone(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # 智能映射核心问题字段
            if "question" not in data or not data["question"]:
                for key in ("problem", "issue", "core_problem", "topic", "title"):
                    if key in data and data[key]:
                        data["question"] = str(data[key]).strip()
                        break
                if "question" not in data or not data["question"]:
                    data["question"] = "核心问题探讨"
            # 兼容 branches 为 None 的情况
            if "branches" not in data or data["branches"] is None:
                data["branches"] = []
        return data


class NoteTopic(BaseModel):
    """单个议题结构（1:1 对齐 Cicada 原生架构，论据一体化，彻底拔除 details）。"""

    start: float = Field(default=0.0, description="议题起始时间戳秒数")
    title: str = Field(default="", description="议题简短标题")
    summary: str = Field(default="", description="该议题1句话精炼总结")
    points: List[PointItem] = Field(default_factory=list, description="议题下的论点列表（论据一体化）")
    consensus: Optional[str] = Field(default=None, description="现场达成的局部共识")
    evidence: List[str] = Field(default_factory=list, description="议题支撑原话行号")
    fishbone: Optional[Fishbone] = Field(default=None, description="原因分析鱼骨图")

    @field_validator("fishbone", mode="before")
    @classmethod
    def sanitize_fishbone(cls, v: Any) -> Any:
        if not v or not isinstance(v, dict):
            return None
        try:
            return Fishbone.model_validate(v)
        except Exception:
            # 兜底：如果鱼骨图无法解析，降级置空，绝不阻断整篇纪要的生成
            return None


class NoteConclusion(BaseModel):
    """结论项（明确拍板定案的定论，1:1 对齐源码无多余分类）。"""

    text: str = Field(..., description="明确拍板定案的定论描述")
    t: Optional[float] = Field(default=None, description="出现时间戳秒数")
    evidence: List[str] = Field(default_factory=list, description="证据原话行号")

    @model_validator(mode="before")
    @classmethod
    def normalize_conclusion(cls, data: Any) -> Any:
        # 支持纯字符串输入
        if isinstance(data, str):
            return {"text": data}
        if isinstance(data, dict):
            # 兼容模型误返回 title 或 content 代替 text
            if "text" not in data or not data["text"]:
                data["text"] = data.get("content") or data.get("title") or "定论事实"
        return data


class NoteTodo(BaseModel):
    """待办事项（支持 deadline 截止时间提取）。"""

    text: str = Field(..., description="具体的待办任务描述")
    owner: Optional[str] = Field(default=None, description="负责人姓名或标识")
    deadline: Optional[str] = Field(
        default=None,
        description="明确的时间截止点（如本周五前），发言未提及则为 None",
    )
    t: Optional[float] = Field(default=None, description="出现时间戳")
    evidence: List[str] = Field(default_factory=list, description="证据原话行号")

    @model_validator(mode="before")
    @classmethod
    def normalize_todo(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # 兼容负责人字段别名
            if "owner" not in data or not data["owner"]:
                for key in ("assignee", "person", "who", "responsible"):
                    if key in data and data[key]:
                        data["owner"] = str(data[key]).strip()
                        break
            # 兼容截止时间字段别名
            if "deadline" not in data or not data["deadline"]:
                for key in ("due", "due_date", "time", "ddl"):
                    if key in data and data[key]:
                        data["deadline"] = str(data[key]).strip()
                        break
        return data


class NoteTerm(BaseModel):
    """专有名词或术语。"""

    name: str = Field(..., description="正确拼写的术语、人名或系统名")
    note: str = Field(default="", description="简短释义或说明")
    type: Literal["person", "jargon", "project"] = Field(
        default="jargon",
        description="术语分类：person(人名) | jargon(专业术语) | project(系统/项目)",
    )
    heard: List[str] = Field(default_factory=list, description="曾被误识别的错误写法")


class MeetingNotes(BaseModel):
    """会议全量结构化纪要模型（顶级 6 字段规范）。"""

    title: Optional[str] = Field(default="实时会议纪要", description="会议主题（10~15字）")
    summary: str = Field(default="", description="全场自然总结（2~3句）")
    topics: List[NoteTopic] = Field(default_factory=list, description="讨论议题列表")
    conclusions: List[NoteConclusion] = Field(
        default_factory=list,
        description="定论、风险与待定列表",
    )
    todos: List[NoteTodo] = Field(default_factory=list, description="待办任务列表")
    terms: List[NoteTerm] = Field(default_factory=list, description="专有名词列表")
    covered: int = Field(default=0, description="当前纪要覆盖的最新发言行序号")
    updated_at: Optional[str] = Field(default=None, description="最新更新时间ISO字符串")
    final: bool = Field(default=False, description="是否为会议结束后的终盘定稿")

    @property
    def is_empty(self) -> bool:
        return (
            not self.summary
            and len(self.topics) == 0
            and len(self.conclusions) == 0
            and len(self.todos) == 0
        )
