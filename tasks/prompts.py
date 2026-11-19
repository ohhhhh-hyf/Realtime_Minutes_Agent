"""会议纪要核心提示词构建（1:1 对齐 Cicada 源码哲学 Notes.swift）。

包含：
1. 2分钟节奏刹车指令（距当前议题起始不足 120 秒禁止新建议题）；
2. 滑动窗口冻结（Earlier topics settled: frozen > 0 时前置议题仅供参考，不重复返回）；
3. 议题语义判定法则（说话人轮替、观点交锋、细节深挖绝非新议题，宁归并勿分裂）；
4. 终盘定稿（final=True）上帝视角全景 8 议题重梳理；
5. points 与 details 专属分工（point为提炼点，details至多1条专属展开，严禁套话车轱辘话）。
"""

import json
from typing import List, Optional
from schemas.asr import TranscriptLine
from schemas.minutes import MeetingNotes


def clock(t: float) -> str:
    """时间戳格式化为 MM:SS 格式。"""
    total = int(max(0.0, t))
    minutes = (total % 3600) // 60
    seconds = total % 60
    return f"{minutes:02d}:{seconds:02d}"


SYSTEM_PROMPT = """你是一名资深会议纪要与商业分析专家。你的任务是将会议发言实时提炼为结构清晰、重点突出、逻辑严密的专业会议纪要。
请基于发言事实客观归纳，语言自然干练、拒绝空洞套话。严格按照要求的 JSON 结构输出，除纯 JSON 外不输出任何其它内容。

请严格返回符合以下结构的唯一 JSON 对象：
{
  "title": "会议主题：用精炼的词概括会议主题（10~15字，类似日历项，不要带引号，不单独叫'会议'）",
  "summary": "整场会议至今的2-3句全局概述，客观精炼",
  "topics": [
    {
      "start": 10.0,
      "title": "议题简短标题",
      "summary": "一句话概括这个议题在讨论什么（40字以内）",
      "points": [
        {
          "point": "一句简短完整的话，概括该分支的核心实质论点或定论（每个议题通常2~6个，深水区深度讨论最高弹性放宽至10个，拒绝短语与零碎切片）",
          "details": ["该点专属支撑的具体数据、参数或落地实操（至多1条，无更深细节填[]）"]
        }
      ],
      "consensus": null,
      "evidence": ["L1", "L2"],
      "fishbone": null
    }
  ],
  "conclusions": [
    {
      "text": "明确拍板的定论（最多5条，通常没有则填[]）",
      "t": 40.0,
      "evidence": ["L3"]
    }
  ],
  "todos": [
    {
      "text": "具体的真实待办（通常没有则填[]；Only real to-dos, often none）",
      "owner": "负责人姓名或你/对方",
      "deadline": null,
      "t": 55.0,
      "evidence": ["L4"]
    }
  ],
  "terms": [
    {
      "name": "正确拼写的术语或人名",
      "note": "简短释义",
      "type": "jargon",
      "heard": ["被误识别的错误写法"]
    }
  ]
}

提炼规则（必须严格遵守）：
- title: 整场会议的名称，类似日历项（10~15字；不含日期，不能单独使用“会议”二字）。除非主题明显转移，否则保持当前标题。
- summary: 最多3句简明短句，客观概括全场背景与核心走向，切忌套话注水。
- topics: 讨论议题按时间顺序排列：
  1. 粒度与承接：一个议题是 3~5 分钟的大板块讨论，绝非零星发言。增量更新时，若最新发言仍在探讨当前活跃议题，请保留该议题已有的 points 并在其后继续扩充要点；仅在讨论方向出现明显改变或主持人明确推进下一项议程时，才开启新议题。
  2. 增量节奏：每次更新至多新增 1 个议题；最后一个议题始终是当前正在讨论的活跃议题。
  3. 议题摘要：每个议题的 summary 必须是对该议题全貌的一句话宏观概括（50字以内），随讨论推进动态重写，严禁以时间轴日志形式机械追加（严禁出现“最新讨论了…”、“随后又补充…”、“接着谈到…”）。
- points 与 details 的提炼法度（严禁过度切分碎拆）：
  1. point：
     - 粒度与属性：一个 Point 是针对当前议题展开的一个【独立实质论点、核心研判或定论脉络】。数量规范：常规议题提炼 2~6 个点，遇核心深水区大议题最高允许弹性放宽至 10 个点；必须是一句简短完整的话，表意明确，拒绝几个字的短语标签，宁归并勿重复。
     - 宁聚拢勿零碎：同一维度下的多句连续表述、因果逻辑或事实支撑，必须高度聚合成一个完整 Point，严禁对一段发言逐句切片。
     - 严禁单一数据独占：金额、百分比、人数等具体量化指标绝不能单独起 Point，必须融入结论句中或下沉至 details。
     - 严禁过程流水账：寒暄、铺垫、过渡句或“某某发言说明了XX”不是要点。
  2. details：
     - details 是当前 point 专属的【数据支撑与实操落地】（至多 1 条），专门用于收纳该点涉及的关键指标、技术参数或执行动作。
     - 若 point 本身已阐述清晰且无关键指标/落地细节，details 坚决填 []；严禁换个句式车轱辘话重复复述 Point。
- conclusions: 最多 5 条结论，必须是现场明确拍板的定论（通常没有或很少，没有填 []）。
- todos: 仅记录真实确凿的待办（通常没有，没有填 []；Only real to-dos, often none）。必须提取负责人；若提及明确截止时间提取至 deadline，未提及填 null。
- terms: 专有名词表，标记 type 为 person（人名）、jargon（专业术语）或 project（系统/项目）。
- evidence: 每一个 topic、point、conclusion、todo 都必须标注对应的原话行号（如 ["L2", "L3"]），严禁编造发言记录中没有的事实。
- fishbone: 仅当议题集中讨论根本原因、故障排查或影响因素时才生成；必须包含 question（探讨的核心问题）与 branches（分支列表）。每个分支必须使用 label（2~6字类别）与 detail（具体单行原因阐述）。若无需深入原因分析则置为 null。
- 输出格式: 仅输出纯 JSON，不要包含任何 markdown 代码块标记，不要输出任何非 JSON 字符。"""


def build_user_prompt(
    current_notes: MeetingNotes,
    transcript_lines: List[TranscriptLine],
    from_line: int = 0,
    frozen: int = 0,
    final: bool = False,
) -> str:
    """构建用户输入 Prompt（支持滑动窗口冻结、2分钟冷却刹车与终盘重构）。"""
    total_lines = len(transcript_lines)
    from_idx = min(from_line, total_lines)
    context_idx = max(0, from_idx - 2)

    prompt_parts: List[str] = []

    # 1. 任务背景说明
    if final:
        task_desc = "你负责为一场【刚刚结束的会议】做纪要。请根据整场会议的全部逐字稿，重新梳理并编写完整定稿。"
    else:
        task_desc = "你负责为一场【正在进行中的会议】做实时纪要。请结合最新发言记录更新纪要。"
    prompt_parts.append(f"{task_desc}\n请用中文书写，仅返回纯 JSON 格式。\n")

    # 2. 会中实时更新：前序已定案议题（Frozen 区域，模型无需返回）
    if not final and frozen > 0 and len(current_notes.topics) >= frozen:
        frozen_list = "\n".join([
            f"- {clock(t.start)} {t.title}: {t.summary}"
            for t in current_notes.topics[:frozen]
        ])
        prompt_parts.append(
            f"Earlier topics (已定案历史议题：已锁定保持原样，切勿在返回的 JSON 中输出它们):\n"
            f"{frozen_list}\n"
        )

    # 3. 当前活跃纪要上下文（仅传可编辑部分）
    if not final and not current_notes.is_empty:
        active_notes = current_notes.model_copy(deep=True)
        if frozen > 0:
            active_notes.topics = current_notes.topics[frozen:]
        active_json = active_notes.model_dump_json(
            exclude={"covered", "updated_at", "final"},
            by_alias=True,
        )
        prompt_parts.append(
            f"Current notes (当前活跃纪要 JSON{'，仅含最近可编辑议题' if frozen > 0 else ''}):\n"
            f"{active_json}\n"
        )

    # 4. 上下文原话（已覆盖，仅供参考连贯性）
    if not final and context_idx < from_idx:
        context_text = "\n".join([
            transcript_lines[i].to_display_string()
            for i in range(context_idx, from_idx)
        ])
        prompt_parts.append(f"Already covered (已覆盖上下文，仅供参考):\n{context_text}\n")

    # 5. 最新待处理发言记录
    lines_header = "Transcript (整场会议逐字稿)" if final else "New lines (最新发言记录)"
    lines_slice = transcript_lines[from_idx:] if not final else transcript_lines
    lines_text = "\n".join([line.to_display_string() for line in lines_slice])
    prompt_parts.append(f"{lines_header} (id, mm:ss, 说话人, 内容):\n{lines_text}")

    # 6. 议题节奏与刹车约束指令（1:1 对齐 Notes.swift 核心机理）
    if final:
        if transcript_lines:
            first_t = transcript_lines[0].t
            last_t = transcript_lines[-1].t
            span = max(0.0, last_t - first_t)
            most_topics = max(1, min(8, int(span / 180.0) + 1))
            prompt_parts.append(
                f"\n本场会议共进行 {clock(span)}：全场请梳理出【至多 {most_topics} 个议题】（use at most {most_topics} topics）。"
                f"请宏观统揽全局，提炼全局标题、全场总结、定论与真实待办。"
            )
    else:
        # 会中检查当前最后活跃议题的时间跨度与要点饱和度（弹性10点上限熔断与下限保护）
        if current_notes.topics and transcript_lines:
            current_topic = current_notes.topics[-1]
            pts_count = len(current_topic.points)
            newest_t = transcript_lines[-1].t
            span = newest_t - current_topic.start
            if pts_count >= 10:
                prompt_parts.append(
                    f"\n【议题容量已达上限警告】当前活跃议题“{current_topic.title}”已积累 {pts_count} 个要点，已达最大上限！"
                    f"严禁再向其 points 列表中追加任何新要点！请结合最新发言归纳并开启一个全新的议题（New Topic）！"
                )
            elif span < 120.0 or pts_count < 2:
                prompt_parts.append(
                    f"\n当前活跃议题“{current_topic.title}”始于 {clock(current_topic.start)}（已讨论 {clock(span)}，现有 {pts_count} 个要点）："
                    f"讨论尚在展开阶段，若未发生明确的议程转换，请优先在该议题的 points 列表中扩充新要点，保持议题饱满连贯。"
                )
        elif not current_notes.topics:
            prompt_parts.append("\n这是会议开场的首批发言：请归纳为一个起始议题。")

    prompt_parts.append("\n请严格输出符合格式的纯 JSON 对象：")
    return "\n".join(prompt_parts)
