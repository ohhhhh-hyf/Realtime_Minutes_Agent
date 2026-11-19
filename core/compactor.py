"""状态脱水机与多轮融合器（1:1 对齐 Cicada 源码级算法 Notes.swift）。

核心职能：
1. carry_times：时间戳锁定，防止多轮提取后结论、待办与议题的时间戳后移漂移；
2. merge_topics：滑动窗口冻结（frozen）、尾部增量重写、最窄相邻议题合并（joined）、摘要长度截断（shortened）；
3. capped & joined：议题容量超限（> 12）时物理合并耗时最短的两个相邻议题；
4. keeping_terms_order：名词表保序，防止前端跳动。
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from schemas.minutes import MeetingNotes, NoteTopic, NoteConclusion, NoteTodo, NoteTerm, PointItem


class NotesCompactor:
    """会议纪要多轮状态脱水与缝合器。"""

    @staticmethod
    def shortened(summary: str, limit: int = 160) -> str:
        """截断失控的议题摘要，保留第一句或截断至上限加省略号（对齐 Notes.swift shortened）。"""
        if not summary or len(summary) <= limit:
            return summary
        head = summary[:limit]
        # 从第 10 个字符后寻找句号/标点截断
        for i in range(10, len(head)):
            if head[i] in "。！？；.!?;":
                return head[: i + 1]
        return head.strip() + "…"

    @staticmethod
    def capped(topics: List[NoteTopic], count: int) -> List[NoteTopic]:
        """至多保留 count 个议题（至少保留1个）；超出的议题要点合并进最后一个保留的议题（对齐 Notes.swift capped）。"""
        res = [t.model_copy(deep=True) for t in topics]
        while len(res) > max(1, count):
            extra = res.pop()
            last = res.pop()
            # 将 extra 的要点与支撑行合并到 last，保留至多 10 个弹性要点
            merged_points = last.points + extra.points
            last.points = merged_points[:10]
            last.evidence = list(dict.fromkeys(last.evidence + extra.evidence))
            res.append(last)
        return res

    @staticmethod
    def joined(a: NoteTopic, b: NoteTopic) -> NoteTopic:
        """将两个相邻议题物理合并为一个（对齐 Notes.swift joined）：
        合并标题与摘要，从各自均匀截取至多 10 个 points（最大弹性上限）。
        """
        from_a = min(len(a.points), max(2, 10 - len(b.points)))
        from_b = min(len(b.points), 10 - from_a)
        merged_points = list(a.points[:from_a] + b.points[:from_b])

        gap = "" if (a.summary and a.summary[-1] in "。！？；.!?;") else " "
        merged_summary = f"{a.summary}{gap}{b.summary}"
        merged_evidence = list(dict.fromkeys(a.evidence + b.evidence))

        return NoteTopic(
            start=min(a.start, b.start) if a.start > 0 and b.start > 0 else (a.start or b.start),
            title=f"{a.title} / {b.title}",
            summary=merged_summary,
            points=merged_points,
            fishbone=a.fishbone or b.fishbone,
            evidence=merged_evidence,
        )

    @staticmethod
    def carry_times(
        old_notes: MeetingNotes,
        next_notes: MeetingNotes,
        evidence_times: Dict[str, float],
    ) -> None:
        """继承并锁定时间戳，防止时间后移漂移（对齐 Notes.swift carryTimes）。"""
        # 1. 结论时间戳继承
        conclusion_times: Dict[str, float] = {}
        for c in old_notes.conclusions:
            if c.t is not None:
                conclusion_times[c.text.strip()] = c.t

        for c in next_notes.conclusions:
            key = c.text.strip()
            if key in conclusion_times:
                c.t = conclusion_times[key]
            elif c.t is None and c.evidence:
                c_times = [evidence_times[e] for e in c.evidence if e in evidence_times]
                if c_times:
                    c.t = min(c_times)

        # 2. 待办时间戳继承
        todo_times: Dict[str, float] = {}
        for t in old_notes.todos:
            if t.t is not None:
                todo_times[t.text.strip()] = t.t

        for t in next_notes.todos:
            key = t.text.strip()
            if key in todo_times:
                t.t = todo_times[key]
            elif t.t is None and t.evidence:
                t_times = [evidence_times[e] for e in t.evidence if e in evidence_times]
                if t_times:
                    t.t = min(t_times)

        # 3. 议题起始时间保护：若相同标题议题已存在，锁定其最初起始时间
        old_topic_times: Dict[str, float] = {
            t.title.strip(): t.start for t in old_notes.topics if t.start > 0
        }
        for topic in next_notes.topics:
            norm_title = topic.title.strip()
            if norm_title in old_topic_times:
                topic.start = old_topic_times[norm_title]
            elif topic.evidence:
                t_times = [evidence_times[e] for e in topic.evidence if e in evidence_times]
                if t_times:
                    first_ev = min(t_times)
                    if topic.start <= 0 or first_ev < topic.start:
                        topic.start = first_ev

    @classmethod
    def merge_topics(
        cls,
        old_topics: List[NoteTopic],
        returned_topics: List[NoteTopic],
        frozen: int = 0,
        live_max_topics: int = 12,
    ) -> List[NoteTopic]:
        """融合模型返回的新议题与历史议题（1:1 对齐 Notes.swift merge 算法）。"""
        if not old_topics:
            res = [t.model_copy(deep=True) for t in returned_topics]
            return cls.capped(res, live_max_topics)

        if not returned_topics:
            return old_topics

        # 1. 确定保留的前置冻结议题
        if frozen > 0:
            kept = [t.model_copy(deep=True) for t in old_topics[:frozen]]
        else:
            # 当 frozen 为 0 时，若模型只返回了最新增量议题，保留模型未覆盖的前序老议题
            new_titles = {t.title.strip() for t in returned_topics}
            kept = [t.model_copy(deep=True) for t in old_topics if t.title.strip() not in new_titles]

        fixed_titles = {t.title.strip() for t in kept}
        own = [t.model_copy(deep=True) for t in returned_topics if t.title.strip() not in fixed_titles]

        # 2. 截取尾部活跃议题
        if not kept:
            tail = own
        else:
            max_tail_len = max(1, len(old_topics) - len(kept) + 1)
            tail = own[-max_tail_len:] if len(own) > max_tail_len else own

        if not tail:
            return old_topics

        for i in range(len(tail)):
            tail[i].summary = cls.shortened(tail[i].summary)

        tail = cls.capped(tail, live_max_topics - min(len(kept), 1))

        # 3. 超出 live_max_topics 时，将前序已稳定的相邻且时间跨度最短的两个议题物理合并
        while len(kept) + len(tail) > live_max_topics and len(kept) >= 2:
            ends = [t.start for t in kept[1:]] + [tail[0].start]
            min_idx = min(
                range(len(kept) - 1),
                key=lambda idx: ends[idx + 1] - kept[idx].start,
            )
            kept[min_idx] = cls.joined(kept[min_idx], kept[min_idx + 1])
            kept.pop(min_idx + 1)

        merged = cls.capped(kept + tail, live_max_topics)
        merged.sort(key=lambda t: t.start)
        # 单个议题弹性上限硬防线：确保任何议题的要点均不超出 10 个
        for t in merged:
            if len(t.points) > 10:
                t.points = t.points[:10]
        return merged

    @staticmethod
    def keeping_terms_order(
        old_terms: List[NoteTerm],
        next_terms: List[NoteTerm],
    ) -> List[NoteTerm]:
        """保持专有名词在 UI 上的排序稳定，新名词排在末尾。"""
        rank: Dict[str, int] = {t.name.strip(): i for i, t in enumerate(old_terms)}
        next_sorted = sorted(
            next_terms,
            key=lambda t: rank.get(t.name.strip(), len(old_terms) + 1),
        )
        return next_sorted

    @classmethod
    def compact(
        cls,
        old_notes: MeetingNotes,
        next_notes: MeetingNotes,
        evidence_times: Dict[str, float],
        covered_line_num: int,
        frozen: int = 0,
        final: bool = False,
        live_max_topics: int = 12,
        max_topics: int = 8,
    ) -> MeetingNotes:
        """执行完整状态脱水、时间戳保护与议题缝合。"""
        # 1. 时间戳锁定与回溯
        cls.carry_times(old_notes, next_notes, evidence_times)

        # 2. 议题缝合与容量控制
        if final:
            # 会议终盘定稿：最多保留 max_topics (8) 个议题
            for t in next_notes.topics:
                t.summary = cls.shortened(t.summary)
            final_topics = cls.capped(next_notes.topics, max_topics)
            final_topics.sort(key=lambda t: t.start)
        else:
            # 会中实时流式：执行滑动窗口缝合与 live_max_topics (12) 动态合并
            final_topics = cls.merge_topics(
                old_topics=old_notes.topics,
                returned_topics=next_notes.topics,
                frozen=frozen,
                live_max_topics=live_max_topics,
            )

        # 确保所有议题的 points 绝不超出 10 个弹性上限
        for t in final_topics:
            if len(t.points) > 10:
                t.points = t.points[:10]

        # 3. 术语保序
        final_terms = cls.keeping_terms_order(old_notes.terms, next_notes.terms)

        # 4. 标题保持（除非新标题显著有效）
        final_title = next_notes.title or old_notes.title or "实时会议纪要"

        # 5. 生成融合产物
        return MeetingNotes(
            title=final_title,
            summary=next_notes.summary or old_notes.summary,
            topics=final_topics,
            conclusions=next_notes.conclusions or old_notes.conclusions,
            todos=next_notes.todos or old_notes.todos,
            terms=final_terms,
            covered=max(covered_line_num, old_notes.covered),
            updated_at=datetime.now(timezone.utc).isoformat(),
            final=final,
        )
