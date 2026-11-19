"""端到端回归测试套件。

覆盖范围：
1. ASR 4 字段严格边界校验；
2. 纪要结构与结论三分类（decision, risk, open）校验；
3. 待办 deadline 字段与议题 details 校验；
4. 状态脱水机 Compactor 时间戳与议题保护测试；
5. FastAPI REST 接口行为回归。
"""

import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from schemas.asr import ASRLineInput, TranscriptLine
from schemas.minutes import (
    PointItem,
    FishboneBranch,
    Fishbone,
    NoteTopic,
    NoteConclusion,
    NoteTodo,
    NoteTerm,
    MeetingNotes,
)
from core.compactor import NotesCompactor
from main import app, session


# ==============================================================================
# 1. ASR 4 字段输入校验测试
# ==============================================================================

def test_asr_valid_input():
    """合法 4 字段输入成功校验。"""
    raw = {
        "line_id": 1,
        "t": 10.5,
        "speaker": "张三",
        "text": "测试发言内容",
    }
    obj = ASRLineInput.model_validate(raw)
    assert obj.line_id == 1
    assert obj.t == 10.5
    assert obj.speaker == "张三"
    assert obj.text == "测试发言内容"

    # 规范化转为系统流水行
    line = TranscriptLine.from_input(obj, assigned_id=1)
    assert line.id == "L1"
    assert line.format_timestamp() == "00:10"
    assert "张三" in line.to_display_string()


def test_asr_auto_increment_line_id():
    """未传 line_id 时正确继承系统分配。"""
    raw = {
        "t": 0.0,
        "speaker": "李四",
        "text": "没有传 line_id 的原话",
    }
    obj = ASRLineInput.model_validate(raw)
    assert obj.line_id is None
    line = TranscriptLine.from_input(obj, assigned_id=42)
    assert line.id == "L42"
    assert line.line_id == 42


def test_asr_invalid_inputs():
    """非法边界输入严格拦截抛出 ValidationError。"""
    # 负数时间戳
    with pytest.raises(ValidationError):
        ASRLineInput.model_validate({"t": -1.0, "speaker": "王五", "text": "内容"})

    # 空发言人
    with pytest.raises(ValidationError):
        ASRLineInput.model_validate({"t": 5.0, "speaker": "   ", "text": "内容"})

    # 空原话文本
    with pytest.raises(ValidationError):
        ASRLineInput.model_validate({"t": 5.0, "speaker": "王五", "text": ""})

    # 缺少必填字段
    with pytest.raises(ValidationError):
        ASRLineInput.model_validate({"speaker": "王五", "text": "缺少时间戳"})


# ==============================================================================
# 2. 纪要结构与三分类校验测试
# ==============================================================================

def test_conclusion_clean_schema():
    """结论回归纯粹定论，去除多余分类字段，验证纯文本与时间戳。"""
    c1 = NoteConclusion(text="拍板定案", t=10.0, evidence=["L1"])
    assert c1.text == "拍板定案"
    assert c1.t == 10.0
    assert c1.evidence == ["L1"]

    # 支持纯字符串自动转换
    c2 = NoteConclusion.model_validate("直接字符串定论")
    assert c2.text == "直接字符串定论"

    # 支持带 title 或 content 的对象自动适配
    c3 = NoteConclusion.model_validate({"content": "网关改造方案定案", "t": 40.0})
    assert c3.text == "网关改造方案定案"


def test_fishbone_category_cause_mapping():
    """验证 LLM 常见 category/cause 等字段别名自动映射至 label/detail，防止 Validation 失败。"""
    raw_branch = {
        "category": "老板管理",
        "cause": "关注不够、不够雷厉风行、未及时解决",
    }
    b = FishboneBranch.model_validate(raw_branch)
    assert b.label == "老板管理"
    assert b.detail == "关注不够、不够雷厉风行、未及时解决"

    # 完整层级解析
    raw_notes = {
        "title": "年度经营复盘",
        "summary": "关于成本与年终奖的复盘",
        "topics": [
            {
                "start": 10.0,
                "title": "成本控制与利润率",
                "summary": "深入剖析成本失控",
                "points": [{"point": "利润率不可观", "details": []}],
                "fishbone": {
                    "question": "成本控制失控根因",
                    "branches": [
                        {"category": "老板管理", "cause": "关注不够、不够雷厉风行、未及时解决"},
                        {"category": "人员成本", "cause": "高薪人员集中，干满一年人数激增"},
                    ],
                },
            }
        ],
    }
    notes = MeetingNotes.model_validate(raw_notes)
    assert notes.topics[0].fishbone is not None
    assert len(notes.topics[0].fishbone.branches) == 2
    assert notes.topics[0].fishbone.branches[0].label == "老板管理"
    assert notes.topics[0].fishbone.branches[0].detail == "关注不够、不够雷厉风行、未及时解决"


def test_todo_deadline_and_points_details():
    """待办 deadline 与议题 points 自适应。"""
    # 待办带截止时间
    todo_with_dl = NoteTodo(
        text="完成网关重构",
        owner="李工",
        deadline="本周五前",
        t=30.0,
    )
    assert todo_with_dl.deadline == "本周五前"

    # 待办无截止时间
    todo_no_dl = NoteTodo(text="测试压测", owner="测试组", deadline=None)
    assert todo_no_dl.deadline is None

    # PointItem 字符串自动升级为结构
    point_from_str = PointItem.model_validate("纯文本主干论点")
    assert point_from_str.point == "纯文本主干论点"
    assert point_from_str.details == []

    # PointItem 严格约束 details 至多 1 条有意义细节
    point_with_details = PointItem(
        point="核心延迟瓶颈分析",
        details=["耗时瓶颈：串行依赖耗时 3.2 秒", "资源限制：连接池仅支持 200 并发"],
    )
    assert len(point_with_details.details) == 1
    assert point_with_details.details[0] == "耗时瓶颈：串行依赖耗时 3.2 秒"


# ==============================================================================
# 3. 状态脱水机 Compactor 测试
# ==============================================================================

def test_compactor_carry_times_and_protection():
    """测试多轮提取时的时间戳锁定与议题保护。"""
    old_notes = MeetingNotes(
        title="架构改造评审",
        summary="第一轮讨论",
        topics=[
            NoteTopic(
                start=10.0,
                title="网关现状",
                summary="现状分析",
                points=[PointItem(point="延迟超标", details=[])],
            )
        ],
        conclusions=[
            NoteConclusion(text="确定改造", t=12.0)
        ],
        todos=[
            NoteTodo(text="出架构图", owner="李工", deadline="明天", t=15.0)
        ],
        terms=[NoteTerm(name="网关", note="API Gateway")],
        covered=5,
    )

    # 新一轮模型提取，未填 t 或议题较少
    next_notes = MeetingNotes(
        title="架构改造评审",
        summary="第二轮讨论推进",
        topics=[
            NoteTopic(
                start=0.0,
                title="压测规划",
                summary="压测方案",
                points=[PointItem(point="10万QPS", details=[])],
                evidence=["L8"],
            )
        ],
        conclusions=[
            # 同文结论，模型返回 t 为 None
            NoteConclusion(text="确定改造", t=None),
            # 新结论
            NoteConclusion(text="防抖必做", t=45.0),
        ],
        todos=[
            NoteTodo(text="出架构图", owner="李工", deadline="明天", t=None)
        ],
        terms=[
            NoteTerm(name="QPS", note="每秒请求数"),
            NoteTerm(name="网关", note="API Gateway"),
        ],
        covered=10,
    )

    evidence_times = {"L8": 50.0}

    compacted = NotesCompactor.compact(
        old_notes=old_notes,
        next_notes=next_notes,
        evidence_times=evidence_times,
        covered_line_num=10,
    )

    # 1. 验证同名结论与待办时间戳成功继承自老轮次 (12.0 与 15.0)
    decision = next(c for c in compacted.conclusions if c.text == "确定改造")
    assert decision.t == 12.0

    todo = next(t for t in compacted.todos if t.text == "出架构图")
    assert todo.t == 15.0

    # 2. 验证老议题与新议题均保留（缝合保护生效）
    assert len(compacted.topics) == 2
    assert compacted.topics[0].title == "网关现状"
    assert compacted.topics[1].title == "压测规划"

    # 3. 验证术语排序稳定：老名词在前，新名词在后
    assert compacted.terms[0].name == "网关"
    assert compacted.terms[1].name == "QPS"


# ==============================================================================
# 4. FastAPI REST 端点测试
# ==============================================================================

def test_fastapi_rest_endpoints():
    """测试 FastAPI 核心接口流程。"""
    client = TestClient(app)

    # 1. 重置会话
    res = client.post("/api/reset")
    assert res.status_code == 200

    # 2. 推送合法 4 字段 ASR 数据
    payload = {
        "line_id": 1,
        "t": 5.0,
        "speaker": "王总",
        "text": "我们开会讨论一下结算网关架构改造。",
    }
    res = client.post("/api/lines", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["line"]["speaker"] == "王总"

    # 3. 推送非法 ASR 数据（缺少 text 字段）
    bad_payload = {
        "line_id": 2,
        "t": 10.0,
        "speaker": "李工",
    }
    res = client.post("/api/lines", json=bad_payload)
    assert res.status_code == 422

    # 4. 获取流水行
    res = client.get("/api/lines")
    assert res.status_code == 200
    lines = res.json()
    assert len(lines) == 1
    assert lines[0]["id"] == "L1"

    # 5. 获取当前纪要
    res = client.get("/api/notes")
    assert res.status_code == 200
    notes = res.json()
    assert "topics" in notes
    assert "conclusions" in notes
    assert "todos" in notes

    # 6. 获取 mock_meeting.jsonl 真实演练语料
    res = client.get("/api/mock_data")
    assert res.status_code == 200
    mock_data = res.json()
    assert len(mock_data) == 260
    # 7. 结束会议触发终盘定稿
    res_end = client.post("/api/end")
    assert res_end.status_code == 200
    assert res_end.json()["status"] == "ending_triggered"


def test_llm_backend_switching():
    """测试 LLM 后端双模（http vs vllm）自由切换配置解析。"""
    from core.config import Settings

    # 1. 模拟 http 模式
    s_http = Settings(
        LLM_BACKEND="http",
        DEEPSEEK_API_KEY="sk-test-http",
        DEEPSEEK_BASE_URL="https://api.deepseek.com",
        DEEPSEEK_MODEL="deepseek-chat",
    )
    conf_http = s_http.resolve_active_llm()
    assert conf_http["backend"] == "http"
    assert conf_http["api_key"] == "sk-test-http"
    assert conf_http["base_url"] == "https://api.deepseek.com"
    assert conf_http["model"] == "deepseek-chat"

    # 2. 模拟 vLLM 模式（含自动规整 URL 尾缀）
    s_vllm = Settings(
        LLM_BACKEND="vllm",
        LLM_VLLM_BASE_URL="http://192.168.1.100:8000/v1/chat/completions",
        LLM_VLLM_API_KEY="sjk_vllm_test",
        LLM_VLLM_MODEL="deepseek-v4-flash-0731",
    )
    conf_vllm = s_vllm.resolve_active_llm()
    assert conf_vllm["backend"] == "vllm"
    assert conf_vllm["api_key"] == "sjk_vllm_test"
    assert conf_vllm["base_url"] == "http://192.168.1.100:8000/v1"
    assert conf_vllm["model"] == "deepseek-v4-flash-0731"


# ==============================================================================
# 5. Cicada 源码对齐特性专用测试（刹车、冻结、容量合并、终盘）
# ==============================================================================

def test_compactor_capped_and_joined():
    """测试 capped 裁剪与 joined 合并两个相邻议题。"""
    t1 = NoteTopic(
        start=10.0,
        title="技术选型",
        summary="关于网关的技术选型探讨。",
        points=[
            PointItem(point="方案A性能高", details=["QPS超10万"]),
            PointItem(point="方案A维护成本大", details=[]),
        ],
        evidence=["L1", "L2"],
    )
    t2 = NoteTopic(
        start=40.0,
        title="风险评估",
        summary="系统并发风险评估。",
        points=[
            PointItem(point="连接池耗尽风险", details=["并发达200即阻塞"]),
            PointItem(point="超时重试风暴", details=[]),
        ],
        evidence=["L3", "L4"],
    )

    # 1. 验证 joined
    joined_topic = NotesCompactor.joined(t1, t2)
    assert joined_topic.start == 10.0
    assert joined_topic.title == "技术选型 / 风险评估"
    assert "关于网关的技术选型探讨。" in joined_topic.summary
    assert "系统并发风险评估。" in joined_topic.summary
    assert len(joined_topic.points) == 4
    assert set(joined_topic.evidence) == {"L1", "L2", "L3", "L4"}

    # 2. 验证 capped
    t3 = NoteTopic(start=80.0, title="后续待办", summary="排期待办", points=[PointItem(point="周五上线", details=[])])
    capped_list = NotesCompactor.capped([t1, t2, t3], count=2)
    assert len(capped_list) == 2
    assert capped_list[0].title == "技术选型"
    # 第三个议题的 points 合并入第二个
    assert capped_list[1].title == "风险评估"
    assert any(p.point == "周五上线" for p in capped_list[1].points)


def test_compactor_shortened_summary():
    """测试议题摘要超出 160 字符时自动截断。"""
    long_summary = "这是一段非常冗长的会议纪要摘要。" * 20  # 320 字
    shortened = NotesCompactor.shortened(long_summary, limit=160)
    assert len(shortened) <= 165
    assert shortened.endswith("。") or shortened.endswith("…")


def test_compactor_live_max_topics_auto_merge():
    """测试会中议题超出 12 个时，自动将前序耗时最短的两个相邻议题物理合并（joined）。"""
    topics = []
    # 构造 13 个议题：其中议题3(100s)与议题4(105s)，议题5始于115s，故议题3与4的配对跨度仅 15s，全场最短
    starts = [0, 50, 100, 105, 115, 200, 300, 400, 500, 600, 700, 800, 900]
    for i, s in enumerate(starts, start=1):
        topics.append(
            NoteTopic(
                start=float(s),
                title=f"议题{i}",
                summary=f"议题{i}摘要",
                points=[PointItem(point=f"要点{i}", details=[])],
            )
        )

    # 12 个老议题（frozen=10，后2个可编辑），尾部追加新议题13，总数达到 13
    merged = NotesCompactor.merge_topics(
        old_topics=topics[:12],
        returned_topics=[topics[10], topics[11], topics[12]],
        frozen=10,
        live_max_topics=12,
    )
    # 严格压缩到 12 个以内
    assert len(merged) <= 12
    # 验证最短相邻议题（议题3 与 议题4）被成功 joined
    joined_titles = [t.title for t in merged]
    assert any("议题3 / 议题4" in title for title in joined_titles)


def test_prompt_two_minute_brake_and_final_pass():
    """测试 Prompt 中 2 分钟刹车提示词注入与终盘定稿模式。"""
    from tasks.prompts import build_user_prompt

    notes = MeetingNotes(
        title="测试会议",
        summary="会议摘要",
        topics=[
            NoteTopic(
                start=60.0,
                title="网关改造",
                summary="网关改造讨论",
                points=[PointItem(point="要点1", details=[])],
            )
        ],
    )

    # 1. 处于展开阶段（讨论不足 120 秒或要点不足 2 个）：注入优先扩充提示
    lines_under_2m = [
        TranscriptLine(id="L1", line_id=1, t=60.0, speaker="张三", text="开始讨论网关"),
        TranscriptLine(id="L2", line_id=2, t=110.0, speaker="李四", text="连接池并发需要优化"),
    ]
    prompt_brake = build_user_prompt(notes, lines_under_2m, from_line=1, frozen=0, final=False)
    assert "当前活跃议题" in prompt_brake
    assert "优先在该议题的 points 列表中扩充新要点" in prompt_brake

    # 2. 讨论已充分（超过 120 秒且要点达到 2 个以上）：不注入初期展开提示
    notes_with_full_pts = MeetingNotes(
        title="测试会议",
        summary="会议摘要",
        topics=[
            NoteTopic(
                start=60.0,
                title="网关改造",
                summary="网关改造讨论",
                points=[PointItem(point="要点1", details=[]), PointItem(point="要点2", details=[])],
            )
        ],
    )
    lines_over_2m = [
        TranscriptLine(id="L1", line_id=1, t=60.0, speaker="张三", text="开始讨论网关"),
        TranscriptLine(id="L2", line_id=2, t=300.0, speaker="李四", text="下面对齐下季度规划"),
    ]
    prompt_normal = build_user_prompt(notes_with_full_pts, lines_over_2m, from_line=1, frozen=0, final=False)
    assert "讨论尚在展开阶段" not in prompt_normal

    # 3. 终盘定稿模式（final=True）：全景 8 议题指令
    prompt_final = build_user_prompt(notes, lines_over_2m, from_line=0, frozen=0, final=True)
    assert "刚刚结束的会议" in prompt_final
    assert "至多" in prompt_final
    assert "个议题" in prompt_final

