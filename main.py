"""FastAPI 统一主服务入口（双轨推流：ASR 原话毫秒级广播 + 纪要增量提炼）。"""

import asyncio
import os
import time
import logging
from contextlib import asynccontextmanager
from typing import List, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from schemas.asr import ASRLineInput, TranscriptLine
from schemas.minutes import MeetingNotes
from core.config import get_settings
from tasks.graph import create_meeting_graph
from tasks.state import MeetingGraphState

# 配置基础日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("minutes_agent")

# 全局内存会话状态
class SessionManager:
    def __init__(self):
        self.transcript: List[TranscriptLine] = []
        self.notes: MeetingNotes = MeetingNotes()
        self.last_covered_line: int = 0
        self.last_update_time: float = 0.0
        self.is_updating: bool = False
        self.active_sockets: Set[WebSocket] = set()
        self.lock = asyncio.Lock()
        self.graph = create_meeting_graph()

    def reset(self):
        self.transcript.clear()
        self.notes = MeetingNotes()
        self.last_covered_line = 0
        self.last_update_time = 0.0
        self.is_updating = False

    async def broadcast(self, message: dict):
        """广播消息至所有活跃 WebSocket 客户端。"""
        dead_sockets = set()
        for ws in self.active_sockets:
            try:
                await ws.send_json(message)
            except Exception:
                dead_sockets.add(ws)
        self.active_sockets -= dead_sockets


session = SessionManager()


async def background_extractor_worker():
    """后台轮询协程：根据时间间隔或积累发言条数触发 LangGraph 纪要提炼。"""
    settings = get_settings()
    while True:
        try:
            await asyncio.sleep(3.0)
            now = time.time()
            total_lines = len(session.transcript)
            uncovered = total_lines - session.last_covered_line

            # 触发策略（1:1 对齐 Cicada Notes.swift 调度机理）：
            # 1. 开场首版：首批发言满 2 条立即提炼，或首句等待满 20 秒提炼
            # 2. 会中加速：未处理条数满 6 条提前冲刺提炼
            # 3. 会中兜底：只要有新发言且距离上次提炼满 45 秒，坚决触发提炼，绝不因条数不足被拦截
            should_trigger = (
                not session.is_updating
                and total_lines > 0
                and uncovered > 0
                and (
                    (session.last_covered_line == 0 and (total_lines >= 2 or (now - session.last_update_time) >= 20.0))
                    or uncovered >= settings.TRIGGER_LINE_BATCH
                    or (session.last_covered_line > 0 and (now - session.last_update_time) >= settings.UPDATE_INTERVAL_SECONDS)
                )
            )

            if should_trigger:
                # 既然已满足时间或条数门槛，直接传 force=True 透传放行，彻底消除双重门禁死锁
                await run_extraction_pass(force=True)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"后台批处理轮询异常: {str(e)}", exc_info=True)


async def run_extraction_pass(force: bool = False, final: bool = False):
    """执行一次 LangGraph 会议纪要提炼（支持会中实时更新与会后终盘定稿）。"""
    async with session.lock:
        if session.is_updating:
            return
        session.is_updating = True

    try:
        await session.broadcast({"type": "status", "data": "updating"})

        initial_state: MeetingGraphState = {
            "transcript": list(session.transcript),
            "notes": session.notes,
            "last_covered_line": session.last_covered_line,
            "force_update": force,
            "should_update": False,
            "final": final,
            "error": None,
        }

        # 调用 LangGraph 执行状态图
        final_state = await session.graph.ainvoke(initial_state)

        if final_state.get("error"):
            logger.warning(f"提炼报告警告: {final_state['error']}")
            await session.broadcast({"type": "status", "data": "error", "message": final_state["error"]})
        else:
            if final_state.get("notes"):
                session.notes = final_state["notes"]
            if final_state.get("last_covered_line") is not None:
                session.last_covered_line = final_state["last_covered_line"]
            session.last_update_time = time.time()

            # 广播纪要更新
            await session.broadcast({
                "type": "notes_update",
                "data": session.notes.model_dump(),
            })
            await session.broadcast({"type": "status", "data": "idle"})

    except Exception as e:
        logger.error(f"运行提炼管线失败: {str(e)}", exc_info=True)
        await session.broadcast({"type": "status", "data": "error", "message": str(e)})
    finally:
        session.is_updating = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动后台批处理定时任务
    worker_task = asyncio.create_task(background_extractor_worker())
    logger.info("实时会议纪要服务已启动，后台增量提炼工作流就绪。")
    yield
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    logger.info("实时会议纪要服务已正常关闭。")


app = FastAPI(
    title="Realtime Minutes Agent",
    version="1.0.0",
    description="基于 Python + LangGraph 的自包含实时会议纪要系统",
    lifespan=lifespan,
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态资源挂载
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def get_index():
    """提供 Web 看板首页（禁用缓存以确保更新即时生效）。"""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(
            index_file,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return {"message": "Realtime Minutes Agent API is running. static/index.html not found."}


@app.post("/api/lines")
async def receive_asr_line(line_input: ASRLineInput):
    """上游 ASR 推送接口（严格 4 字段校验）。

    接收后毫秒级广播至所有前端 WebSocket，推至原话流水栏。
    """
    assigned_num = len(session.transcript) + 1
    norm_line = TranscriptLine.from_input(line_input, assigned_id=assigned_num)

    session.transcript.append(norm_line)
    if session.last_update_time == 0.0:
        session.last_update_time = time.time()

    # 毫秒级即时广播原话行
    await session.broadcast({
        "type": "new_line",
        "data": norm_line.model_dump(),
    })

    return {
        "status": "success",
        "line": norm_line.model_dump(),
        "total_lines": len(session.transcript),
    }


@app.get("/api/lines")
async def get_all_lines():
    """获取所有历史发言流水。"""
    return [line.model_dump() for line in session.transcript]


@app.get("/api/notes")
async def get_current_notes():
    """获取当前最新结构化会议纪要。"""
    return session.notes.model_dump()


@app.get("/api/mock_data")
async def get_mock_data():
    """读取本地 data/mock_meeting.jsonl 中的全量数据并返回。"""
    import json
    mock_file = os.path.join(os.path.dirname(__file__), "data", "mock_meeting.jsonl")
    if not os.path.exists(mock_file):
        raise HTTPException(status_code=404, detail="data/mock_meeting.jsonl 文件不存在")
    items = []
    with open(mock_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            s = line.strip()
            if s:
                try:
                    data = json.loads(s)
                    items.append({
                        "line_id": data.get("line_id", idx),
                        "t": float(data.get("t", (idx - 1) * 10.0)),
                        "speaker": str(data.get("speaker", f"发言人{idx}")),
                        "text": str(data.get("text", "")),
                    })
                except Exception:
                    pass
    return items


@app.post("/api/trigger")
async def force_trigger():
    """手动强制立即执行一次纪要提炼。"""
    if len(session.transcript) == 0:
        raise HTTPException(status_code=400, detail="暂无发言记录，无法生成纪要")
    asyncio.create_task(run_extraction_pass(force=True, final=False))
    return {"status": "triggered"}


@app.post("/api/end")
async def end_meeting():
    """结束会议：触发终盘重整（Final Pass），推倒重来提炼至多 8 个宏观议题。"""
    if len(session.transcript) == 0:
        raise HTTPException(status_code=400, detail="暂无发言记录，无法生成终盘纪要")
    asyncio.create_task(run_extraction_pass(force=True, final=True))
    return {"status": "ending_triggered"}


@app.post("/api/reset")
async def reset_session():
    """重置/清空会议记录。"""
    async with session.lock:
        session.reset()
    await session.broadcast({"type": "reset"})
    return {"status": "reset"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """前端双向推流通讯端点。"""
    await websocket.accept()
    session.active_sockets.add(websocket)
    try:
        # 初始握手：向新连接推送当前全量流水与最新纪要
        await websocket.send_json({
            "type": "init",
            "lines": [l.model_dump() for l in session.transcript],
            "notes": session.notes.model_dump(),
            "status": "updating" if session.is_updating else "idle",
        })

        while True:
            # 保持长连接与心跳
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        session.active_sockets.discard(websocket)
    except Exception:
        session.active_sockets.discard(websocket)
