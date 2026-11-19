# 实时会议纪要 Agent（Real-time Meeting Minutes Agent）

基于 **Python + LangGraph + FastAPI** 构建的完全自包含、独立的实时会议纪要生成系统。

本项目在架构设计、提示词工程与状态脱水算法上 **1:1 深度对齐 Cicada（知了）Mac 原版源码（`Notes.swift`）**，兼具工业级会议系统（如飞书妙记、钉钉听悟）的流式体验与结构化纪要品质。

---

## 核心特性与架构机制

### 1. 双轨异步流式引擎
- **实时推流轨（毫秒级）**：上游 ASR 通过 `POST /api/lines` 推送发言，系统完成 4 字段强类型校验后，通过 WebSocket 毫秒级广播至所有客户端，左栏原话流水即刻滚动上屏。
- **纪要增量提炼轨（批处理状态图）**：LangGraph 后台工作流智能感知数据积攒（积攒 6 条新原话或距离上次提炼满 45 秒自适应触发），调用大模型提取结构化纪要，右栏平滑增量更新。

### 2. 源码级议题控制与状态脱水（1:1 对齐 `Notes.swift`）
- **节奏感知与 2 分钟刹车**：动态感知最新原话距离当前活跃议题的跨度，不足 2 分钟时在 Prompt 注入强约束指令，严禁开启新建议题，确保议题具备 3~10 分钟的大板块厚度，拒绝碎片化短命议题。
- **滑动窗口冻结（Frozen Topics）**：会中增量轮次仅向大模型暴露尾部 2 个活跃议题（`EDITABLE_TOPICS = 2`）供重写，更早的历史议题列入 `Settled` 区域锁定，彻底解决长会 Token 爆炸与前期已定案议题遗忘漂移问题。
- **会中容量自愈合并（Compactor joined & capped）**：会中最大议题数放宽至 12 个（`LIVE_MAX_TOPICS = 12`）。一旦议题超出 12 个，脱水机算法自动寻找前序耗时最短的两个相邻议题进行物理合并（`议题A / 议题B`，摘要拼接，要点各取所长均匀缩减），保障会中空间永远流动。
- **摘要防滚雪球熔断（shortened）**：议题 `summary` 超出 160 字时自动在标点截断并加省略号，杜绝大模型无限日志式机械追加（如“最新/又/补充”）。
- **终盘全景定稿重塑（Meeting End Final Pass）**：提供 `POST /api/end` 接口与前端“结束会议 (生成定稿)”按钮。会议结束时推倒会中局部草稿，以上帝全知视角从 L1 到 LN 纵览全场逐字稿，一次性重构出至多 8 个宏观全局议题（`MAX_TOPICS = 8`）。

### 3. 精细化纪要结构（纯粹克制，告别过度分类）
- **内容梳理（议题与要点）**：
  - `point`：每个议题提炼 **2~5 个**核心点。要求是**一句简短完整的话**（表意明确，讲清事实脉络或核心观点），坚决拒绝三四个字的干瘪短语标签；
  - `details`：**严格限制至多 1 条**。仅在发言有具体量化数据、技术配置参数或实操步骤时输出专属深层下挂；若发言已完整表述无更深细节，坚决置空 `[]`。
- **结论与真实待办**：
  - `conclusions`：最多 5 条现场拍板确定的定论，去除多余繁复分类；
  - `todos`：严格提取确凿待办（Only real to-dos, often none），支持负责人与截止时间（`deadline`）结构化归拢。
- **鱼骨图与专有名词**：
  - `fishbone`：仅在议题集中排查根因时生成，支持 `label/detail` 别名自适应强容错；
  - `terms`：专有名词保序脱水，前端清爽隐藏专有名词区，信息聚焦主干。

### 4. LLM 后端双模自由切换
- 支持在 `.env` 中通过 `LLM_BACKEND` 实现双模一键切换：
  - **`http`**：使用云端 DeepSeek 官方 API（高性价比，开箱即用）；
  - **`vllm`**：使用本地或内网服务器私有部署的 vLLM OpenAI 兼容接口，URL 尾缀自动规整清洗。

### 5. 极简素雅 Web 看板（飞书级丝滑质感）
- **黑白灰素雅排版**：无玩具感 Emoji 与高饱和花哨色块，以 `#111827`、`#F9FAFB` 与 `#E5E7EB` 呈现专业测试看板；
- **状态感知**：当前讨论议题挂载 `[正在讲]` 动态微标，提炼中就地渲染 `SkeletonBar` 微光骨架条；
- **自然滚动跟随**：飞书风格的视口管理，新内容产出自动平滑跟随滚动；用户向上翻阅历史时自动锁定视口，并悬浮浮动药丸 `[↓ 滚动至最新纪要]`。

---

## 目录结构

```text
realtime_minutes_agent/
├── .env                          # 环境变量真实配置文件（私密 Key，已被 .gitignore 忽略）
├── .env.example                  # 环境变量配置模板
├── .gitignore                    # Git 忽略配置（忽略 .env、__pycache__、.pytest_cache 等）
├── requirements.txt              # 独立 Python 依赖清单
├── main.py                       # FastAPI 主服务（REST 路由、WebSocket 通道、静态资源）
├── README.md                     # 本项目说明文档
│
├── core/                         # 核心基础设施
│   ├── config.py                 # 系统配置加载（基于 pydantic-settings，双模后端支持）
│   ├── llm.py                    # 大模型客户端初始化（ChatOpenAI + 强 JSON 模式锁）
│   └── compactor.py              # 状态脱水机（carry_times、merge_topics、capped、joined、shortened）
│
├── schemas/                      # 强类型数据契约（Pydantic V2）
│   ├── asr.py                    # ASR 4 字段输入模型与规范化流水行
│   └── minutes.py                # 结构化会议纪要数据模型（PointItem、NoteTopic、MeetingNotes）
│
├── tasks/                        # LangGraph 工作流与提示词工程
│   ├── state.py                  # 状态机 State 契约定义
│   ├── prompts.py                # 对齐源码哲学的中文提示词构建（时间刹车、冻结、终盘定稿）
│   └── graph.py                  # LangGraph 状态图编译与调度
│
├── static/                       # 前端交互面板
│   └── index.html                # 纯原生 JS/CSS 黑白灰双栏流式看板
│
├── data/                         # 真实演练语料
│   └── mock_meeting.jsonl        # 真实技术会议 260 条全量发言测试语料
│
├── simulate_stream.py            # 命令行 ASR 模拟推流脚本（支持 --interval 参数）
│
└── tests/                        # 自动化回归测试套件
    └── test_pipeline.py          # 13 项单元与端到端测试（契约校验、脱水机算法、后端切换、API）
```

---

## 快速上手

### 1. 安装依赖
建议使用 Python 3.10+ 虚拟环境：
```bash
pip install -r requirements.txt
```

### 2. 配置环境变量
复制 `.env.example` 为 `.env`：
```bash
cp .env.example .env
```
根据实际情况配置后端类型：

- **模式 A：使用云端 DeepSeek 官方 API**
  ```ini
  LLM_BACKEND=http
  DEEPSEEK_API_KEY=sk-your-deepseek-api-key
  DEEPSEEK_BASE_URL=https://api.deepseek.com
  DEEPSEEK_MODEL=deepseek-chat
  ```

- **模式 B：使用私有部署的 vLLM OpenAI 兼容接口**
  ```ini
  LLM_BACKEND=vllm
  LLM_VLLM_BASE_URL=http://192.168.1.100:8000/v1
  LLM_VLLM_API_KEY=EMPTY
  LLM_VLLM_MODEL=deepseek-v4-flash-0731
  ```

### 3. 启动服务
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
服务启动后：
- **Web 实时测试看板**：打开浏览器访问 `http://127.0.0.1:8000`
- **Swagger API 文档**：访问 `http://127.0.0.1:8000/docs`

### 4. 模拟推流演练
- **方式一（网页一键演练）**：在浏览器页面右上角点击 **“开始模拟 (10s/条)”**，网页将每隔 10 秒自动向后端推送一条真实技术会议发言；
- **方式二（命令行独立脚本）**：
  ```bash
  python simulate_stream.py --interval 10
  ```

---

## API 接口规范

### 1. `POST /api/lines`
上游 ASR 推送接口（严格 4 字段校验），接收后毫秒级推至 WebSocket 原话栏：
```json
{
  "line_id": 1,
  "t": 10.5,
  "speaker": "张三",
  "text": "我们今天开会讨论下结算网关并发改造的事情。"
}
```
*注：`line_id` 若未提供，后端将按序列自增分配 `1, 2, 3...`。*

### 2. `POST /api/trigger`
手动强制立即触发一次 LangGraph 增量提炼（用于调试，绕过 45 秒等待）。

### 3. `POST /api/end`
结束会议并触发终盘定稿（Final Pass）：清空会中临时草稿，上帝视角纵览全场逐字稿并重新梳理出至多 8 个宏观议题。

### 4. `POST /api/reset`
清空当前会议的全部原话流水与纪要状态，重置为初始状态。

### 5. `GET /api/notes`
获取当前最新结构化会议纪要 JSON 对象。

### 6. `GET /api/lines`
获取全部历史转写流水列表。

### 7. `GET /api/mock_data`
读取本地 `data/mock_meeting.jsonl` 中的 260 条演练语料。

### 8. `WebSocket /ws`
双向流式推流通讯端点：
- 连接成功自动下发初始快照（`type: init`）；
- ASR 原话推送即时广播（`type: new_line`）；
- 纪要提炼结果增量广播（`type: notes_update`）；
- AI 处理状态广播（`type: status`，`updating` / `idle` / `error`）。

---

## 自动化测试

运行全套 13 项自动化回归测试套件（覆盖数据契约、脱水机合并裁剪算法、2分钟刹车、终盘定稿与双模后端解析）：
```bash
python -m pytest tests/test_pipeline.py -v
```

**测试结果预期：**
```text
tests/test_pipeline.py::test_asr_valid_input PASSED
tests/test_pipeline.py::test_asr_auto_increment_line_id PASSED
tests/test_pipeline.py::test_asr_invalid_inputs PASSED
tests/test_pipeline.py::test_conclusion_clean_schema PASSED
tests/test_pipeline.py::test_fishbone_category_cause_mapping PASSED
tests/test_pipeline.py::test_todo_deadline_and_points_details PASSED
tests/test_pipeline.py::test_compactor_carry_times_and_protection PASSED
tests/test_pipeline.py::test_fastapi_rest_endpoints PASSED
tests/test_pipeline.py::test_llm_backend_switching PASSED
tests/test_pipeline.py::test_compactor_capped_and_joined PASSED
tests/test_pipeline.py::test_compactor_shortened_summary PASSED
tests/test_pipeline.py::test_compactor_live_max_topics_auto_merge PASSED
tests/test_pipeline.py::test_prompt_two_minute_brake_and_final_pass PASSED

============================= 13 passed in 4.83s ==============================
```
