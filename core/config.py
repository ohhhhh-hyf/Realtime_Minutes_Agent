"""系统配置加载模块（基于 pydantic-settings）。
支持云端 API 与本地/服务器 vLLM OpenAI 兼容接口自由切换（1:1 参考 Minutes_Agent 机制）。
"""

import os
from functools import lru_cache
from typing import Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全局配置。"""

    # 1. 后端切换控制: http (云端 API) | vllm (服务器/本地 vLLM)
    LLM_BACKEND: str = Field(default="http", description="LLM 后端类型: http | vllm")

    # 2. 云端官方 API (DeepSeek)
    DEEPSEEK_API_KEY: str = Field(default="", description="DeepSeek 官方 API 密钥")
    DEEPSEEK_BASE_URL: str = Field(default="https://api.deepseek.com", description="DeepSeek 官方 Base URL")
    DEEPSEEK_MODEL: str = Field(default="deepseek-chat", description="DeepSeek 模型名称")
    DEEPSEEK_TEMPERATURE: float = Field(default=0.2, description="采样温度")

    # 3. 本地 / 服务器 vLLM OpenAI 兼容接口
    LLM_VLLM_BASE_URL: str = Field(default="http://127.0.0.1:8000/v1", description="服务器 vLLM Base URL")
    LLM_VLLM_API_KEY: str = Field(default="EMPTY", description="服务器 vLLM API Key")
    LLM_VLLM_MODEL: str = Field(default="deepseek-v4-flash-0731", description="服务器 vLLM 模型名称")
    LLM_VLLM_TEMPERATURE: float = Field(default=0.2, description="vLLM 采样温度")

    # 4. 通用超时与重试
    LLM_TIMEOUT: float = Field(default=60.0, description="请求超时时间（秒）")
    LLM_MAX_RETRIES: int = Field(default=2, description="最大重试次数")

    # 5. 触发与议题容量策略（1:1 对齐 Cicada 源码 Notes.swift）
    UPDATE_INTERVAL_SECONDS: float = 45.0  # 距离上次提取至少 45 秒
    TRIGGER_LINE_BATCH: int = 6           # 积攒至少 6 条新转写行
    MAX_TOPICS: int = 8                   # 终盘定稿最大议题数
    LIVE_MAX_TOPICS: int = 12             # 会中流式最大议题数
    EDITABLE_TOPICS: int = 2              # 会中可编辑/重写的尾部议题数（前面的被 frozen 冻结）
    TOPIC_SUMMARY_LIMIT: int = 160        # 议题摘要字符上限，超长自动截断
    TOPIC_MIN_SPAN_SECONDS: float = 120.0 # 议题最小跨度（2分钟冷却刹车）
    TOPIC_MIN_POINTS: int = 4             # 议题最低要点门槛（未满 4 个严禁切新议题）
    TOPIC_MAX_POINTS: int = 10            # 议题最大要点上限（达到 10 个强制熔断切新议题）
    CONTEXT_LINES: int = 2                # 注入的前置上下文行数

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def resolve_active_llm(self) -> Dict[str, Any]:
        """根据 LLM_BACKEND 解析出当前激活生效的 LLM 详细参数字典。"""
        backend = (self.LLM_BACKEND or "http").strip().lower()
        if backend in ("vllm", "local_vllm", "server_vllm"):
            api_key = (self.LLM_VLLM_API_KEY or "EMPTY").strip()
            base_url = (self.LLM_VLLM_BASE_URL or "http://127.0.0.1:8000/v1").strip().rstrip("/")
            model = (self.LLM_VLLM_MODEL or "deepseek-v4-flash-0731").strip()
            temperature = self.LLM_VLLM_TEMPERATURE
        else:
            api_key = (self.DEEPSEEK_API_KEY or "").strip()
            base_url = (self.DEEPSEEK_BASE_URL or "https://api.deepseek.com").strip().rstrip("/")
            model = (self.DEEPSEEK_MODEL or "deepseek-chat").strip()
            temperature = self.DEEPSEEK_TEMPERATURE

        # 规范化 URL: 剥离可能多余的 /chat/completions 尾缀
        if base_url.endswith("/chat/completions"):
            base_url = base_url[:-len("/chat/completions")].rstrip("/")

        return {
            "backend": backend,
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "temperature": temperature,
            "timeout": self.LLM_TIMEOUT,
            "max_retries": self.LLM_MAX_RETRIES,
        }


@lru_cache()
def get_settings() -> Settings:
    """获取缓存的单例配置实例。"""
    return Settings()
