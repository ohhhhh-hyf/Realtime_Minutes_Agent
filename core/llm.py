"""LLM 客户端初始化（适配云端 API 与本地/服务器 vLLM，支持强 JSON 模式）。"""

from typing import Optional
from langchain_openai import ChatOpenAI
from .config import get_settings


def get_chat_model(temperature: Optional[float] = None) -> ChatOpenAI:
    """初始化用于结构化输出的 ChatOpenAI 实例。

    根据 .env 中的 LLM_BACKEND 自动无缝切换：
    - http: DeepSeek 官方 API
    - vllm: 服务器/本地 vLLM OpenAI 兼容接口
    """
    settings = get_settings()
    active_conf = settings.resolve_active_llm()

    api_key = active_conf["api_key"] or "mock-key"
    base_url = active_conf["base_url"]
    model_name = active_conf["model"]
    temp = temperature if temperature is not None else active_conf["temperature"]

    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=temp,
        model_kwargs={
            "response_format": {"type": "json_object"}
        },
        timeout=int(active_conf["timeout"]),
        max_retries=int(active_conf["max_retries"]),
    )
