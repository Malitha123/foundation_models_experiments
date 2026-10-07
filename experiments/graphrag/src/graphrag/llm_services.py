"""Provider factories with explicit credentials and YAML defaults."""
from typing import Any, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from graphrag.config import (
    GENERAL_MODEL, GENERAL_PROVIDER, STRONG_MODEL, STRONG_PROVIDER,
    EMBEDDING_MODEL, PROVIDER, OPENROUTER_BASE_URL, get_api_key, get_config,
)


def _provider_kwargs(provider):
    if provider not in {"openai", "openrouter"}:
        raise ValueError(f"Unsupported provider: {provider}. Use 'openrouter' or 'openai'.")
    key = get_api_key(provider)
    if not key or not key.strip():
        raise ValueError(f"Missing required secret: {provider.upper()}_API_KEY")
    result = {"api_key": key}
    if provider == "openrouter":
        result["base_url"] = OPENROUTER_BASE_URL
    return result


def _build_llm(model: str, provider: str, temperature: Optional[float] = None, **kwargs: Any):
    defaults = get_config().get("llm", {})
    options = {
        "model": model,
        "temperature": defaults.get("temperature", 0) if temperature is None else temperature,
        "streaming": defaults.get("streaming", False),
        "max_tokens": defaults.get("max_tokens", 2000),
        "timeout": defaults.get("request_timeout", 60),
        **kwargs,
        **_provider_kwargs(provider),
    }
    return ChatOpenAI(**options)


def get_general_llm(temperature: Optional[float] = None, **kwargs: Any):
    return _build_llm(GENERAL_MODEL, GENERAL_PROVIDER, temperature, **kwargs)


def get_strong_llm(temperature: Optional[float] = None, **kwargs: Any):
    return _build_llm(STRONG_MODEL, STRONG_PROVIDER, temperature, **kwargs)


def get_embeddings(batch_size: Optional[int] = None, show_progress: Optional[bool] = None, **kwargs: Any):
    defaults = get_config().get("embedding", {})
    options = {
        "model": EMBEDDING_MODEL,
        "show_progress_bar": defaults.get("show_progress", False) if show_progress is None else show_progress,
        "chunk_size": defaults.get("batch_size", 100) if batch_size is None else batch_size,
        **kwargs,
        **_provider_kwargs(PROVIDER),
    }
    return OpenAIEmbeddings(**options)
