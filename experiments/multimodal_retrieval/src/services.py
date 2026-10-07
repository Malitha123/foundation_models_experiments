"""Model and embedding factories for the multimodal retrieval experiment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import yaml


def load_config(config_path: str | None = None) -> dict[str, Any]:
    path = Path(config_path).resolve() if config_path is not None else Path(__file__).resolve().parents[1] / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    for key in ("data_root", "artifacts_root"):
        config[key] = str((path.parent / config.get(key, "./" + key.removesuffix("_root"))).resolve())
    return config


def get_llm(config: dict[str, Any]):
    provider = config["llm_provider"]
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        model = _openrouter_model(config)
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=_required_key("OPENROUTER_API_KEY"),
            model=model,
            temperature=config.get("temperature", 0.2),
            max_tokens=config.get("max_tokens", 384),
            timeout=config.get("request_timeout", 60),
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.get("llm_model", "gpt-4o-mini"),
            api_key=_required_key("OPENAI_API_KEY"),
            timeout=config.get("request_timeout", 60),
            temperature=config.get("temperature", 0.2),
            max_tokens=config.get("max_tokens", 384),
        )
    raise ValueError(f"Unsupported llm_provider: {provider}")


def get_text_embeddings(config: dict[str, Any]):
    provider = config["text_emb_provider"]
    model_name = config["text_emb_model"]
    if provider == "sbert":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": config.get("embedding_device", "cpu")},
            encode_kwargs={"normalize_embeddings": config.get("normalize_embeddings", True)},
        )
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model_name, api_key=_required_key("OPENAI_API_KEY"), request_timeout=config.get("request_timeout", 60))
    raise ValueError(f"Unsupported text_emb_provider: {provider}")


def get_clip_model(config: dict[str, Any]):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.get("clip_model", "clip-ViT-B-32"), device=config.get("clip_device", "cpu"))


def validate_api_keys(config: dict[str, Any], verbose: bool = True) -> dict[str, bool]:
    keys = {
        "OPENROUTER_API_KEY": bool(os.getenv("OPENROUTER_API_KEY")),
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
    }
    if verbose:
        required = "OPENROUTER_API_KEY" if config.get("llm_provider") == "openrouter" else "OPENAI_API_KEY"
        required_keys = {required}
        if config.get("text_emb_provider") == "openai":
            required_keys.add("OPENAI_API_KEY")
        for key in sorted(required_keys):
            if not keys.get(key, False):
                print(f"Warning: {key} is not configured for the selected providers.")
        if not keys.get(required, False):
            print(f"Warning: {required} is not configured. Retrieval can still run, but LLM generation will fail.")
    return keys


def print_config_summary(config: dict[str, Any]) -> None:
    if config.get("llm_provider") == "openrouter":
        model = _openrouter_model(config)
    else:
        model = config.get("llm_model")
    print(f"LLM: {config.get('llm_provider')} / {model}")
    print(f"Text embeddings: {config.get('text_emb_provider')} / {config.get('text_emb_model')}")
    print(f"CLIP: {config.get('clip_model')}")


def _required_key(name):
    key = os.getenv(name)
    if not key or not key.strip():
        raise ValueError(f"Missing required secret: {name}")
    return key


def _openrouter_model(config):
    model = config.get('openrouter_model', 'gpt-4o-mini')
    return model if '/' in model else f"{config.get('openrouter_provider', 'openai')}/{model}"
