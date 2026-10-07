"""
Application configuration - loads from YAML param files.

CONFIGURATION POLICY:
====================
Configuration is loaded from config/params.yaml and config/models.yaml.
Secrets (API keys, DB credentials) live ONLY in .env and are loaded via os.getenv().

Supported API modes in this prototype:
- OpenRouter, using its OpenAI-compatible chat and embeddings endpoints
- OpenAI direct
"""

from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml
from loguru import logger
from dotenv import load_dotenv
from copy import deepcopy

# ========================================
# Project Paths
# ========================================

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _PROJECT_ROOT / "config"
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_PROJECT_ROOT.parents[1] / ".env")

# ========================================
# YAML Config Loading
# ========================================

def _load_yaml(filename: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    filepath = _CONFIG_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {filepath}")
    return data


def _get_nested(d: Dict, *keys, default=None):
    """Get nested dictionary value safely."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d if d is not None else default


# Load configs
_PARAMS = _load_yaml("params.yaml")
_MODELS = _load_yaml("models.yaml")

# ========================================
# Provider Configuration
# ========================================

PROVIDER = _get_nested(_PARAMS, "provider", "default", default="openrouter")
MODEL_TIER = _get_nested(_PARAMS, "provider", "tier", default="general")
OPENROUTER_BASE_URL = _get_nested(_PARAMS, "provider", "openrouter_base_url",
                                   default="https://openrouter.ai/api/v1")

# ========================================
# Model Names (from models.yaml)
# ========================================

def get_chat_model(provider: Optional[str] = None, tier: Optional[str] = None) -> str:
    """Get chat model name for specified provider and tier."""
    provider = provider or PROVIDER
    tier = tier or MODEL_TIER
    model = _get_nested(_MODELS, provider, "chat", tier)
    if provider not in {"openrouter", "openai"} or not model:
        raise ValueError(f"Unsupported chat provider/tier: {provider}/{tier}")
    return model


EMBEDDING_TIER = _get_nested(_PARAMS, "embedding", "tier", default="default")


def get_embedding_model(provider: Optional[str] = None, tier: Optional[str] = None) -> str:
    """Get embedding model name for specified provider and tier."""
    provider = provider or PROVIDER
    tier = tier or EMBEDDING_TIER
    model = _get_nested(_MODELS, provider, "embedding", tier)
    if provider not in {"openrouter", "openai"} or not model:
        raise ValueError(f"Unsupported embedding provider/tier: {provider}/{tier}")
    return model


# ========================================
# Two-model GraphRAG architecture
# ========================================
# Two roles optimised for GraphRAG:
#   General:  gpt-4o-mini via OpenRouter — routing, grading, generation
#   Strong:   gpt-4o via OpenRouter      — entity extraction, entity resolution

GENERAL_MODEL = get_chat_model(tier="general")
GENERAL_PROVIDER = PROVIDER

STRONG_MODEL = get_chat_model(tier="strong")
STRONG_PROVIDER = PROVIDER

EMBEDDING_MODEL = get_embedding_model()

# ========================================
# Embedding Dimensions
# ========================================

EMBEDDING_DIM = 1536  # Default for text-embedding-3-small
if "large" in EMBEDDING_MODEL.lower():
    EMBEDDING_DIM = 3072
elif "small" in EMBEDDING_MODEL.lower() or "ada" in EMBEDDING_MODEL.lower():
    EMBEDDING_DIM = 1536

# ========================================
# LLM Defaults
# ========================================

LLM_TEMPERATURE = _get_nested(_PARAMS, "llm", "temperature", default=0.0)
LLM_MAX_TOKENS = _get_nested(_PARAMS, "llm", "max_tokens", default=2000)
LLM_STREAMING = _get_nested(_PARAMS, "llm", "streaming", default=False)

# ========================================
# Embedding Defaults
# ========================================

EMBEDDING_BATCH_SIZE = _get_nested(_PARAMS, "embedding", "batch_size", default=100)
EMBEDDING_SHOW_PROGRESS = _get_nested(_PARAMS, "embedding", "show_progress", default=False)

# ========================================
# Neo4j Graph Database
# ========================================

NEO4J_URI = os.getenv(
    _get_nested(_PARAMS, "graph", "uri_env", default="NEO4J_URI"),
    None,
)
NEO4J_USERNAME = os.getenv(
    _get_nested(_PARAMS, "graph", "user_env", default="NEO4J_USERNAME"),
    "neo4j",
)
NEO4J_PASSWORD = os.getenv(
    _get_nested(_PARAMS, "graph", "password_env", default="NEO4J_PASSWORD"),
    None,
)
NEO4J_DATABASE = _get_nested(_PARAMS, "graph", "database", default="neo4j")

# ========================================
# Project Paths (from params.yaml)
# ========================================

DATA_DIR = _PROJECT_ROOT / _get_nested(_PARAMS, "paths", "data_dir", default="data")
RAW_DIR = _PROJECT_ROOT / _get_nested(_PARAMS, "paths", "raw_dir", default="data/raw")
PROCESSED_DIR = _PROJECT_ROOT / _get_nested(_PARAMS, "paths", "processed_dir", default="data/processed")

# ========================================
# Helper Functions
# ========================================

def get_api_key(provider: Optional[str] = None) -> Optional[str]:
    """Get API key for the specified provider."""
    provider = provider or PROVIDER
    key_map = {
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    env_var = key_map.get(provider, f"{provider.upper()}_API_KEY")
    return os.getenv(env_var)


def validate(require_llm: bool = True) -> None:
    """Validate configuration and create required directories."""
    api_key = get_api_key()
    if require_llm and not api_key:
        key_name = "OPENROUTER_API_KEY" if PROVIDER == "openrouter" else f"{PROVIDER.upper()}_API_KEY"
        raise ValueError(
            f"Missing required secret: {key_name}\n"
            f"Please add it to your .env file."
        )

    graph_credentials = get_graph_credentials()
    if not graph_credentials["uri"]:
        raise ValueError(
            "Missing required secret: NEO4J_URI\n"
            "Please add it to your .env file (e.g., neo4j+s://xxxxx.databases.neo4j.io)"
        )
    if not graph_credentials["password"]:
        raise ValueError(
            "Missing required secret: NEO4J_PASSWORD\n"
            "Please add it to your .env file."
        )

    for dir_path in [DATA_DIR, RAW_DIR, PROCESSED_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


def dump() -> None:
    """Print all active non-secret configuration values for debugging."""
    logger.info("\n" + "=" * 60)
    logger.info("CONFIGURATION (NON-SECRETS ONLY)")
    logger.info("=" * 60)

    logger.info("\n Provider:")
    logger.info(f"   Provider: {PROVIDER}")
    logger.info(f"   Model Tier: {MODEL_TIER}")
    logger.info(f"   General Model: {GENERAL_MODEL}")
    logger.info(f"   Strong Model: {STRONG_MODEL}")
    logger.info(f"   Embedding Model: {EMBEDDING_MODEL}")
    logger.info(f"   Embedding Dimensions: {EMBEDDING_DIM}")

    logger.info("\n Neo4j Graph:")
    logger.info(f"   URI: {'Set' if NEO4J_URI else 'Not set'}")
    logger.info(f"   Username: {NEO4J_USERNAME}")
    logger.info(f"   Password: {'Set' if NEO4J_PASSWORD else 'Not set'}")
    logger.info(f"   Database: {NEO4J_DATABASE}")

    logger.info("\n Paths:")
    logger.info(f"   Data: {DATA_DIR}")
    logger.info(f"   Raw: {RAW_DIR}")
    logger.info(f"   Processed: {PROCESSED_DIR}")

    logger.info("\n" + "=" * 60 + "\n")


def get_all_models() -> Dict[str, Any]:
    """Return all available models from models.yaml."""
    return deepcopy(_MODELS)


def get_config() -> Dict[str, Any]:
    """Return full config dictionary."""
    return deepcopy(_PARAMS)


def get_graph_credentials() -> Dict[str, Any]:
    """Read credentials when constructing a client, including notebook env changes."""
    return {
        "uri": os.getenv(_get_nested(_PARAMS, "graph", "uri_env", default="NEO4J_URI")),
        "username": os.getenv(_get_nested(_PARAMS, "graph", "user_env", default="NEO4J_USERNAME"), "neo4j"),
        "password": os.getenv(_get_nested(_PARAMS, "graph", "password_env", default="NEO4J_PASSWORD")),
        "database": NEO4J_DATABASE,
    }
