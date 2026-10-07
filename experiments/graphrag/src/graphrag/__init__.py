"""Reusable infrastructure for the Neo4j GraphRAG experiment."""

from graphrag.config import validate, dump, get_config, get_api_key
from graphrag.neo4j_client import Neo4jClient, get_neo4j_client


def get_general_llm(*args, **kwargs):
    from graphrag.llm_services import get_general_llm as factory
    return factory(*args, **kwargs)


def get_strong_llm(*args, **kwargs):
    from graphrag.llm_services import get_strong_llm as factory
    return factory(*args, **kwargs)


def get_embeddings(*args, **kwargs):
    from graphrag.llm_services import get_embeddings as factory
    return factory(*args, **kwargs)

__all__ = [
    "validate", "dump", "get_config", "get_api_key",
    "get_general_llm", "get_strong_llm", "get_embeddings",
    "Neo4jClient", "get_neo4j_client",
]
