"""Shared utilities for LangGraph examples.

This module is independent from the root SDK (the_agents_playbook).
Utilities are copied/adapted here so langgraph-examples can run standalone.
"""

from .settings import Settings, settings, validate_config, get_openai_llm, get_anthropic_llm
from .vectors import cosine_similarity, normalize
from .compactor import SessionCompactor

__all__ = [
    "Settings",
    "settings",
    "validate_config",
    "get_openai_llm",
    "get_anthropic_llm",
    "cosine_similarity",
    "normalize",
    "SessionCompactor",
]
