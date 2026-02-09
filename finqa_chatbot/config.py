"""Application configuration via Pydantic Settings."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration — reads from environment / .env file."""

    # LLM
    openai_api_key: str = ""
    model_name: str = "gpt-5-nano"
    embed_model: str = "text-embedding-3-small"
    temperature: float = 0.0

    # Self-consistency
    num_candidates: int = 5
    candidate_temperature: float = 0.7

    # DeALOG scheduler
    max_rounds: int = 3

    # Retrieval
    num_few_shot: int = 5
    retrieval_topn: int = 3

    # Paths
    dataset_dir: Path = Path(__file__).resolve().parent.parent / "data"
    output_dir: Path = Path(__file__).resolve().parent.parent / "output"

    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "finqa-chatbot"

    model_config = {
        "env_file": [
            str(Path(__file__).resolve().parent.parent / ".env"),
            str(Path(__file__).resolve().parent.parent.parent / ".env"),
        ],
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
