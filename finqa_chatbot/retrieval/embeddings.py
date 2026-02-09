"""OpenAI embedding-based retrieval."""

from __future__ import annotations

import time

import numpy as np
from openai import OpenAI

from ..config import get_settings


def _get_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key)


def openai_embed(texts: list[str], max_retries: int = 3) -> list[list[float]]:
    """Get embeddings from OpenAI API with retry."""
    client = _get_client()
    model = get_settings().embed_model
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e


def openai_retrieve(
    question: str,
    facts: list[tuple[str, str]],
    topn: int = 3,
) -> list[tuple[str, str]]:
    """Return the top-n facts most similar to *question* via OpenAI embeddings."""
    if not facts:
        return []
    docs = [text for _, text in facts]
    all_texts = [question] + docs
    embeddings = openai_embed(all_texts)
    q_emb = np.array(embeddings[0])
    doc_embs = np.array(embeddings[1:])
    sims = doc_embs @ q_emb / (
        np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(q_emb) + 1e-10
    )
    ranked_indices = np.argsort(-sims)
    return [facts[i] for i in ranked_indices[:topn]]
